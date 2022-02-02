import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as k
from matplotlib import pyplot as plt
import os
from datetime import datetime, timedelta
import time
import pickle
from tqdm.auto import tqdm


def load_pkl(filename):
    """Loads object from pickle file"""
    try:
        with open(filename, "rb") as fh:
            obj = pickle.load(fh)
        return obj
    except Exception:
        return None

def save_pkl(obj, filename):
    """Saves pickled object to .pkl file"""
    if filename.endswith(".pkl") is False:
        filename += ".pkl"
    with open(filename, "wb") as fh:
        pickle.dump(obj, fh)

def train_test_split(Data, split=0.9):
    """
    Splits dataframe into training and testing sets.

    Args:
        Data: Pandas dataframe to be split.
        split: float - between 0 and 1; decides portion of
        training split.

    Return: training_split, testing_split
    """
    cut = int(Data.shape[0]*split)
    if type(Data) is np.ndarray:
        return Data[:cut, ...], Data[cut:, ...]
    return Data.iloc[:cut, :], Data.iloc[cut+1:, :]

def save_resampled_dataframe(ext="", load_folder="/content/drive/MyDrive/", save_folder="/content/drive/MyDrive/BTC_Resampled/", offset="H"):

    path = save_folder+ext+"BTC_{}/".format(offset)

    if not os.path.exists(save_folder):
        os.makedirs(path)

    data, cols_to_norm = load_BTC_data(folder_path=load_folder, offset=offset)

    norm_obj = Norm(Data=data, cols_to_norm=cols_to_norm)
    norm_obj.save(path)

    train_data, test_data = train_test_split(data, split=0.98)

    print("Nans: ", train_data.shape[0] - train_data.dropna(axis=0, how="any").shape[0])

    train_data.to_csv(path+"train_data.csv")
    save_pkl(train_data.shape[0], path+"total_size.pkl")
    save_pkl(test_data.shape[0], path+"test_total_size.pkl")
    test_data.to_csv(path+"test_data.csv")

    return path, test_data

def read_data(path, batch_size=32):
    for batch in pd.read_csv(path, chunksize=batch_size):
        yield batch, int(batch.shape[0])

def load_BTC_data(df=None, folder_path="/content/drive/MyDrive/", offset="H"):
    if df is None:
        csv_path = folder_path+"bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv"
        df = pd.read_csv(csv_path)

    df.dropna(axis=0, how="any", inplace=True)
    df["Timestamp"] = df["Timestamp"].map(datetime.fromtimestamp)
    df.set_index("Timestamp", inplace=True)

    mp = {"High": "max", "Low": "min",
          "Close": "last", "Open": "mean",
          "Volume_(BTC)": "sum", "Volume_(Currency)": "sum"}
    cols_to_norm = mp.keys()
    df = df.resample(offset).agg(mp)
    df.interpolate(method='polynomial', order=df.shape[-1]-1, inplace=True)
    df.reset_index(inplace=True)
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
    df["DayOfMonth"] = df["Timestamp"].dt.day


    return df, cols_to_norm


class Norm():

    def __init__(self, Data=None, load_folder=False, cols_to_norm=None):
        if load_folder:
            self.min = load_pkl(load_folder+"min_vector.pkl")
            self.max = load_pkl(load_folder+"max_vector.pkl")
        else:
            self.min = Data[cols_to_norm].min()
            self.max = Data[cols_to_norm].max()

    def normalize(self, dataframe):
        """
        Min - Max normalization function.

        Args:
            dataframe: Pandas dataframe.

        Return:
            Normalized dataframe.
        """
        idx = self.min.index.to_list()
        dataframe[idx] = (dataframe[idx] - self.min[idx]) / (self.max[idx] - self.min[idx])
        return dataframe

    def inverse_norm(self, data, key="Close", isnumpy=False):
        """
        Min - Max Inverse normalization function.

        Args:
            dataframe: Pandas dataframe.

        Return:
            Normalized dataframe inverted to true value.
        """
        if isnumpy:
            return (data*(self.max[key]-self.min[key]))+self.min[key]
        else:
            data[key] = (data[key]*(self.max[key]-self.min[key]))+self.min[key]
        return data

    def save(self, folder):
        """Saves min and max vectors."""

        exists = os.path.exists(folder)
        if not exists:
            os.makedirs(folder)

        save_pkl(self.min, folder+"min_vector.pkl")
        save_pkl(self.max, folder+"max_vector.pkl")

class Generator():

    def __init__(self, path, num_batches, batch_size=32):
        """
        Generator class constructor.

        Attributes:
            path: path to folder containing data.
            batch_size: Number of points to load from data in
            each call to the iterator.
            num_batches: Total number of batches to generate.
        """
        self.path = path
        self.batch_size = batch_size
        self.num_batches = num_batches

    def get_data(self, path, skiprows=None):
        """
        Retrieves dataset.

        Args:
            path: Path to the dataset.
            skiprows: List containing indices of rows to skip
            in the dataset.

        Return: Input data, target data
        """
        df = pd.read_csv(path, skiprows=skiprows)
        df.drop(columns=['Unnamed: 0', 'Timestamp'], inplace=True)
        norm_obj = Norm(load_folder=self.path)
        df = norm_obj.normalize(df)
        targets = df.pop("Close").to_numpy()
        return df.to_numpy(), targets

    def generator(self, file="train_data.csv", window_size=24):
        """
        Iterable method for generating batches of data
        via sliding window.

        Args:
            file: File name where data is located.

        yield: Input variables, targets
        """
        train_data, targets = self.get_data(self.path+file)

        for i in range(targets.size-window_size):
            data = np.concatenate((train_data[i:i+window_size], targets[i:i+window_size][..., np.newaxis]), axis=1)
            target = targets[i+window_size]
            yield data[np.newaxis, ...], target[np.newaxis, ...]

    def data_batch_gen(self, shuffle=True, file="train_data.csv", window_size=24):
        gen = self.generator(file, window_size=window_size)

        for i in range(self.num_batches):
            data_batch, target_batch = [], []

            for j in range(self.batch_size):
                data, target = next(gen)
                data_batch.append(data)
                target_batch.append(target)
            data_batch = np.concatenate(data_batch)
            target_batch = np.concatenate(target_batch)

            if shuffle:
                shuff = np.arange(target_batch.shape[0])
                shuff = np.random.choice(shuff, shuff.size, replace=False)
                data_batch = data_batch[shuff]
                target_batch = target_batch[shuff]
            yield data_batch, target_batch


class BTC_Forecasting():

    def __init__(self, path, input_dim=None, output_dim=None, nodes=None, lr=0.001, window_size=24, batch_size=32, load_model=False):
        """
        Class constructor for BTC forecasting model.

        Attributes:
            path: Path to folder containing BTC data.
            model: LSTM Recurrent Neural Network.
            cell_states: Latest internal cell state
            representation of the RNN.
            h_states: Latest internal hidden state
            representation of the RNN.
            total_size: Total number of points in the dataset.
            norm_obj: Custom Normalization object.
        """

        self.path = path
        self.total_size = load_pkl(path+"total_size.pkl")
        self.total_test_size = load_pkl(path+"test_total_size.pkl")
        self.norm_obj = Norm(load_folder=path)
        self.window_size = window_size
        self.batch_size = batch_size
        self.opt = k.optimizers.Adam(learning_rate=lr)

        if load_model:
            self.model = k.models.load_model(path+"trained_model")
        else:
            self.model = self.lstm_rnn(
                input_dim, window_size,
                output_dim, nodes
                )

    def lstm_rnn(self, input_dim, window_size, output_dim, nodes):
        """
        Method for generating and LSTM Recurrent Neural Network.

        Args:
            input_dim: Dimension of the input data.
            window_size: Length of the sequences.
            output_dim: Dimension of the neural net output.
            nodes: Nodes in the LSTM layer.

        Return: model
            mdl: LSTM Recurrent Neural Network.
        """
        model = k.models.Sequential([
            k.Input(shape=(window_size, input_dim)),
            k.layers.LSTM(nodes, return_sequences=True, name="LSTM_1"),
            k.layers.Dropout(0.2),
            k.layers.LSTM(nodes, name="LSTM_2"),
            k.layers.Dropout(0.2),  
            k.layers.Dense(units=1)
        ])
        model.compile(optimizer=self.opt, loss='mean_squared_error')

        return model

    def train_model(self, epochs, save=False):
        """
        Trains the RNN on the training dataset.

        Args:
            epochs: Number of total epochs to train.
            save: File name for saving model.
        """

        num_batches = int((self.total_size - self.window_size) / self.batch_size) - 1
        try:
            if (self.total_size - self.window_size) % num_batches != 0:
                num_batches += 1
        except:
            pass

        Gen = Generator(self.path, num_batches, self.batch_size)

        history = []

        for i in tqdm(range(epochs)):

            verbose = True if i % 10 == 0 else False
            train_gen = Gen.data_batch_gen(shuffle=True)
            hist = self.model.fit_generator(
                generator=train_gen, steps_per_epoch=num_batches,
                epochs=1, verbose=verbose
                )
            history.append(hist.history)

        if save:
            self.model.save(self.path+save)

        return history


    def plot(self, prediction, targets, plot_name="BTC Closing Price Predictions", save_file=False):
        """
        Plots model predictions and the true target values.

        Args:
            prediction: Models sequence predictions.
            targets: True target values of the sequence.
            plot_name: Title of the plot.
            save_file: File name for saving plot, else False
        """

        time_steps = np.arange(targets.size)
        plt.figure(figsize=(14, 10))
        plt.plot(time_steps, targets, marker='.', label="actual")
        plt.plot(time_steps, prediction, 'r', label="prediction")
        plt.tight_layout()
        plt.subplots_adjust(left=0.07)
        plt.ylabel('Closing Price', size=15)
        plt.xlabel('Time step', size=15)
        plt.suptitle(plot_name)
        plt.legend(fontsize=15)
        if save_file:
            plt.savefig(self.path+save_file, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def predict_testset(self, fit=True, fit_lr=False, fit_epochs=3, plot_ext=""):
        """
        Script for testing model performance on the test set.
        Predicts and fits each test set point at every step.

        Args:
            window_size: Size of leading sequence to predict before
            each point.
            fit: Boolean, will fit the true value to the model
            after each prediction.
            fit_lr: Float or False; if float, the model optimizer
            will use this parameter for resetting the learning rate.
            fit_epochs: Number of epochs to fit each true point
            after each prediction.

        Return: Mean Absolute Error
        """

        if fit_lr:
            self.opt.learning_rate.assign(fit_lr)

        num_batches = int((self.total_test_size - self.window_size) / self.batch_size)
        try:
            if (self.total_test_size - self.window_size) % num_batches != 0:
                num_batches += 1
        except:
            pass

        Gen = Generator(self.path, num_batches, self.batch_size)

        data_gen = Gen.generator(file="test_data.csv")

        predictions = []
        targets = []
        for i in tqdm(range(self.total_test_size-self.window_size)):
            data, target = next(data_gen)
            predictions.append(self.model.predict(data))
            targets.append(target)

            if fit:
                self.model.fit(data, target, verbose=False, epochs=fit_epochs)

        targets = np.concatenate(targets)
        prediction = np.concatenate(predictions)
        prediction = self.norm_obj.inverse_norm(prediction, isnumpy=True)
        targets = self.norm_obj.inverse_norm(targets, isnumpy=True)

        prediction, targets = prediction.flatten(), targets.flatten()
        cut = int(targets.size * .5)

        self.plot(
            prediction, targets,
            plot_name="BTC Test Set Predictions",
            save_file=plot_ext+"TestSet.jpg"
            )
        self.plot(
            prediction[cut:], targets[cut:],
            plot_name="BTC Test Set Predictions",
            save_file=plot_ext+"TestSet_end.jpg"
            )
        return self.MAE(targets, prediction)

    def plot_history(self, history, save_file="training_loss.jpg"):
        time_steps = np.arange(history.size)
        plt.figure(figsize=(12, 8))
        plt.plot(time_steps, history, 'r', label="Loss")
        plt.tight_layout()
        plt.subplots_adjust(left=0.07)
        plt.ylabel('Mean Squared Error', size=15)
        plt.xlabel('Epoch', size=15)
        plt.suptitle("Loss During Training")
        plt.legend(fontsize=15)
        if save_file:
            plt.savefig(self.path+save_file)
        else:
            plt.show()

    def MAE(self, Y_true, Y_pred):
        return np.absolute(np.subtract(Y_true, Y_pred)).mean()

    def get_data(self, path, skiprows=None):
        """
        Retrieves dataset.

        Args:
            path: Path to the dataset.
            skiprows: List containing indices of rows to skip
            in the dataset.

        Return: Input data, target data
        """
        df = pd.read_csv(path, skiprows=skiprows)
        df.drop(columns=['Unnamed: 0', 'Timestamp'], inplace=True)
        df = self.norm_obj.normalize(df)
        targets = df.pop("Close").to_numpy()
        return df.to_numpy(), targets
