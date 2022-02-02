from forecast_btc import save_resampled_dataframe, BTC_Forecasting

# save_resampled_dataframe takes two additional parameters
# load_folder = path of folder containing bitcoin data csv
# save_folder = path with new folder name where you want the data
# to be saved.

path = save_resampled_dataframe(offset="H")
input_dim =	8
output_dim = 1
nodes =	50
batch_size = 32
epochs = 64
plot_file = "test_performace.jpg"


TS = BTC_Forecasting(
    path, input_dim, output_dim, nodes,
    load_model=False, batch_size=batch_size
    )

history = TS.train_model(epochs, save=False)

error = TS.predict_testset()

print("Performance Error: ", error)