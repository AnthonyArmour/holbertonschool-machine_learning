# Bitcoin Closing Price Model Using an LSTM Recurrent Neural Network

## Task

This code can be used for training sn LSTM network for predicting the closing price of bitcoin. I intend to improve the codes documentation as soon as I have the time. I'll leave a script example, a link for retrieving the bitcoin data, and a link to a blog post I wrote that covers the project.

### [Bitstamp Data](https://www.kaggle.com/mczielinski/bitcoin-historical-data "Bitstamp Data")

### [Time Series Forecasting Using an LSTM Recurrent Neural Network](https://docs.google.com/document/d/1n-3OD8hzZtw3uh0kmm7c2z3PbSFf51io-_69bG_3-YY/edit?usp=sharing "Time Series Forecasting Using an LSTM Recurrent Neural Network")

``` python
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
```