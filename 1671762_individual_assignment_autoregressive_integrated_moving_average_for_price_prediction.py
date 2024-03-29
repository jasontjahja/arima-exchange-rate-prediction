# -*- coding: utf-8 -*-
"""1671762_Individual_Assignment_AutoRegressive_Integrated_Moving_Average_for_Price_Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UWikxjuJzREOEpJ_BXeTL6zigX5xjG24

# Data Collection from Yahoo Finance
"""

pip install yfinance

"""yfinance library fetches data from Yahoo Finance, and the availability of certain currency exchange rates may vary. Additionally, it's always a good practice to check the terms of use for the data provider to ensure compliance with their policies"""

import yfinance as yfsource

"""This code uses the yf.download function to download historical data for the specified ticker symbol (USDIDR=X). Make sure to adjust the start and end parameters according to your desired time range."""

# Specify the ticker symbol for the USD to IDR exchange rate
usd_to_idr_ticker = "USDIDR=X"

# Download historical data for the exchange rate
usd_idr_exchange_rate_data = yfsource.download(usd_to_idr_ticker, start="2013-1-1", end="2023-12-16")

# Print the last available exchange rate
usd_idr_last_exchange_rate = usd_idr_exchange_rate_data["Close"].iloc[-1]
print(f"Last USD to IDR exchange rate: {usd_idr_last_exchange_rate} IDR")

"""Making sure the right rate quotation is downloaded"""

usd_idr_exchange_rate_data

# Specify the ticker symbol for BTC to USD
btc_to_usd_ticker = "BTC-USD"

# Download historical data for BTC to USD
btc_to_usd_data = yfsource.download(btc_to_usd_ticker, start="2019-11-1", end="2023-12-16")

# Print the last available BTC to USD closing price
last_btc_price = btc_to_usd_data["Close"].iloc[-1]
print(f"Last BTC to USD closing price: ${last_btc_price:.2f}")

"""In this code, BTC-USD is the ticker symbol for Bitcoin to USD on Yahoo Finance. Adjust the start and end parameters according to desired time range."""

btc_to_usd_data

"""# Data cleaning and differencing for time series analysis

**Remove rows containing null values**
"""

# Original lengths before cleaning N/A values
original_length_usd_idr = len(usd_idr_exchange_rate_data)
original_length_btc_usd = len(btc_to_usd_data)

# Drop N/A values
usd_idr_exchange_rate_data = usd_idr_exchange_rate_data.dropna()
btc_to_usd_data = btc_to_usd_data.dropna()

# Lengths after cleaning N/A values
cleaned_length_usd_idr = len(usd_idr_exchange_rate_data)
cleaned_length_btc_usd = len(btc_to_usd_data)

# Print the results
print("Original Length (USD to IDR):", original_length_usd_idr)
print("Cleaned Length (USD to IDR):", cleaned_length_usd_idr)
print("Rows Removed (USD to IDR):", original_length_usd_idr - cleaned_length_usd_idr)

print("\nOriginal Length (BTC to USD):", original_length_btc_usd)
print("Cleaned Length (BTC to USD):", cleaned_length_btc_usd)
print("Rows Removed (BTC to USD):", original_length_btc_usd - cleaned_length_btc_usd)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.express as px

"""**Here's a brief description of the libraries and modules being imported:**



*   numpy (np): A library for numerical operations in Python.
*   pandas (pd): A data manipulation and analysis library.
*   matplotlib.pyplot (plt): A plotting library for creating visualizations.
*   math: The Python math module for mathematical operations.
*   statsmodels.tsa.arima.model.ARIMA: The ARIMA (AutoRegressive Integrated Moving Average) model from the statsmodels library for time series analysis.
*   sklearn.metrics.mean_squared_error: A function for calculating the mean squared error.
*   sklearn.metrics.mean_absolute_error: A function for calculating the mean absolute error.
*   plotly.express: A plotting library that generated avdanced and more interactive visualisation

**Making sure the data stationary for time series analysis**
"""

from statsmodels.tsa.stattools import adfuller

def difference_and_adf_test(series, title):
    # Perform differencing
    diff_series = series.diff().dropna()

    # Perform ADF test
    result = adfuller(diff_series)
    print(f'ADF Statistic for {title}: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values: {result[4]}')

# Perform differencing and ADF test for USD to IDR data
difference_and_adf_test(usd_idr_exchange_rate_data['Adj Close'], 'USD to IDR Exchange Rate')

print("")

# Perform differencing and ADF test for BTC to USD data
difference_and_adf_test(btc_to_usd_data['Close'], 'BTC to USD Rate')

"""This code defines a function (difference_and_adf_test) that performs differencing, plots the differenced series, and conducts the ADF test. The ADF test is used to check for stationarity.

# Plot for quotation of USD-IDR and BTC-USD pair
"""

import plotly.express as px

# Plot USD to IDR exchange rate with dark background
fig_usd_to_idr = px.line(usd_idr_exchange_rate_data, x=usd_idr_exchange_rate_data.index, y="Adj Close", title="USD to IDR Exchange Rate")
fig_usd_to_idr.update_layout(template="plotly_dark")

# Plot BTC to USD rate with dark background and golden line color
fig_btc_to_usd = px.line(btc_to_usd_data, x=btc_to_usd_data.index, y="Adj Close", title="BTC to USD Rate", line_shape="linear")
fig_btc_to_usd.update_traces(line=dict(color='gold'))

fig_btc_to_usd.update_layout(template="plotly_dark")

# Show the plots
fig_usd_to_idr.show()
fig_btc_to_usd.show()

"""# Split Data into Training and Testing for ARIMA model prediction

Determine Training and Testing Data Splits:

* to_row_usd_idr: Calculates the index that corresponds to 90% of the length of the USD to IDR exchange rate data.

* to_row_btc_usd: Calculates the index that corresponds to 90% of the length of the BTC to USD rate data.
"""

to_row_usd_idr = int(len(usd_idr_exchange_rate_data)*0.9)

training_data_usd_idr = list(usd_idr_exchange_rate_data[0:to_row_usd_idr]['Adj Close'])

testing_data_usd_idr = list(usd_idr_exchange_rate_data[to_row_usd_idr:]['Adj Close'])

to_row_btc_usd = int(len(btc_to_usd_data)*0.9)

training_data_btc_usd = list(btc_to_usd_data[0:to_row_btc_usd]['Adj Close'])

testing_data_btc_usd = list(btc_to_usd_data[to_row_btc_usd:]['Adj Close'])

"""Create Training and Testing Datasets:

* training_data_usd_idr: Extracts the adjusted closing prices (Adj Close) of the USD to IDR exchange rate from the beginning of the dataset up to the 90% split index.

* testing_data_usd_idr: Extracts the adjusted closing prices (Adj Close) of the USD to IDR exchange rate from the 90% split index to the end of the dataset.

* training_data_btc_usd: Extracts the adjusted closing prices (Adj Close) of the BTC to USD rate from the beginning of the dataset up to the 90% split index.

* testing_data_btc_usd: Extracts the adjusted closing prices (Adj Close) of the BTC to USD rate from the 90% split index to the end of the dataset.
"""

plt.figure(figsize=(10,6))
plt.title('USD IDR Prediction')
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(usd_idr_exchange_rate_data[0:to_row_usd_idr]['Adj Close'], 'green', label='Training data')
plt.plot(usd_idr_exchange_rate_data[to_row_usd_idr:]['Adj Close'], 'blue', label='Testing data')
plt.legend()

plt.figure(figsize=(10,6))
plt.title('BTC USD Prediction')
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(btc_to_usd_data[0:to_row_btc_usd]['Adj Close'], 'green', label='Training data')
plt.plot(btc_to_usd_data[to_row_btc_usd:]['Adj Close'], 'blue', label='Testing data')
plt.legend()

"""# Figuring out the best ARIMA model order"""

pip install pmdarima

"""This code below automates the process of selecting the best ARIMA model for the given USD to IDR exchange rate data by using the auto_arima function from the pmdarima library. The selected model's summary provides insights into the chosen hyperparameters and the overall fit of the model to the data."""

from pmdarima import auto_arima
# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

stepwise_fit = auto_arima(usd_idr_exchange_rate_data['Adj Close'],
                          suppress_warnings=True)

stepwise_fit.summary()

from pmdarima import auto_arima
# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

stepwise_fit = auto_arima(btc_to_usd_data['Adj Close'],
                          suppress_warnings=True)

stepwise_fit.summary()

"""# Training and Testing Data using ARIMA"""

model_predictions_usd_idr = []
n_test_observer_usd_idr = len(testing_data_usd_idr)

model_predictions_btc_usd = []
n_test_observer_btc_usd = len(testing_data_btc_usd)

"""This code is initializing empty lists model_predictions_usd_idr and model_predictions_btc_usd to store the predicted values of two financial time series: USD to IDR exchange rate (usd_idr_exchange_rate_data) and BTC to USD rate (btc_to_usd_data).

Additionally, it defines variables n_test_observer_usd_idr and n_test_observer_btc_usd to represent the number of observations in the testing datasets for the respective financial time series.

**ARIMA Model Fitting:**

* model = ARIMA(training_data_usd_idr, order=(2, 1, 3)): Initializes an ARIMA model with a specific set of hyperparameters (order = (2, 1, 3)).

* model_fit = model.fit(): Fits the ARIMA model to the training data (training_data_usd_idr). This involves estimating the model parameters based on historical data.
"""

for i in range(n_test_observer_usd_idr):
    model = ARIMA(training_data_usd_idr, order=(4, 1, 5))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    model_predictions_usd_idr.append(yhat)
    actual_test_value = testing_data_usd_idr[i]
    training_data_usd_idr.append(actual_test_value)

"""**Model Prediction:**

* output = model_fit.forecast(): Makes a one-step forecast using the fitted ARIMA model. The forecast() method is used to predict the next value in the time series.

* yhat = output[0]: Extracts the predicted value from the forecast output.

**Updating Predictions and Training Data:**

* model_predictions_usd_idr.append(yhat): Appends the predicted value (yhat) to the list model_predictions_usd_idr. This list is being used to store the model predictions.

* actual_test_value = testing_data_usd_idr[i]: Retrieves the actual value from the testing dataset corresponding to the current iteration.

* training_data_usd_idr.append(actual_test_value): Updates the training dataset by appending the actual test value. This simulates a scenario where the model is retrained with the actual observation from the testing set, and the process repeats.
"""

for i in range(n_test_observer_btc_usd):
    model = ARIMA(training_data_btc_usd, order=(2, 1, 2))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    model_predictions_btc_usd.append(yhat)
    actual_test_value = testing_data_btc_usd[i]
    training_data_btc_usd.append(actual_test_value)

"""The code above also have the same functions like in the usd idr iteration"""

print(len(model_predictions_usd_idr))
print(len(testing_data_usd_idr))

print(len(model_predictions_btc_usd))
print(len(testing_data_btc_usd))

"""The code above ensures that model predictions and testing data have the same length of array, this is important to make sure it can be plotted accurately

# Plot for comparison between ARIMA Prediction and actual Adjusted Close Price for USD-IDR and BTC-USD
"""

import plotly.express as px

# Assuming you have 'date_range', 'model_predictions_usd_idr', and 'testing_data_usd_idr' available
date_range = usd_idr_exchange_rate_data[to_row_usd_idr:-1].index  # Ensure the same length for date_range

# Create a DataFrame for plotting
plot_data_usd_idr = pd.DataFrame({
    'Date': date_range,
    'USD IDR Predicted Rate': model_predictions_usd_idr[:-1],
    'USD IDR Actual Price': testing_data_usd_idr[:-1]
})

# Plot using Plotly Express
fig = px.line(plot_data_usd_idr, x='Date', y=['USD IDR Predicted Rate', 'USD IDR Actual Price'], title='USD IDR Prediction')
fig.update_layout(template="plotly_dark")
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Price')
fig.show()

plt.figure(figsize=(15,9))
plt.grid(True)

date_range = usd_idr_exchange_rate_data[to_row_usd_idr:-1].index  # Ensure the same length for date_range

plt.plot(date_range, model_predictions_usd_idr[:-1], color='blue', marker='o', linestyle='dashed', label='USD IDR Predicted Rate')
plt.plot(date_range, testing_data_usd_idr[:-1], color='red', label='USD IDR Actual Price')

plt.title('USD IDR Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

import plotly.express as px

# Assuming you have 'date_range', 'model_predictions_btc_usd', and 'testing_data_btc_usd' available
date_range = btc_to_usd_data[to_row_btc_usd:].index

# Create a DataFrame for plotting
plot_data_btc_usd = pd.DataFrame({
    'Date': date_range,
    'BTC USD Predicted Rate': model_predictions_btc_usd,
    'BTC USD Actual Price': testing_data_btc_usd
})

# Plot using Plotly Express
fig = px.line(plot_data_btc_usd, x='Date', y=['BTC USD Predicted Rate', 'BTC USD Actual Price'], title='BTC USD Prediction')
fig.update_layout(template="plotly_dark")
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Price')
fig.show()

plt.figure(figsize=(15,9))
plt.grid(True)

date_range = btc_to_usd_data[to_row_btc_usd:].index

plt.plot(date_range, model_predictions_btc_usd, color = 'blue', marker = 'o', linestyle = 'dashed', label = 'BTC USD Predicted Rate')
plt.plot(date_range, testing_data_btc_usd, color = 'red', label = 'BTC Actual Price')

plt.title('BTC USD Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

"""# Price Prediction

**ARIMA Model Fitting:**

* model = ARIMA(usd_idr_exchange_rate_data['Adj Close'], order=(2, 1, 3)): Initializes an ARIMA model with the specified order (p=2, d=1, q=3) and fits it to the historical adjusted closing prices of the USD to IDR exchange rate (usd_idr_exchange_rate_data['Adj Close']).

* model_fit = model.fit(): Fits the ARIMA model to the historical data, estimating the model parameters.

**Forecasting Future Values:**

* future_steps = 20: Specifies the number of future steps to forecast. In this case, it's set to 20, meaning the code will forecast the next 20 values.

* future_dates = pd.date_range(start="2023-12-17", periods=future_steps, freq='D'): Generates a sequence of future dates starting from the given date ("2023-12-17") with a daily frequency.

* future_predictions = model_fit.get_forecast(steps=future_steps).predicted_mean: Uses the get_forecast method to generate future forecasts. The predicted_mean attribute contains the predicted values for the specified number of future steps.

**Printing the Future Predictions:**

* future_predictions_df: Creates a DataFrame (future_predictions_df) containing the forecasted dates and corresponding ARIMA predictions.

* print(future_predictions_df): Prints the DataFrame, displaying the future dates and the corresponding ARIMA predictions.
"""

model = ARIMA(usd_idr_exchange_rate_data['Adj Close'], order=(4, 1, 5))
model_fit = model.fit()

# Forecast future values
future_steps = 20  # Adjust the number of future steps as needed
future_dates = pd.date_range(start="2023-12-17", periods=future_steps, freq='D')  # Daily frequency
future_predictions = model_fit.get_forecast(steps=future_steps).predicted_mean

# Print the date and ARIMA predictions
future_predictions_df = pd.DataFrame({
    'Date': future_dates,
    'ARIMA Predictions': future_predictions
})

print(future_predictions_df)

"""the codes below have the same function like code above, but for predicting the price of BTC USD in the next 20 days"""

model2 = ARIMA(btc_to_usd_data['Adj Close'], order=(2, 1, 2))
model2_fit = model2.fit()

# Forecast future values
future_steps = 20  # Adjust the number of future steps as needed
future_dates = pd.date_range(start="2023-12-17", periods=future_steps, freq='D')  # Daily frequency
future_predictions2 = model2_fit.get_forecast(steps=future_steps).predicted_mean

# Print the date and ARIMA predictions
future_predictions_df2 = pd.DataFrame({
    'Date': future_dates,
    'ARIMA Predictions': future_predictions2
})

print(future_predictions_df2)

"""the code below provides a qualitative analysis of the trend in the USD to IDR exchange rate and BTC to USD based on the ARIMA model's predictions for the next 20 days.

**Extracting First and Last Predicted Exchange Rates:**

* first_prediction: Extracts the first predicted exchange rate from the DataFrame.

* last_prediction: Extracts the last predicted exchange rate from the DataFrame.

# Trend conclusion based on ARIMA predictions
"""

# Extract the first and last predicted exchange rates
first_prediction = future_predictions_df['ARIMA Predictions'].iloc[0]
last_prediction = future_predictions_df['ARIMA Predictions'].iloc[-1]

# Formulate the trend conclusion for USD to IDR exchange rate
if last_prediction > first_prediction:
    trend_conclusion = "Based on ARIMA prediction model by looking next 20 days of price prediction, the market for USD IDR is uptrending. Consider buying or holding USD"
elif last_prediction < first_prediction:
    trend_conclusion = "Based on ARIMA prediction model by looking next 20 days of price prediction, the market for USD IDR is downtrending. Consider selling USD"
else:
    trend_conclusion = "Based on ARIMA prediction model by looking next 20 days of price prediction, the market for USD IDR is sideways. Consider holding either USD or IDR."

# Print the trend conclusion
print(trend_conclusion)

"""**Formulating the Trend Conclusion:**

* Checks whether the last predicted exchange rate is higher (>) or lower (<) than
the first predicted exchange rate.

* Formulates a trend conclusion message based on the direction of the trend (uptrend, downtrend, or sideways).
"""

# Extract the first and last predicted prices
first_prediction = future_predictions_df2['ARIMA Predictions'].iloc[0]
last_prediction = future_predictions_df2['ARIMA Predictions'].iloc[-1]

# Formulate the trend conclusion
if last_prediction > first_prediction:
    trend_conclusion = "Based on ARIMA prediction model by looking next 20 days of price prediction, the market for BTC USD is uptrending. Consider buying or holding BTC USD asset."
elif last_prediction < first_prediction:
    trend_conclusion = "Based on ARIMA prediction model by looking next 20 days of price prediction, the market for BTC USD is downtrending. Consider selling or converting to USD asset."
else:
    trend_conclusion = "Based on ARIMA prediction model by looking next 20 days of price prediction, the market for BTC USD is sideways. Consider keeping or holding BTC USD asset."

# Print the trend conclusion
print(trend_conclusion)

"""# Model Performance Analysis"""

mape = np.mean(np.abs(np.array(model_predictions_usd_idr) - np.array(testing_data_usd_idr))/np.abs(testing_data_usd_idr))
print('Mean Absolute Percentage Error for usd-idr predictions:  ' + str(mape))

mape2 = np.mean(np.abs(np.array(model_predictions_btc_usd) - np.array(testing_data_btc_usd))/np.abs(model_predictions_btc_usd))
print('Mean Absolute Percentage Error for btc-usd predictions:  ' + str(mape2))