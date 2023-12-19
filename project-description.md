# Currency Forecasting for Better Cash Reserve Management for Financial Managers in International Trade Companies 🌎💵


## Business Understanding
### What problem do I want to solve?🕵🏻‍♂️

Developing country such as Indonesia; has a fluctuating, volatile, and higher range of currency exchange value compared to more established countries; this problem brings a problem for company in Indonesia that actively do international trades and often need to convert their cash position either for investment or operation between USD and Indonesian rupiah currency

 ![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/80f607fa-22c8-48c7-93ac-cb0c3071bea9)

### How do I define this problem from a data perspective?

In this project, I want to Develop a data-driven approach to predict future exchange rates between USD and IDR, enabling the financial manager to make informed decisions about when to convert currencies and manage potential financial risks.



2 of the main key components I need to pay attention to are:

1.	Exchange Rate Prediction: Build models that can forecast future exchange rates accurately. This involves predicting how the USD-IDR exchange rate will fluctuate over time.
2.	Risk Management: Implement strategies to mitigate the impact of currency exchange rate fluctuations. This may involve identifying optimal times for currency conversion or using financial instruments to hedge against adverse movements.

In this project, the data that I require to perform the analysis is Historical Exchange Rate Data. Where I will gather historical data on the USD/IDR exchange rates and BTC/USD rates. This data will be crucial for training and evaluating predictive models.


## Who will benefit from my product?

 ![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/7db9a12d-fe06-4092-8365-b1dcb714d355)


I made foreign exchange cash and digital currency advisor using Python for the financial manager in a company who actively in charge of doing international trades and requires frequent conversion between foreign exchanges monthly.

Not just foreign exchange such as USD but also the main cryptocurrency asset which is BTC (Bitcoin), since many companies right now in developing countries accept BTC, the company has a reserve of digital currency assets.

My product can also be used by anyone especially who frequently needs to convert from one currency to another. 

## What do people normally do and is there a technique I would be able to improve upon? How does my product work in a nutshell?

Normally, we rely on traditional analysis such as sentiment, simple moving average, or even without analysis at all when making choices to reserve our currency. This leads to inefficiency in making better choices to have an advantageous currency asset reserve. And oftentimes, we have immediate needs to convert our currency to another, we have a probability of missing the chance to buy the currency at a reasonable price that can save our money.

My product will tell the user to either buy, sell, or hold the currency asset that they are holding based on predicting the next 20 days' price prediction movement. So let's say that the model predicts that in the next 20 days USD IDR will be uptrend, then we can conclude to either buy a USD IDR currency pair or hold it.

Here is a snapshot of my product to make you excited, which I guide you in detail about how it works step by step using easy-to-understand language. In the first snapshot, it is simply a plot created to show my model prediction and the actual price, as you can see it is interesting how my model can almost predict the price movement accurately.

![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/085facde-6c33-4b09-9567-c43e7c409487)


By using my product, I will use statistical data modelling technique called ARIMA which stands for Auto Regressive Integrated Moving Average. I will cover in detail about ARIMA later. But to simply understand how my product work is that find its way the best way to predict the price movement by trial and error it by itself to find which pattern is the best to predict the price movement. When the model finds the best model parameter order, we will use it to predict the price movement of the currency we are predicting

## Why should we use ARIMA to predict price movement? What are the advantages using it, and what are the cons

![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/0978469c-d1aa-43e4-a4ac-241da6a3bce6)

Advantages of ARIMA:
1.	Simple and Interpretable:
ARIMA models are relatively simple and easy to understand, making them accessible for users without extensive statistical or machine learning expertise.

2.	Effectiveness for Stationary Data:
ARIMA works well when the underlying time series data is stationary, meaning that its statistical properties (mean, variance) do not change over time. I will explain what stationarity matters in my analysis another paragraph, it is really important for time series forecasting.

3.	Captures Linear Trends:
ARIMA is effective at capturing linear trends in time series data. It can model the dependencies between observations separated by a fixed time lag.


Disadvantages of ARIMA:
1.	Sensitive to Outliers:
ARIMA models are sensitive to outliers in the data. Extreme values can have a significant impact on parameter estimation, potentially leading to suboptimal predictions.

2.	Fixed Time Lags:
ARIMA models assume that the time lag for autoregression and moving averages is fixed and does not change over time. This may limit their ability to adapt to dynamic changes in the underlying data patterns.

ARIMA is a valuable tool for time series forecasting, particularly when the data is stationary and exhibits linear patterns. However, its effectiveness depends on the characteristics of the data, and it may not perform well in the presence of nonlinearities or strong seasonality. Users should consider the specific requirements of their forecasting task and explore alternative models for comparison.

## What are the criteria for product success for this product?

The model should achieve a high level of accuracy in predicting future USD/IDR and BTC/USD exchange rates, minimizing the error between predicted and actual values. There is a model performance analysis called Mean Absolute Percentage Error, which will indicate the accuracy of the model.

## What are the steps I made to build my product using the ARIMA model?

I built this product using Python programming, which is where the ARIMA model can be worked with using this programming language. And I use  the CRISP-DM framework which stands for Cross-Industry Standard Process for Data Mining, where it is a widely used and well-documented framework for guiding data mining and machine learning projects. It provides a structured approach with defined steps and tasks to ensure a systematic and comprehensive process.

As I already explained about the business understanding of this project clearly, now I will start to get into how do I gathered the data required for making the product or making the analysis.

# Data Understanding 🔍

## How do I collect reliable data sources for currency price quotations?

In order to gather reliable historical data, I  yfinance which is a Python library that provides a simple and convenient way to access financial data from Yahoo Finance. It allows you to retrieve historical market data, real-time quotes, and other financial information for a wide range of assets, including stocks, exchange-traded funds (ETFs), currencies, and more.
```python
# Specify the ticker symbol for the USD to IDR exchange rate
usd_to_idr_ticker = "USDIDR=X"

# Download historical data for the exchange rate
usd_idr_exchange_rate_data = yfsource.download(usd_to_idr_ticker, start="2013-1-1", end="2023-12-16")

# Print the last available exchange rate
usd_idr_last_exchange_rate = usd_idr_exchange_rate_data["Close"].iloc[-1]
```

![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/709ef9e1-7599-40ef-b442-296746bf137b)


For the USD-IDR data, I choose to download data from the date 1 January 2013 to 16 December 2023. And to make sure I retrieved the accurate quotation I printed the last exchange rate.

This is the snapshot of USD-IDR exchange rate data that I retrieved:

![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/79eaad06-49e6-4eb7-9760-0b89c8083380)


Date	:	 This column represents the date of the corresponding exchange rate data

Open	:	The "Open" column represents the opening exchange rate for USD to IDR on a particular day. This is the exchange rate at the beginning of the trading day.

High	: 	The "High" column represents the highest exchange rate for USD to IDR reached during the trading day. It indicates the maximum value of the exchange rate within that day.

Low	: 	The "Low" column represents the lowest exchange rate for USD to IDR reached during the trading day. It indicates the minimum value of the exchange rate within that day.

Close	: 	The "Close" column represents the closing exchange rate for USD to IDR on a particular day. This is the exchange rate at the end of the trading day.

Adj Close	: 	The "Adj Close" (Adjusted Close) column represents the adjusted closing exchange rate for USD to IDR. Adjusted closing rates are modified to account for any relevant adjustments or events, providing a more accurate representation for historical analysis.

I also did the same thing for BTC-USD data, but for instead I downloaded the data from 1 November 2019 to 16 December 2023

```python
# Specify the ticker symbol for BTC to USD
btc_to_usd_ticker = "BTC-USD"

# Download historical data for BTC to USD
btc_to_usd_data = yfsource.download(btc_to_usd_ticker, start="2019-11-1", end="2023-12-16")

# Print the last available BTC to USD closing price
last_btc_price = btc_to_usd_data["Close"].iloc[-1]
print(f"Last BTC to USD closing price: ${last_btc_price:.2f}")
```
![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/8adcc82e-c843-41ce-bce2-75b046183e8d)

This is the snapshot of BTC-USD rate data that I retrieved:

![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/1e77257f-4ea3-4f21-9d30-97702445853a)


Date	:	 This column represents the date of the corresponding price data.

Open	: 	The "Open" column represents the opening price of Bitcoin on a particular day. This is the price of Bitcoin at the beginning of the trading day.

High	: 	The "High" column represents the highest price of Bitcoin reached during the trading day. It indicates the maximum value of Bitcoin's price within that day.

Low	: 	The "Low" column represents the lowest price of Bitcoin reached during the trading day. It indicates the minimum value of Bitcoin's price within that day.

Close	: 	The "Close" column represents the closing price of Bitcoin on a particular day. This is the price of Bitcoin at the end of the trading day.

Adj Close	:	The "Adj Close" (Adjusted Close) column represents the adjusted closing price of Bitcoin.

Volume	:	The "Volume" column represents the trading volume of Bitcoin on that day. It indicates the total number of Bitcoin units traded during the trading day. High trading volumes can suggest increased market activity.

# Data Preparation 📊

## How do I asses the data completeness? 

I make sure that rows that containing null values are removed by using dropna() function, and since in this case, there is no rows removed at all indicating that both dataset I have downloaded was complete.

```python
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

```

![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/162809a6-83fa-4cd2-a4c0-27f1460cbcda)

## How do I identify potential data issues for prediction later on?

Before touching directly about the dataset, I want to make an analogy of what stationary data means and why is it matter for this analysis.

Imagine you have a cup of hot coffee. You want to know if the coffee's temperature changes over time.

### ☕️ Stationary Coffee: 
If the coffee is stationary, it means its temperature remains constant over time. Whether you measure it today, tomorrow, or next week, you expect the temperature to be the same.

Stationary coffee is like a calm lake – it doesn't change much, making it easy to predict its temperature at any given moment.

### ☕️ Non-Stationary Coffee:
If the coffee is non-stationary, it means its temperature fluctuates. Maybe it cools down as time passes, or it gets hotter during certain hours of the day.

Non-stationary coffee is like a roller coaster – the temperature is all over the place, making it hard to predict what it will be at any given time.

## Why Stationarity Matters?

### Predictability
Stationary coffee allows you to predict its temperature easily because it doesn't change much.

Non-stationary coffee makes predictions challenging because its temperature behaves unpredictably.

### Modelling:
Stationary things are simpler to model – you can use a straightforward rule (e.g., temperature remains constant).

Non-stationary things require more complex models to capture their changing patterns.

### Bringing It Back to Time Series⏱️:

![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/169c85cf-ffbf-4fbe-8ad3-dc64631c4678)


### Stationary Time Series

The statistical properties of the series (like mean and variance) don't change over time.
Predictions are more reliable, and models like ARIMA work well.

### Non-Stationary Time Series:

The statistical properties change over time, making predictions more challenging.
Techniques like differencing are used to make the series more stationary.

In this analysis, I use The Augmented Dickey-Fuller (ADF) test which is a statistical test used to determine whether a time series is stationary or non-stationary. Stationarity is a crucial concept in time series analysis, especially when applying models like ARIMA. Then, I create a function to make sure that the data I use for prediction is stationary by differencing it using adfuller library.

```python
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
```
![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/fcf1f9cd-84bc-4edd-850a-847186e0fc28)

### Importance of Stationarity for ARIMA:
•	Stationarity is a key assumption for ARIMA models.
•	Non-stationary time series may exhibit trends, seasonality, or other patterns that make predictions difficult.
•	ARIMA models work well when the underlying time series is stationary because they assume that the statistical properties of the series do not change over time.
•	Stationarity simplifies the modelling process and allows for more reliable forecasts.

In summary, the ADF test results suggest that both the USD to IDR exchange rate and the BTC to USD rate are likely stationary after differencing is applied to the data, which is beneficial for applying ARIMA models for time series analysis and forecasting.

## Making sure the data can be plotted elegantly

Now, before taking step into modelling. I want to see how the downloaded data is plotted elegantly using interactive plotting library called plotly.express which is more interactive than normal matplotlib.pylot library. So later on we can use the same visualisation style in order to compare the model prediction and the actual price of the currency.

![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/fbb2c0d7-b5ed-4e32-a185-a40f1a1d4ead)

![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/07e2bd7c-2436-4af0-880d-042c586c0cd0)



# Modelling

## Why do I choose the ARIMA model instead of the other model?



Remember about the business problem I mentioned at Business Understanding? I mentioned that we often switch between currencies for company transaction purposes, which means that we don’t hold our currency for the long term. The way I define long-term here is above a month. Let's say that in a week we have 5 working days, which means in a month there are 20 working days more or less. And considering that ARIMA has the advantage over its short-term analysis, that is the first reason I choose ARIMA.

The second reason was that here we predict currency that does not fluctuate aggressively like stocks, assuming that ARIMA has issues working with data that have strong seasonality. 

## What parameter used for forecasting the price movement using ARIMA?

![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/1a81758c-b0e5-472f-91ec-65951085d245)

### p (AutoRegressive Order):
•	What it is: The number of steps in the past we look at to predict the future.
•	Think of it as: Imagine predicting tomorrow's weather by looking at the weather for the past few days. If we consider the last 3 days, p would be 3.
•	How to find it: Look at the AutoCorrelation Function (ACF) plot. The lag where the correlation is still significant indicates the value of p.

### d (Integrated Order):
•	What it is: How many times we need to difference the data to make it stable and predictable.
•	Think of it as: If your daily sales data has a trend (like increasing or decreasing), you might need to subtract the sales of the previous day to make it more consistent. The number of times you do this is d.
•	How to find it: Observe the trend in my time series data. If it's not stationary, I difference it and see how many times you need to do this until it becomes stable.

### q (Moving Average Order):
•	What it is: The size of the window we use to smooth out the noise in the data.
•	Think of it as: If your sales data has some random ups and downs, using a moving average helps you see the overall trend by smoothing out these fluctuations. The size of the window is q.
•	How to find it: Look at the Partial AutoCorrelation Function (PACF) plot. The lag where the correlation becomes insignificant indicates the value of q.

### In summary:
•	p: How far back in time we look for predicting.
•	d: How many times we difference the data to make it stable.
•	q: The size of the window for smoothing out noise.

These parameters work together to create an ARIMA model that captures the patterns and relationships in your time series data for making accurate predictions.

## Why should I split the data I collected for analysis into training and testing for ARIMA model prediction?

Splitting data into training and testing sets is a fundamental practice in machine learning and time series analysis, including when working with ARIMA models. Here's why it's important:

### 1.	Model Evaluation:
By splitting your data, you can train the ARIMA model on one subset (training set) and evaluate its performance on another subset (testing set). This allows you to assess how well the model generalizes to new, unseen data.

### 2.	Avoiding Overfitting:
Overfitting occurs when a model learns the training data too well, capturing noise or peculiarities that don't represent the underlying patterns in the data. Splitting the data helps you detect whether the model is overfitting by evaluating its performance on data it hasn't seen during training.

### 3.	Realistic Performance Assessment:
Testing the model on a separate dataset provides a more realistic assessment of its performance in real-world scenarios. It helps you understand how well the model is likely to perform when making predictions on future, unseen data.

In my analysis, the way I perform data splitting into training and testing is by selecting a percentage (90%) of the data for training and the remaining portion for testing. The 'Adj Close' values are extracted for both training and testing sets for both USD to IDR and BTC to USD datasets. These subsets are then used for model training and evaluation.

```python
to_row_usd_idr = int(len(usd_idr_exchange_rate_data)*0.9)

training_data_usd_idr = list(usd_idr_exchange_rate_data[0:to_row_usd_idr]['Adj Close'])

testing_data_usd_idr = list(usd_idr_exchange_rate_data[to_row_usd_idr:]['Adj Close'])

to_row_btc_usd = int(len(btc_to_usd_data)*0.9)

training_data_btc_usd = list(btc_to_usd_data[0:to_row_btc_usd]['Adj Close'])

testing_data_btc_usd = list(btc_to_usd_data[to_row_btc_usd:]['Adj Close'])
```

![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/0fdfbad1-56f3-4730-b697-c59043bd8abd)



Now, here is the  visualisation of how the historical data being used for training and testing at the model prediction:

```python
plt.figure(figsize=(10,6))
plt.title('USD IDR Prediction')
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(usd_idr_exchange_rate_data[0:to_row_usd_idr]['Adj Close'], 'green', label='Training data')
plt.plot(usd_idr_exchange_rate_data[to_row_usd_idr:]['Adj Close'], 'blue', label='Testing data')
plt.legend()
```

![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/a94bbabf-bc85-4333-aa1b-ef2e595c796c)


```python
plt.figure(figsize=(10,6))
plt.title('BTC USD Prediction')
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(btc_to_usd_data[0:to_row_btc_usd]['Adj Close'], 'green', label='Training data')
plt.plot(btc_to_usd_data[to_row_btc_usd:]['Adj Close'], 'blue', label='Testing data')
plt.legend()
```


## How do I determine of how should I input the parameter for this model prediction? 

Here, I wrote code that lets a Python library associated with ARIMA model called 'pmdarima' to automatically determine the best ARIMA model for the given historical rate data. Rather than manually trial and error finding the best way tuning the model parameter.

But important to notice that while ARIMA itself is not a machine learning algorithm, it is often used in conjunction with machine learning techniques for time series forecasting. For example, the auto_arima function employs a stepwise algorithm to automatically select the best ARIMA model based on certain statistical criteria. 

This stepwise approach involves a form of search and optimization. In summary, ARIMA is a statistical approach, but it can be part of a broader forecasting strategy that may involve machine learning techniques. 

```python
from pmdarima import auto_arima
# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

stepwise_fit = auto_arima(usd_idr_exchange_rate_data['Adj Close'], 
                          suppress_warnings=True)           

stepwise_fit.summary()

```
![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/108cfa18-93ba-4e4b-b255-3931fab7e136)

```python
from pmdarima import auto_arima
# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

stepwise_fit = auto_arima(btc_to_usd_data['Adj Close'], 
                          suppress_warnings=True)           

stepwise_fit.summary()
```
![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/43844677-76fd-4817-8560-b21baec59190)

From both code cells, I got two important results which are for USD-IDR model, the best parameter order was (4,1,5) and for BTC-USD model, the best order was (2,1,2). These order will be used after this for training and testing the data through the model.

## How do I train and test the data using ARIMA for predictions?

First, I made code to store all of the predicitions that ARIMA made for both USD-IDR and BTC-USD. I also made code to track how training predictions have been made, this is useful for making an index number ensuring the x and y plot what we use later have the same dimension, so the plotting part to visualize the prediction and actual price in the past is accurate.

```python
model_predictions_usd_idr = []
n_test_observer_usd_idr = len(testing_data_usd_idr)

model_predictions_btc_usd = []
n_test_observer_btc_usd = len(testing_data_btc_usd)
```

Second, I made a loop to iterate as many as the testing data and create prediction as many as the testing data:
```python
for i in range(n_test_observer_usd_idr):
    model = ARIMA(training_data_usd_idr, order=(4, 1, 5))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]  
    model_predictions_usd_idr.append(yhat)
    actual_test_value = testing_data_usd_idr[i]
    training_data_usd_idr.append(actual_test_value)
```
```python
for i in range(n_test_observer_btc_usd):
    model = ARIMA(training_data_btc_usd, order=(2, 1, 2))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]  
    model_predictions_btc_usd.append(yhat)
    actual_test_value = testing_data_btc_usd[i]
    training_data_btc_usd.append(actual_test_value)
```
As you can see above, I utilize the order from previous 'auto_arima' code to be inserted into the parameter of ARIMA model. Now let’s check whether we already make prediction as many as we test the data:

![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/fcfc1262-e3b7-4bc9-b3b5-730f4f55ba1c)

And the result are we made it equal, means that the model predictions are made as we intended and ready to be compared through the plots.

## How do I plot or visualize the model prediction results with the actual price of currency to ensure its reliability?

Making a plot is essential part to interpret how the model is accurately predict the price of currency I work with for this product. Here, I use to plotly.express for plotting, so when we can hover over the chart to directly see the prediction price and actual price at any day.

```python
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
```
![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/890d7226-1434-442c-b276-1897f06cd4af)


Next, I also made another plot for USD-IDR, but this plot is more static than the previous plot, the purpose of this plot is to see when the prediction is made before the actual price moves in certain direction. The prediction is represented in blue dots like shown below:

```python
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
```
![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/274765c1-fd2d-4b4a-8871-288f4a174904)

```python
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

```
![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/2f51bf5a-9fc1-4f45-b958-9251946222e3)

```python
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

```
![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/264cb976-9cb4-4885-918c-eb145a660a65)


Important to notice that to make these plot, I have encountered error several times this is because as we know that where we need to use the same dimension for testing data and model prediction. But the since only testing data values length is integer values while model predictions end up in float type. The difference like for example (261, ) and (262) recognized as not matched dimension for plotting.

To tackle this issue I must configure by adding [:-1]  to 'model_predictions_usd_idr' and 'testing_data_usd_idr' at the plot function parameter, what does that do is to obey the last value in the index, so the dimension is equal for both model predictions and testing data. 

# Model Evaluation

## What method do I choose to evaluate the model prediction?

```python
mape = np.mean(np.abs(np.array(model_predictions_usd_idr) - np.array(testing_data_usd_idr))/np.abs(testing_data_usd_idr))
print('Mean Absolute Percentage Error for usd-idr predictions:  ' + str(mape))

mape2 = np.mean(np.abs(np.array(model_predictions_btc_usd) - np.array(testing_data_btc_usd))/np.abs(model_predictions_btc_usd))
print('Mean Absolute Percentage Error for btc-usd predictions:  ' + str(mape2))

```

![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/d8de6861-593f-48ce-8203-37d4e2aa6bfb)

To evaluate the model prediction, I use Mean Absolute Percentage Error (MAPE) which is a metric commonly used to evaluate the accuracy of forecasting models, including currency exchange rate predictions. The MAPE is calculated as the average percentage difference between the predicted values and the actual values.

In my USD-IDR predictions, the MAPE value was 0.004030056475883926, in this case it's approximately 0.40%. This suggests that, on average, the model's predictions have an error of about 0.40% compared to the actual values. A lower MAPE indicates better accuracy. In this context, a MAPE of 0.40% is relatively low, suggesting that the model is making accurate predictions, on average, for the USD-IDR exchange rate.

Moving on, in my BTC-USD prediction, the MAPE value was 0.013002891853874975 . The MAPE is expressed as a percentage, and in this case, it's around 1.30%. This suggests that, on average, the model's predictions for BTC-USD have an error of about 1.30% compared to the actual values.

But it is unfair if we compare the USD-IDR prediction with BTC-USD prediction head to head because both have different seasonality pattern at the length of the data is far different.

Remember that MAPE is just one metric, and it provides a measure of accuracy but does not provide information about the direction of errors (overestimation or underestimation). Always consider a combination of metrics and visual inspections to assess the overall performance of forecasting model.

## What can I conclude from the model evaluation? Is there an issue with the model performance analysis?

It is almost impossible that a model can predict something with 99% accuracy, given the model in this analysis can give almost 99% accuracy, we can see that this model might have overfitting problem. Overfitting means that the data took too much training data and fits the pattern too much until the prediction accuracy looks like perfect.

In real world application, we need to better adjust the prediction into more make sense. So because the model have an overfitting issue, I decided to improve the model to make the accuracy level more make sense.

# How do I improve the prediction model?

Remember, when I used 90% of the data for training data and the rest 10% for testing data? In my hypothesis, I suggest that there is too much imbalance between the training and testing, while we already know that we need to use more training data than testing data of course. But to make it more make sense, I lowered the use of training data to 70% and 30% for testing data.

![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/e020768d-4b9e-4cb1-bc78-55c1d042477a)

Now, I turned the code into this:

![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/1f9b1c33-579b-458f-8f9c-f92271b0d57b)


But it turns out, that increasing testing percentage does not decrease the accuracy of the model. 

# Deployment

## What are the prediction results from ARIMA model for the currency? 

As I already made train and test the model and confirming its accuracy of predicting, now it is time to deploy the model into predicting the future currency price movement.

I made code to forecast next 20 days of currency price value. Based on the prediction, I use the same parameter order for ARIMA model. 

```python
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
```
![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/c6c2332d-35ad-45ef-a72c-80373b4157f7)

```python
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

```
![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/793ea26e-f4e6-4492-ad29-e2b2143f7059)

## How the product can give advice regarding the trend of the currency?

Rather than manually read the prediction one by one, my code will automatically detect whether the trend of the currency is heading toward up, down, or sideways. 

The way the code works is by comparing the first value in the future prediction index and the last value in future prediction index. If the last value is higher in prediction index, it will indicate uptrend of the currency, if it has lower value, then the price prediction is down trending, if it is the same value, means it is sideways.

```python
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

```

![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/7f11b54c-8ab4-4eb2-a146-316283d80b27)

```python
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

```

![image](https://github.com/jasontjahja/arima-exchange-rate-prediction/assets/144135838/743141bb-d1d3-43f0-b227-5e6626fcffcb)

Based on the first output from the trend conclusion code, it is stated how USD IDR move towards downtrend. A user that reads the conclusion can follow the trend conclusion in order to protect their USD-IDR asset by selling it and convert it to the pair currency. 

In the opposite, based on this case, ARIMA predicted the uptrend market of BTC USD. This would be a buying or holding signal for BTC USD asset.




# Ethical Concern ❤️👁️📝

## What are the things should user of this product concern related to the ethical practical use in predicting currency asset using this product?

From a data mining perspective, there are several ethical concerns associated with predicting BTC-USD and USD-IDR using ARIMA models or any other predictive modelling approach. 

To encourage responsible use of predictions and provide clear guidelines for users I need to highlight some key ethical considerations:

### Use of Predictions:
Consider the potential consequences of the predictions. Since at the beginning of designing this product, I intended using it for financial related decision-making, there could be significant impacts if a user use this model prediction independently for making decision regarding trading their company cash reserve.

### Bias and Fairness:
Models trained on historical data may inherit biases present in the data. If historical data reflects discriminatory practices, the model might perpetuate those biases in predictions.

### Data Ownership:
I have ensured that the data source used for this prediction using ARIMA is collected and used ethically. In this case I have used a library created by Yahoo Finance, a trusted and reliable financial information media that generate Python library for open source purpose.

### Transparency and Explainability:
Lack of transparency in model development and predictions can be a concern. If you are the use of  this product,  you must understand how the prediction generated by this model.

### Regulatory Compliance:
To Ensure compliance with relevant data protection and privacy regulations. Different regions may have specific rules governing the use of personal or financial data in a company. Make sure to use this model regarding to the regulation of government and also company policy for making financial related decision.






























































