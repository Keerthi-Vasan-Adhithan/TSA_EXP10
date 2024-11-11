# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL FOR Apple Stock DATA
### Date: 
### Developed by: KEERTHI VASAN A
### Register Number: 212222240048

### AIM:
To implement the SARIMA model using Python for time series analysis on Apple Stock data.

### ALGORITHM:
1. Explore the Dataset - Load the Apple Stock dataset and perform initial exploration, focusing on the `year` and `value` columns. Plot the time series to visualize trends.

2. Check for Stationarity of Time Series - Plot the time series data and apply the Augmented Dickey-Fuller (ADF) test to check for stationarity.

3. Determine SARIMA Model Parameters (p, d, q, P, D, Q, m)
  
4. Fit the SARIMA Model
  
5. Make Time Series Predictions and Auto-fit the SARIMA Model 
   
6. Evaluate Model Predictions
   

### PROGRAM:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# Load the dataset from uploaded file
data = pd.read_csv('/content/apple_stock.csv')  # Replace 'apple_stock.csv' with your uploaded file

# Convert 'Date' column to datetime and set it as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Extract the 'Close' column (or another relevant column)
series = data['Close'].dropna()

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(series, label='Stock Closing Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Data Time Series')
plt.grid(True)
plt.show()

# Function to perform ADF test
def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'    {key}: {value}')

# Check stationarity
check_stationarity(series)

# Plot ACF and PACF to determine ARIMA parameters
plot_acf(series)
plt.title('ACF Plot')
plt.show()

plot_pacf(series)
plt.title('PACF Plot')
plt.show()

# Define SARIMA parameters based on ACF/PACF plots (example values)
p, d, q = 1, 1, 1   # Non-seasonal parameters
P, D, Q, m = 1, 1, 1, 12  # Seasonal parameters (assumes annual seasonality with monthly data)

# Split the data into train and test sets (80% train, 20% test)
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

# Fit SARIMA model
sarima_model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, m))
sarima_result = sarima_model.fit()

# Make predictions
predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate RMSE for evaluation
mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('SARIMA Model Predictions')
plt.grid(True)
plt.legend()
plt.show()

```

### OUTPUT:

Consumption time series:

![image](https://github.com/user-attachments/assets/adfe713f-6f0f-4dbd-9492-b08ba02c440c)
![image](https://github.com/user-attachments/assets/a5cf31c6-4f94-4dc7-b0d5-3adb9e8f56fb)


Autocorrelation:

![image](https://github.com/user-attachments/assets/647e693f-5011-4d16-a6b0-81a752420947)



Partial Autocorrelation:

![image](https://github.com/user-attachments/assets/33496faa-7913-4541-830d-e1ba03983a6a)


SARIMA Model Prediction:

![image](https://github.com/user-attachments/assets/1bed29c0-5e59-4eb9-bdc5-3a581556da7c)



### RESULT:
Thus the program using SARIMA model is executed successfully.
