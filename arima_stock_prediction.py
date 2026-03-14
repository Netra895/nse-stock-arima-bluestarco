# NSE Stock Price Forecast using ARIMA
# Stock: BLUESTARCO

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Visualization style
sns.set_style("darkgrid")


# -----------------------------
# 1. Load Dataset
# -----------------------------

data = pd.read_csv("data/BLUESTARCO.csv")

print("\nFirst 5 rows of dataset:")
print(data.head())


# -----------------------------
# 2. Data Preprocessing
# -----------------------------

# Convert DATE column to datetime
data['DATE'] = pd.to_datetime(data['DATE'])

# Remove commas and convert CLOSE column to float
data['CLOSE'] = data['CLOSE'].astype(str).str.replace(',', '')
data['CLOSE'] = data['CLOSE'].astype(float)

# Sort by date
data = data.sort_values('DATE')

# Set DATE as index
data.set_index('DATE', inplace=True)

# Handle missing values
data['CLOSE'] = data['CLOSE'].fillna(method='ffill')

print("\nMissing values in CLOSE column:", data['CLOSE'].isnull().sum())


# -----------------------------
# 3. Dataset Summary
# -----------------------------

print("\nDataset Summary")
print(data.describe())


# -----------------------------
# 4. ADF Test (Stationarity)
# -----------------------------

print("\nADF TEST RESULT")

adf_result = adfuller(data['CLOSE'])

print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])

for key, value in adf_result[4].items():
    print("Critical Value", key, ":", value)

if adf_result[1] < 0.05:
    print("Series is stationary")
else:
    print("Series is not stationary")


# -----------------------------
# 5. Fit ARIMA Model
# -----------------------------

print("\nTraining ARIMA Model...")

model = ARIMA(data['CLOSE'], order=(1,1,1))
model_fit = model.fit()

print("\nARIMA Model Summary")
print(model_fit.summary())


# -----------------------------
# 6. Forecast Next 30 Days
# -----------------------------

forecast_steps = 30

forecast = model_fit.get_forecast(steps=forecast_steps)

forecast_values = forecast.predicted_mean
confidence_interval = forecast.conf_int()

print("\nNext 30 Day Forecast:")
print(forecast_values)


# -----------------------------
# 7. Dashboard Visualization
# -----------------------------

fig, axes = plt.subplots(2,2, figsize=(16,10))


# Closing Price Trend
axes[0,0].plot(data['CLOSE'], color='royalblue', linewidth=2)
axes[0,0].set_title("BLUESTARCO Closing Price Trend")
axes[0,0].set_xlabel("Date")
axes[0,0].set_ylabel("Price")


# ACF Plot (Orange)
plot_acf(data['CLOSE'], ax=axes[0,1], color='orange')
axes[0,1].set_title("Autocorrelation Function (ACF)")


# PACF Plot (Green)
plot_pacf(data['CLOSE'], ax=axes[1,0], color='green')
axes[1,0].set_title("Partial Autocorrelation Function (PACF)")


# Forecast Graph
axes[1,1].plot(data['CLOSE'], label="Historical Price", color="royalblue")

axes[1,1].plot(forecast_values, label="Forecast Price", color="red")

axes[1,1].fill_between(
    forecast_values.index,
    confidence_interval.iloc[:,0],
    confidence_interval.iloc[:,1],
    color="pink",
    alpha=0.3
)

axes[1,1].set_title("30-Day Price Forecast")
axes[1,1].legend()


plt.tight_layout()
plt.show()