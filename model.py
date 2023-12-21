import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Fetch historical stock data using Yahoo Finance API
stock_symbol = 'AAPL'  # Replace with your desired stock symbol
stock_data = yf.download(stock_symbol, start='2022-01-01', end='2023-12-21')

# Feature engineering - assuming 'Close' is the target variable and other columns are features
features = ['Open', 'High', 'Low', 'Volume']
target = 'Close'

# Splitting the data into features and target variable
X = stock_data[features]
y = stock_data[target]

# Convert data to NumPy arrays for scaling
X_np = X.values
y_np = y.values

# Create a StandardScaler and fit/transform features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_np)

# Create a Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model on all available data
rf.fit(X_scaled, y_np)

# Predicting for a specific date (21st December 2023)
last_data_point = X_np[-1].reshape(1, -1)
scaled_last_data_point = scaler.transform(last_data_point)
predicted_price = rf.predict(scaled_last_data_point)

print(f"Predicted Close Price on 21st December 2023 for {stock_symbol}: {predicted_price[0]}")