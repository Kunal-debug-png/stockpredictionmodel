Stock Price Prediction using Random Forest Regressor
This Python script fetches historical stock data using Yahoo Finance API for a specified stock symbol, performs feature engineering, scales the data, and predicts the closing stock price using a Random Forest Regressor model.

Requirements
Python 3.x
Libraries: yfinance, numpy, scikit-learn
Usage
Install Required Libraries

You can install the required libraries via pip:

bash
Copy code
pip install yfinance numpy scikit-learn
Run the Script

Replace 'AAPL' with your desired stock symbol in the stock_symbol variable.

python
Copy code
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# ... [Code Snippet from your script] ...

# Predicting for a specific date (21st December 2023)
last_data_point = X_np[-1].reshape(1, -1)
scaled_last_data_point = scaler.transform(last_data_point)
predicted_price = rf.predict(scaled_last_data_point)

print(f"Predicted Close Price on 21st December 2023 for {stock_symbol}: {predicted_price[0]}")
Understanding the Script

stock_symbol: Replace it with the desired stock symbol for which you want to predict the closing price.
The script fetches historical stock data, performs feature engineering using 'Open', 'High', 'Low', 'Volume' as features, and 'Close' as the target variable.
It scales the data using StandardScaler and trains a RandomForestRegressor model with 100 estimators.
Finally, it predicts the closing price for a specific date (21st December 2023) based on the trained model.
Output

Upon running the script, it will display the predicted closing price for the specified stock symbol on 21st December 2023.
