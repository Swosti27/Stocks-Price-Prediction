# Stocks-Price-Prediction
# 📈 Stock Price Prediction using LSTM

This project uses a Long Short-Term Memory (LSTM) neural network to predict stock prices based on historical data. It includes feature engineering with moving averages and volume data to enhance predictive performance.

## 🧠 Project Overview

The project aims to:
- Predict future stock closing prices using LSTM neural networks.
- Incorporate technical indicators such as MA10 and MA50.
- Evaluate the performance of the model using R² Score.
- Visualize actual vs predicted normalized prices.

## 📁 Dataset

The model uses a CSV file named `price.csv` which must contain at least the following columns:

- `Close`: The closing price of the stock.
- `Volume`: Trading volume.

Make sure your CSV is formatted correctly and includes sufficient historical data for meaningful rolling averages and LSTM sequences.

## 🛠️ Features Used

- **MA10**: 10-day Moving Average
- **MA50**: 50-day Moving Average
- **Volume**: Daily trading volume

These features are normalized using MinMaxScaler before feeding into the LSTM.

## 📦 Libraries Required

Install the dependencies using pip:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow
