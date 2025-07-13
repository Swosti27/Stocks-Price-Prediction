import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load Data
data = pd.read_csv('price.csv')

# Step 2: Feature Engineering
data['MA10'] = data['Close'].rolling(window=10).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()
data = data.dropna()

# Step 3: Define Features and Target
features = ['MA10', 'MA50', 'Volume']
target = 'Close'

# Step 4: Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features + [target]])

# Step 5: Prepare Sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, :-1])  # All features except the target
        y.append(data[i, -1])                # The target column
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)

# Step 6: Train/Test Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Step 7: Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Step 8: Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Step 9: Predictions
y_pred = model.predict(X_test)

# Reverse normalization (only for plotting & metrics)
scaled_close = scaled_data[seq_length:, -1]
actual_prices = y_test
predicted_prices = y_pred.flatten()

# Step 10: Evaluation
r2 = r2_score(actual_prices, predicted_prices)

print(f"R² Score (Accuracy): {r2:.2%}")

# Step 11: Plot Actual vs Predicted
plt.figure(figsize=(14,6))
plt.plot(range(len(actual_prices)), actual_prices, label='Actual Price', color='blue')
plt.plot(range(len(predicted_prices)), predicted_prices, label='Predicted Price', color='red')
plt.title('LSTM Price Prediction')
plt.xlabel('Time')
plt.ylabel('Normalized Price')
plt.legend()
plt.grid(True)
plt.show()
