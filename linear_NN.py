import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ========================
# 1. Load and Prepare Data
# ========================
df = pd.read_excel('data/studentDataLinear6.xls', header=None, names=['u', 'y'])

# Create time-lagged features (system memory)
n_lags = 1  # Number of past inputs to use (adjust as needed)
X = np.column_stack([df['u'].shift(i) for i in range(n_lags + 1)])  # u(t), u(t-1), ...
X = X[n_lags:]  # Remove NaN rows
y = df['y'][n_lags:].values  # Align target

# ========================
# 2. Split Data Sequentially
# ========================
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# ========================
# 3. Train MLP (Linear System)
# ========================
mlp = MLPRegressor(
    hidden_layer_sizes=(10,),  # Single hidden layer
    activation='identity',     # Linear activation
    solver='adam',
    max_iter=5000,
    random_state=42
)
mlp.fit(X_train, y_train)

# ========================
# 4. Simulate Outputs
# ========================
y_sim = []
current_state = X_test[0].copy()  # Initial state [u(t), u(t-1), ...]

for _ in range(len(X_test)):
    # Predict next output
    y_pred = mlp.predict([current_state])[0]
    y_sim.append(y_pred)
    
    # Update state for next prediction
    current_state = np.roll(current_state, -1)
    current_state[-1] = y_pred  # Assume y(t) becomes u(t+1) for simulation

# ========================
# 5. Generate Plots
# ========================
# Plot 1: Time Series (True vs. Simulated)
plt.figure(figsize=(12, 5))
plt.plot(y_test, label='True Output', linewidth=2)
plt.plot(y_sim, '--', label='Simulated Output', linewidth=1.5)
plt.xlabel('Time Step', fontsize=12)
plt.ylabel('Output $y(t)$', fontsize=12)
plt.title('Linear System: Output vs. Time', fontsize=14)
plt.legend()
plt.grid(True)

# Plot 2: Input vs. Output Relationship
plt.figure(figsize=(8, 6))
plt.scatter(df['u'], df['y'], s=8, alpha=0.6, label='All Data')
plt.scatter(X_test[:, 0], y_test, s=20, color='red', label='Test Data')
plt.xlabel('Input $u(t)$', fontsize=12)
plt.ylabel('Output $y(t)$', fontsize=12)
plt.title('Linear Relationship: Input vs. Output', fontsize=14)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ========================
# 6. Print Performance
# ========================
print(f'Train R²: {mlp.score(X_train, y_train):.3f}')
print(f'Test R²: {mlp.score(X_test, y_test):.3f}')