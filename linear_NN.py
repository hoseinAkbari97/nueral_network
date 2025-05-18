import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ========================
# 1. Load Data
# ========================
df = pd.read_excel('data/studentDataLinear6.xls', header=None, names=['u', 'y'])

# ========================
# 2. Prepare Static System Data (No Time Lags)
# ========================
X = df[['u']].values  # Input: u(t)
y = df['y'].values    # Output: y(t) (directly depends on u(t))

# Split randomly (since system is static)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========================
# 3. Train MLP (Simple Linear Model)
# ========================
mlp = MLPRegressor(
    hidden_layer_sizes=(1,),  # Single neuron (equivalent to linear regression)
    activation='identity',    # Linear activation
    solver='adam',
    max_iter=1000,
    random_state=42
)
mlp.fit(X_train, y_train)

# ========================
# 4. Predict on Test Data
# ========================
y_sim = mlp.predict(X_test)

# ========================
# 5. Plot True vs. Simulated (y vs. u)
# ========================
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, s=30, label='True Output', alpha=0.7)
plt.scatter(X_test, y_sim, s=30, marker='x', label='Simulated Output', color='red')
plt.xlabel('Input $u$', fontsize=12)
plt.ylabel('Output $y$', fontsize=12)
plt.title('Linear System: True vs. Simulated Output (y vs. u)', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# ========================
# 6. Print Performance
# ========================
print(f'Train R²: {mlp.score(X_train, y_train):.3f}')
print(f'Test R²: {mlp.score(X_test, y_test):.3f}')
