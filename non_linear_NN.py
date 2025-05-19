import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ========================
# 1. Load Data (Nonlinear)
# ========================
df = pd.read_excel('data/studentDataNonLinear6.xls', header=None, names=['u', 'y'])
X = df[['u']].values
y = df['y'].values

# ========================
# 2. Scale Inputs/Outputs
# ========================
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Split data (static system, random split)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# ========================
# 3. Configure MLP (Nonlinear)
# ========================
mlp = MLPRegressor(
    hidden_layer_sizes=(50, 30, 20),  # Deeper architecture
    activation='relu',                 # Nonlinear activation
    solver='adam',                     # Better for nonlinear problems
    alpha=0.001,                       # L2 regularization
    learning_rate_init=0.001,
    max_iter=5000,                     # Increased iterations
    early_stopping=True,               # Prevent overfitting
    random_state=42
)

# ========================
# 4. Train the Model
# ========================
mlp.fit(X_train, y_train)

# ========================
# 5. Predict and Inverse Scaling
# ========================
y_sim_scaled = mlp.predict(X_test)
y_sim = scaler_y.inverse_transform(y_sim_scaled.reshape(-1, 1)).ravel()
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

# ========================
# 6. Plot y vs. u (True vs. Simulated)
# ========================
plt.figure(figsize=(8, 6))
plt.scatter(scaler_X.inverse_transform(X_test), y_test_original, 
            s=30, label='True Output', alpha=0.7)
plt.scatter(scaler_X.inverse_transform(X_test), y_sim, 
            s=30, marker='x', label='Simulated Output', color='red')
plt.xlabel('Input $u$', fontsize=12)
plt.ylabel('Output $y$', fontsize=12)
plt.title('Nonlinear System: True vs. Simulated Output', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# ========================
# 7. Evaluate Performance
# ========================
print(f'Train R²: {mlp.score(X_train, y_train):.3f}')
print(f'Test R²: {mlp.score(X_test, y_test):.3f}')

# ========================
# 8. Load Test Data and Simulate
# ========================
test_df = pd.read_excel('data/studentDatafortest6.xls', header=None, names=['u'])
u_test = test_df[['u']].values

# Scale test input
u_test_scaled = scaler_X.transform(u_test)

# Predict and inverse-scale
y_test_scaled = mlp.predict(u_test_scaled)
y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()

# ========================
# 9. Save and Plot Results
# ========================
output_df = pd.DataFrame({
    'Input (u)': test_df['u'],
    'Simulated Output (y)': y_test
})
output_df.to_excel('data/simulated_test_outputs_nonlinear.xlsx', index=False)

# ========================
# 10. Combined Plot (All Data)
# ========================
plt.figure(figsize=(10, 6))

# Plot training data
plt.scatter(df['u'], df['y'], s=10, alpha=0.4, label='Training Data', color='blue')

# Plot test split
plt.scatter(scaler_X.inverse_transform(X_test), y_test_original,
            s=30, alpha=0.7, label='True Test Data', color='orange')
plt.scatter(scaler_X.inverse_transform(X_test), y_sim,
            s=30, marker='x', label='Simulated Test Data', color='red')

# Plot external test simulation
plt.scatter(output_df['Input (u)'], output_df['Simulated Output (y)'],
            s=30, marker='+', label='External Test Simulation', color='green')

plt.xlabel('Input $u$', fontsize=12)
plt.ylabel('Output $y$', fontsize=12)
plt.title('Nonlinear System: Combined Data Plot', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()
