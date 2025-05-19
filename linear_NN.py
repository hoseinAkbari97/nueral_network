import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ========================
# 1. Load Data
# ========================
df = pd.read_excel('data/studentDataLinear6.xls', header=None, names=['u', 'y'])
X = df[['u']].values
y = df['y'].values

# ========================
# 2. Scale Inputs/Outputs
# ========================
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Split data (static system, random split is acceptable)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# ========================
# 3. Configure MLP for Linear Regression
# ========================
mlp = MLPRegressor(
    hidden_layer_sizes=(1,),      
    activation='identity',        
    solver='lbfgs',               
    alpha=0.0,                    
    max_iter=10000,               
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
# 6. Plot y vs. u
# ========================
plt.figure(figsize=(8, 6))
plt.scatter(scaler_X.inverse_transform(X_test), y_test_original, 
            s=30, label='True Output', alpha=0.7)
plt.scatter(scaler_X.inverse_transform(X_test), y_sim, 
            s=30, marker='x', label='Simulated Output', color='red')
plt.xlabel('Input $u$', fontsize=12)
plt.ylabel('Output $y$', fontsize=12)
plt.title('Linear System: True vs. Simulated Output (y vs. u)', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# ========================
# 7. Evaluate Performance
# ========================
print(f'Train R²: {mlp.score(X_train, y_train):.3f}')
print(f'Test R²: {mlp.score(X_test, y_test):.3f}')

# True slope (from raw data)
true_slope = np.polyfit(df['u'], df['y'], 1)[0]

# Simulated slope (from MLP predictions)
sim_slope = np.polyfit(scaler_X.inverse_transform(X_test).flatten(), y_sim, 1)[0]

print(f'True Slope: {true_slope:.4f}')
print(f'Simulated Slope: {sim_slope:.4f}')

# ========================
# 8. Load Test Data and Simulate
# ========================
test_df = pd.read_excel('data/studentDatafortest6.xls', header=None, names=['u'])
u_test = test_df[['u']].values

# Scale test input (DO NOT re-fit scaler_X!)
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
output_df.to_excel('data/simulated_test_linear_outputs.xlsx', index=False)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(output_df['Input (u)'], output_df['Simulated Output (y)'], 
            s=30, alpha=0.7, color='green', label='Test Simulation')
plt.xlabel('Input $u$', fontsize=12)
plt.ylabel('Simulated Output $y$', fontsize=12)
plt.title('Test Data: Input vs. Simulated Output', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# ========================
# 10. Combined Plot: All Data Together
# ========================
plt.figure(figsize=(10, 6))

# Plot training data (all points)
plt.scatter(df['u'], df['y'], s=10, alpha=0.4, label='Training Data', color='blue')

# Plot test data (true and simulated)
plt.scatter(scaler_X.inverse_transform(X_test), y_test_original, 
            s=30, alpha=0.7, label='True Test Data', color='orange')
plt.scatter(scaler_X.inverse_transform(X_test), y_sim, 
            s=30, marker='x', label='Simulated Test Data', color='red')

# Plot external test simulation
plt.scatter(output_df['Input (u)'], output_df['Simulated Output (y)'], 
            s=30, marker='+', label='External Test Simulation', color='green')

plt.xlabel('Input $u$', fontsize=12)
plt.ylabel('Output $y$', fontsize=12)
plt.title('Combined Plot: Training, Test, and Simulated Data', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

print("Test simulation complete! Results saved to 'simulated_test__linear_outputs.xlsx'.")
