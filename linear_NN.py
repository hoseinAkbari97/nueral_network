import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# 1. Load data without headers, name columns 'u' (input) and 'y' (output)
df = pd.read_excel(
    'data/studentDataLinear6.xls',
    header=None,
    names=['u', 'y']
)

# 2. Extract input (X) and output (y) arrays
X = df[['u']].values
y = df['y'].values

# 3. Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 4. Standardize inputs and outputs
scaler_X = StandardScaler().fit(X_train)
scaler_y = StandardScaler().fit(y_train.reshape(-1,1))
X_train_s = scaler_X.transform(X_train)
X_test_s  = scaler_X.transform(X_test)
y_train_s = scaler_y.transform(y_train.reshape(-1,1)).ravel()
y_test_s  = scaler_y.transform(y_test.reshape(-1,1)).ravel()

# 5. Define the MLP regressor with three hidden layers
mlp = MLPRegressor(
    hidden_layer_sizes=(50, 30, 10),
    activation='relu',               
    solver='adam',
    learning_rate_init=1e-3,
    max_iter=2000,
    random_state=42
)

# 6. Train the network
mlp.fit(X_train_s, y_train_s)

# 7. Evaluate performance
print(f"Train R²: {mlp.score(X_train_s, y_train_s):.3f}")
print(f"Test  R²: {mlp.score(X_test_s,  y_test_s):.3f}")

# 8. Simulation: feed each predicted output back as next input
y_sim_s = []
u_curr = X_test_s[0, 0]
for _ in range(len(X_test_s)):
    y_pred_s = mlp.predict([[u_curr]])
    y_sim_s.append(y_pred_s[0])
    u_curr = y_pred_s[0]

# 9. Inverse-transform simulated outputs
y_sim = scaler_y.inverse_transform(
    np.array(y_sim_s).reshape(-1,1)
).ravel()

# 10. Plot true vs. simulated outputs
plt.figure()
plt.plot(y_test,  label='True Output')
plt.plot(y_sim,  linestyle='--', label='Simulated Output')
plt.xlabel('Time Step')
plt.ylabel('Output y')
plt.title('Linear System: True vs. Simulated (MLP)')
plt.legend()
plt.show()
