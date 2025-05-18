import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

# 1. Load data without headers, name columns 'u' (input) and 'y' (output)
df = pd.read_excel(
    'studentDataLinear6.xls',
    header=None,            # no header row in file :contentReference[oaicite:6]{index=6}
    names=['u', 'y']        # assign custom column names :contentReference[oaicite:7]{index=7}
)

# 2. Extract input (X) and output (y) arrays
X = df[['u']].values       # shape (n_samples, 1)
y = df['y'].values         # shape (n_samples,)

# 3. Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,          # 20% test data :contentReference[oaicite:8]{index=8}
    random_state=42         # reproducible splits :contentReference[oaicite:9]{index=9}
)

# 4. Standardize inputs and outputs
scaler_X = StandardScaler().fit(X_train)     # scale features :contentReference[oaicite:10]{index=10}
scaler_y = StandardScaler().fit(y_train.reshape(-1,1))
X_train_s = scaler_X.transform(X_train)
X_test_s  = scaler_X.transform(X_test)
y_train_s = scaler_y.transform(y_train.reshape(-1,1)).ravel()
y_test_s  = scaler_y.transform(y_test.reshape(-1,1)).ravel()

# 5. Define the MLP regressor with three hidden layers
mlp = MLPRegressor(
    hidden_layer_sizes=(50, 30, 10),  # three hidden layers :contentReference[oaicite:11]{index=11}
    activation='relu',                # ReLU activations :contentReference[oaicite:12]{index=12}
    solver='adam',                    # Adam optimizer :contentReference[oaicite:13]{index=13}
    learning_rate_init=1e-3,          # initial learning rate
    max_iter=2000,                    # maximum epochs
    random_state=42                   # reproducibility :contentReference[oaicite:14]{index=14}
)

# 6. Train the network
mlp.fit(X_train_s, y_train_s)         # backpropagation training

# 7. Evaluate performance
print(f"Train R²: {mlp.score(X_train_s, y_train_s):.3f}")
print(f"Test  R²: {mlp.score(X_test_s,  y_test_s):.3f}")

# 8. Simulation: feed each predicted output back as next input
y_sim_s = []
u_curr = X_test_s[0, 0]
for _ in range(len(X_test_s)):
    y_pred_s = mlp.predict([[u_curr]])  # predict scaled output :contentReference[oaicite:15]{index=15}
    y_sim_s.append(y_pred_s[0])
    u_curr = y_pred_s[0]               # feedback loop

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
plt.title('Linear System: True vs. Simulated (MLP)')  # :contentReference[oaicite:16]{index=16}
plt.legend()
plt.show()
