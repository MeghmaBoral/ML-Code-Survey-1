"""
CODING ACTIVITY 1: Linear Regression - ML From Scratch
Find the best-fitted line among 10 candidate models using MSE,
then demonstrate Gradient Descent on the best model.

Author  : Jhalak Hota
Tool    : PyCharm
Library : NumPy, Matplotlib, Seaborn, Pandas
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ============================================================
# 1. DATA SETUP
# ============================================================

Data   = np.linspace(-4, 4, 100)
target = np.sin(Data)

# ============================================================
# 2. DEFINE 10 CANDIDATE MODELS
# ============================================================

itr1  = 0.30 * Data - 0.20
itr2  = 0.40 * Data - 0.40
itr3  = 0.45 * Data - 0.45
itr4  = 0.50 * Data - 0.50
itr5  = 0.30 * Data - 0.20
itr6  = 1.00 * Data - 1.00
itr7  = 0.35 * Data - 0.60
itr8  = 0.25 * Data - 0.80
itr9  = 0.50 * Data - 1.20
itr10 = 0.60 * Data - 0.20

models = {
    'Itr1':  itr1,  'Itr2':  itr2,  'Itr3':  itr3,
    'Itr4':  itr4,  'Itr5':  itr5,  'Itr6':  itr6,
    'Itr7':  itr7,  'Itr8':  itr8,  'Itr9':  itr9,
    'Itr10': itr10
}

# ============================================================
# 3. PLOT 1 — Itr1 vs Actual  (as given in original notebook)
# ============================================================

DATA_df = pd.DataFrame({'Data': Data, 'Target': target, 'Itr1': itr1})
plt.figure(figsize=(8, 4))
sns.lineplot(data=DATA_df, x='Data', y='Target', label='Actual')
sns.lineplot(data=DATA_df, x='Data', y='Itr1',   label='Itr1')
plt.title('Figure 1: Actual vs Itr1')
plt.tight_layout()
plt.show()

# ============================================================
# 4. PLOT 2 — All 10 Models vs Actual
# ============================================================

DATA_df2 = pd.DataFrame({'Data': Data, 'Target': target, **models})
plt.figure(figsize=(10, 5))
sns.lineplot(data=DATA_df2, x='Data', y='Target', label='Actual', linewidth=2.5)
for name in models:
    sns.lineplot(data=DATA_df2, x='Data', y=name, label=name)
plt.title('Figure 2: All 10 Candidate Models vs Actual')
plt.legend(fontsize=8, ncol=2, loc='upper left')
plt.tight_layout()
plt.show()

# ============================================================
# 5. FIND BEST MODEL USING MSE
# ============================================================

def compute_mse(actual, predicted):
    """Mean Squared Error — computed from scratch."""
    n = len(actual)
    return (1 / n) * np.sum((actual - predicted) ** 2)

mse_values = {}
for name, pred in models.items():
    mse_values[name] = compute_mse(target, pred)

best_model_name = min(mse_values, key=mse_values.get)
best_mse        = mse_values[best_model_name]
best_pred       = models[best_model_name]

# ============================================================
# 6. PLOT 3 — MSE Bar Chart
# ============================================================

bar_colors = ['#2ecc71' if k == best_model_name else '#3498db' for k in mse_values]
plt.figure(figsize=(8, 4))
plt.bar(mse_values.keys(), mse_values.values(), color=bar_colors, edgecolor='white')
plt.title('Figure 3: MSE Comparison of All Candidate Models\n(Green = Best Model)')
plt.xlabel('Model')
plt.ylabel('MSE')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 7. GRADIENT DESCENT ON BEST MODEL
# ============================================================

# Initial coefficients of the best model (Itr1: m=0.30, b=-0.20)
best_coeffs = {
    'Itr1':  (0.30, -0.20), 'Itr2':  (0.40, -0.40),
    'Itr3':  (0.45, -0.45), 'Itr4':  (0.50, -0.50),
    'Itr5':  (0.30, -0.20), 'Itr6':  (1.00, -1.00),
    'Itr7':  (0.35, -0.60), 'Itr8':  (0.25, -0.80),
    'Itr9':  (0.50, -1.20), 'Itr10': (0.60, -0.20),
}

m_init, b_init = best_coeffs[best_model_name]

X = Data
y = target
n = len(X)

# Hyperparameters
learning_rate = 0.01
epochs        = 500

m = m_init
b = b_init
mse_history = []

for epoch in range(epochs):
    # Forward pass
    y_pred = m * X + b

    # Compute MSE loss
    loss = compute_mse(y, y_pred)
    mse_history.append(loss)

    # Compute gradients (partial derivatives of MSE)
    error = y_pred - y
    dm    = (2 / n) * np.dot(error, X)   # dMSE/dm
    db    = (2 / n) * np.sum(error)       # dMSE/db

    # Update parameters
    m = m - learning_rate * dm
    b = b - learning_rate * db

print(f"  Final    : m = {m:.4f}, b = {b:.4f}")
print(f"  Final MSE: {mse_history[-1]:.6f}")

# ============================================================
# 8. PLOT 4 — Gradient Descent Loss Curve
# ============================================================

plt.figure(figsize=(8, 4))
plt.plot(mse_history, color='crimson', linewidth=2)
plt.title(f'Figure 4: Gradient Descent Loss Curve (Starting from {best_model_name})')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 9. PLOT 5 — Before vs After Gradient Descent
# ============================================================

plt.figure(figsize=(8, 4))
plt.plot(X, y, label='Actual sin(x)', color='black', linewidth=2)
plt.plot(X, best_pred, label=f'{best_model_name} (before GD)', color='orange',
         linestyle='--', linewidth=2)
plt.plot(X, m * X + b, label=f'After GD  (m={m:.3f}, b={b:.3f})',
         color='green', linewidth=2)
plt.title(f'Figure 5: Before vs After Gradient Descent ({best_model_name})')
plt.xlabel('Data')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 10. SUMMARY
# ============================================================

print("\n" + "=" * 45)
print("  SUMMARY")
print("=" * 45)
print(f"  Best Model (lowest MSE) : {best_model_name}")
print(f"  MSE before GD           : {best_mse:.4f}")
print(f"  MSE after GD (500 ep)   : {mse_history[-1]:.6f}")
print(f"  Final slope  (m)        : {m:.4f}")
print(f"  Final intercept (b)     : {b:.4f}")
print("=" * 45)
