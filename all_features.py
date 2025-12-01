import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore') # Suppress warnings from GridSearchCV

# 1. 📁 Load Dataset (Contains all features, including engineered ones)
df = pd.read_csv("enhanced_cs_features.csv")

# 2. 🧹 Identify and Encode Categorical Features
# List all object-type columns that are NOT 'major' (which is constant 'Computer Science')
# We also exclude the student_id and the engineered binary target 'Pass_Fail' for the regression task.
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
# Note: 'major' is present in this file but should be constant, so we explicitly drop it later.

# Perform One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 3. 🎯 Define Features (X) and Target (y)
# Target: The continuous 'exam_score'
# Features (X): All other columns except identifiers ('student_id', 'major') and targets ('exam_score', 'Pass_Fail')
X = df_encoded.drop(['exam_score', 'student_id', 'major_Computer Science', 'Pass_Fail'], axis=1, errors='ignore')
y = df_encoded['exam_score']

# Verify the final feature set size
print(f"Total features used in the model: {X.shape[1]}")

# 4. 🔀 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. ⚙️ Feature Scaling
# Scale numerical features for better model convergence
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. 🧮 Custom accuracy function
def custom_accuracy(y_true, y_pred, tolerance=5):
    """Calculates the percentage of predictions within a given tolerance of the true value."""
    accurate = np.abs(y_true - y_pred) <= tolerance
    return np.mean(accurate) * 100

# 7. 🔥 Gradient Boosting with Hyperparameter Tuning (GB Hyperparameter)
print("\nStarting Gradient Boosting Hyperparameter Tuning on ALL Features...")

# Define the parameter grid to search
param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5],
    'subsample': [0.8] # Using a fixed subsample for efficiency
}

gb = GradientBoostingRegressor(random_state=42)

# Use R2 score as the primary optimization metric
grid_search_gb = GridSearchCV(estimator=gb, param_grid=param_grid_gb,
                              cv=3, scoring='r2', verbose=1, n_jobs=-1)

grid_search_gb.fit(X_train_scaled, y_train)

# Get the best model
best_gb = grid_search_gb.best_estimator_
gb_preds = best_gb.predict(X_test_scaled)

# 8. 📊 Evaluation of Best Model
print("\n" + "="*70)
print("🚀 Tuned Gradient Boosting Results (ALL FEATURES):")
print(f"Best Parameters: {grid_search_gb.best_params_}")
print("="*70)
print(f"R² Score: {r2_score(y_test, gb_preds):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, gb_preds)):.4f}")
print(f"Custom Accuracy (±5 marks): {custom_accuracy(y_test, gb_preds):.2f}%")
print("="*70)
# use rf,svm , voting ensemblence.

