import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore') # Suppress warnings from GridSearchCV

# 1. 📁 Load Dataset
df = pd.read_csv("study_habits_features.csv")

# 2. 🧹 Encode Categorical Features
categorical_cols = ['learning_style', 'study_environment']
# Note: The engineered columns and numerical habits/scores are left as is.
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 3. 🎯 Define Features and Target
# Drop identifiers and classification target, keeping only regression target (exam_score)
X = df_encoded.drop(['exam_score', 'student_id', 'Pass_Fail'], axis=1)
y = df_encoded['exam_score']

# 4. 🔀 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. ⚙️ Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. 🧮 Custom accuracy function
def custom_accuracy(y_true, y_pred, tolerance=5):
    """Calculates the percentage of predictions within a given tolerance of the true value."""
    accurate = np.abs(y_true - y_pred) <= tolerance
    return np.mean(accurate) * 100

# 7. 🔥 Gradient Boosting with Hyperparameter Tuning (GB Hyperparameter)
print("Starting Gradient Boosting Hyperparameter Tuning on Study Habits Features...")

# Define the parameter grid to search
param_grid_gb = {
    'n_estimators': [100, 200, 300],  # Number of boosting stages
    'learning_rate': [0.05, 0.1, 0.2], # Contribution of each tree
    'max_depth': [3, 5, 7],           # Depth of the individual regression estimators
    'subsample': [0.7, 0.8]           # Fraction of samples used to fit the base learners
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
print("🚀 Tuned Gradient Boosting Results (Study Habits Focus):")
print(f"Best Parameters: {grid_search_gb.best_params_}")
print("="*70)
print(f"R² Score: {r2_score(y_test, gb_preds):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, gb_preds)):.4f}")
print(f"Custom Accuracy (±5 marks): {custom_accuracy(y_test, gb_preds):.2f}%")
print("="*70)