import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV # Meta-learner for Stacking
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore') # Suppress warnings from GridSearchCV and Stacking

# 1. 📁 Load and Filter Dataset
df = pd.read_csv("filtered_students_by_major.csv")

# 💻 Filter data for 'Computer Science' major only
df_cs = df[df['major'] == 'Computer Science'].copy()
df_cs.drop('major', axis=1, inplace=True) # Major is now constant

# 2. 💡 Feature Engineering: Create an Interaction Term
# Suggestion: Stress combined with low study effort can be highly detrimental.
df_cs['Study_Stress_Interaction'] = df_cs['study_hours_per_day'] * df_cs['stress_level']

# 3. 🧹 Encode Categorical Features
categorical_cols = [
    'gender', 'part_time_job', 'diet_quality', 'exercise_frequency',
    'parental_education_level', 'internet_quality', 'mental_health_rating',
    'extracurricular_participation', 'semester', 'dropout_risk',
    'social_activity', 'study_environment', 'access_to_tutoring',
    'family_income_range', 'parental_support_level', 'learning_style'
]
df_encoded = pd.get_dummies(df_cs, columns=categorical_cols, drop_first=True)

# 4. 🎯 Define Features and Target
X = df_encoded.drop(['exam_score', 'student_id'], axis=1)
y = df_encoded['exam_score']

# 5. 🔀 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. ⚙️ Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. 🧮 Custom accuracy function
def custom_accuracy(y_true, y_pred, tolerance=5):
    """Calculates the percentage of predictions within a given tolerance of the true value."""
    accurate = np.abs(y_true - y_pred) <= tolerance
    return np.mean(accurate) * 100

# === 8. Hyperparameter Tuning for Base Models (RF and GB) ===
print("--- 1/3. Finding Best Random Forest Parameters ---")
param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
rf = RandomForestRegressor(random_state=42)
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=3, scoring='r2', n_jobs=-1)
grid_search_rf.fit(X_train_scaled, y_train)
best_rf = grid_search_rf.best_estimator_

print("\n--- 2/3. Finding Best Gradient Boosting Parameters (GB Hyperparameter) ---")
param_grid_gb = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
gb = GradientBoostingRegressor(random_state=42)
grid_search_gb = GridSearchCV(estimator=gb, param_grid=param_grid_gb, cv=3, scoring='r2', n_jobs=-1)
grid_search_gb.fit(X_train_scaled, y_train)
best_gb = grid_search_gb.best_estimator_


# === 9. Implement Stacking Regressor (Advanced Ensembling) ===
print("\n--- 3/3. Training Stacking Regressor ---")

# Define the base estimators (the tuned RF and GB models)
estimators = [
    ('rf', best_rf),
    ('gb', best_gb)
]

# Use a simple Ridge Regression as the meta-learner to combine the predictions
stacking_reg = StackingRegressor(
    estimators=estimators, 
    final_estimator=RidgeCV(cv=5),
    n_jobs=-1
)

stacking_reg.fit(X_train_scaled, y_train)
stacking_preds = stacking_reg.predict(X_test_scaled)


# === 10. Final Evaluation of Stacking Model ===
print("\n" + "="*70)
print("🏆 FINAL STACKING ENSEMBLE MODEL (Tuned RF + Tuned GB) RESULTS:")
print("="*70)
print(f"Best RF Parameters: {grid_search_rf.best_params_}")
print(f"Best GB Parameters: {grid_search_gb.best_params_}")
print("-" * 70)
print(f"R² Score: {r2_score(y_test, stacking_preds):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, stacking_preds)):.4f}")
print(f"Custom Accuracy (±5 marks): {custom_accuracy(y_test, stacking_preds):.2f}%")
print("="*70)