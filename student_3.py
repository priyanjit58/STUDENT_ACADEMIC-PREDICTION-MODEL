import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore') # Suppress warnings from GridSearchCV

# 1. 📁 Load and Filter Dataset
df = pd.read_csv("filtered_students_by_major.csv")

# 💻 Filter data for 'Computer Science' major only
df_cs = df[df['major'] == 'Computer Science'].copy()

# 2. 🧹 Encode Categorical Features
categorical_cols = [
    'gender', 'part_time_job', 'diet_quality', 'exercise_frequency',
    'parental_education_level', 'internet_quality', 'mental_health_rating',
    'extracurricular_participation', 'semester', 'dropout_risk',
    'social_activity', 'study_environment', 'access_to_tutoring',
    'family_income_range', 'parental_support_level', 'learning_style'
]
# Exclude 'major' as it's now constant ('Computer Science')
df_encoded = pd.get_dummies(df_cs.drop('major', axis=1), columns=categorical_cols, drop_first=True)

# 3. 🎯 Define Features and Target
# Drop 'student_id' as it's just an identifier
X = df_encoded.drop(['exam_score', 'student_id'], axis=1)
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

# 7. 🌲 Random Forest with Hyperparameter Tuning (Grid Search)
print("Starting Random Forest Hyperparameter Tuning...")

# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],  # None means nodes are expanded until all leaves are pure
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)

# Use R2 score as the primary optimization metric
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=3, scoring='r2', verbose=1, n_jobs=-1)

grid_search.fit(X_train_scaled, y_train)

# Get the best model
best_rf = grid_search.best_estimator_
rf_preds = best_rf.predict(X_test_scaled)

# 8. 📊 Evaluation of Best Model
print("\n" + "="*50)
print("👑 Tuned Random Forest (Computer Science Students Only) Results:")
print(f"Best Parameters: {grid_search.best_params_}")
print("="*50)
print(f"R² Score: {r2_score(y_test, rf_preds):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, rf_preds)):.4f}")
print(f"Custom Accuracy (±5 marks): {custom_accuracy(y_test, rf_preds):.2f}%")
print("="*50)

# Optional: You can also try Gradient Boosting with Hyperparameter Tuning
# gb = GradientBoostingRegressor(random_state=42)
# param_grid_gb = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]}
# ... (Follow similar GridSearchCV steps)