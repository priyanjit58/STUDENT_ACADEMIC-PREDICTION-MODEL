import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
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

# 7. 🔥 Gradient Boosting with Hyperparameter Tuning (Grid Search)
print("Starting Gradient Boosting Hyperparameter Tuning...")

# Define the parameter grid to search for Gradient Boosting
param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8] # Use a fraction of the data for each tree (stochastic GB)
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
print("\n" + "="*50)
print("🚀 Tuned Gradient Boosting (Computer Science Students Only) Results:")
print(f"Best Parameters: {grid_search_gb.best_params_}")
print("="*50)
print(f"R² Score: {r2_score(y_test, gb_preds):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, gb_preds)):.4f}")
print(f"Custom Accuracy (±5 marks): {custom_accuracy(y_test, gb_preds):.2f}%")
print("="*50)