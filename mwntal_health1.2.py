import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================
# 1. Load Dataset
# =============================================================
df = pd.read_csv("wellness_mental_health_features.csv")

# =============================================================
# 2. Encode Categorical Features
# =============================================================
categorical_cols = ['diet_quality']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# =============================================================
# 3. Features and Target
# =============================================================
X = df_encoded.drop(['exam_score', 'student_id', 'Pass_Fail'], axis=1)
y = df_encoded['exam_score']

print(f"Total features used: {X.shape[1]}")

# =============================================================
# 4. Train-Test Split
# =============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================================================
# 5. Feature Scaling
# =============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================
# 6. Custom Accuracy Function
# =============================================================
def custom_accuracy(y_true, y_pred, tolerance=5):
    accurate = np.abs(y_true - y_pred) <= tolerance
    return np.mean(accurate) * 100

# =============================================================
# 7. Hyperparameter Tuning — SVR (SVM)
# =============================================================
print("\n🔍 Starting SVR Hyperparameter Tuning...")

svr_params = {
    'C': [1, 10, 50],
    'kernel': ['rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

svr = SVR()

grid_svr = GridSearchCV(
    estimator=svr,
    param_grid=svr_params,
    scoring='r2',
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_svr.fit(X_train_scaled, y_train)
best_svr = grid_svr.best_estimator_

print("\nBest SVR Parameters:", grid_svr.best_params_)

# =============================================================
# 8. Stacking Framework (SF Ensemble)
# =============================================================
print("\n🚀 Training Stacking Ensemble (SF)...")

stack_model = StackingRegressor(
    estimators=[
        ('svr', best_svr),
        ('gbr', GradientBoostingRegressor(random_state=42)),
        ('rf', RandomForestRegressor(random_state=42))
    ],
    final_estimator=GradientBoostingRegressor(random_state=42)
)

stack_model.fit(X_train_scaled, y_train)
stack_preds = stack_model.predict(X_test_scaled)

# =============================================================
# 9. Voting Regressor Ensemble
# =============================================================
print("\n🤝 Training Voting Regressor Ensemble...")

voting_model = VotingRegressor(
    estimators=[
        ('svr', best_svr),
        ('gbr', GradientBoostingRegressor(random_state=42)),
        ('rf', RandomForestRegressor(random_state=42))
    ]
)

voting_model.fit(X_train_scaled, y_train)
voting_preds = voting_model.predict(X_test_scaled)

# =============================================================
# 10. Evaluation Function
# =============================================================
def evaluate(name, y_true, y_pred):
    print("\n" + "="*70)
    print(f"📊 RESULTS — {name}")
    print("="*70)
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"Custom Accuracy (±5 Marks): {custom_accuracy(y_true, y_pred):.2f}%")
    print("="*70)

# =============================================================
# 11. Evaluate All Models
# =============================================================
evaluate("Best SVR", y_test, best_svr.predict(X_test_scaled))
evaluate("Stacking Ensemble (SF)", y_test, stack_preds)
evaluate("Voting Ensemble", y_test, voting_preds)
