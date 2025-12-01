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
df = pd.read_csv("enhanced_cs_features.csv")

# =============================================================
# 2. Encode Categorical Features
# =============================================================
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# =============================================================
# 3. Define Features and Target
# =============================================================
X = df_encoded.drop(
    ['exam_score', 'student_id', 'major_Computer Science', 'Pass_Fail'],
    axis=1, errors='ignore'
)

y = df_encoded['exam_score']

print(f"Total features used in the model: {X.shape[1]}")

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
# 6. Custom Accuracy
# =============================================================
def custom_accuracy(y_true, y_pred, tolerance=5):
    accurate = np.abs(y_true - y_pred) <= tolerance
    return np.mean(accurate) * 100


# =============================================================
# 7. Hyperparameter Tuning: SVM (SVR)
# =============================================================
print("\n🔍 Starting SVR Hyperparameter Tuning...")

param_grid_svr = {
    'C': [1, 10, 50],
    'kernel': ['rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

svr = SVR()

grid_svr = GridSearchCV(
    estimator=svr,
    param_grid=param_grid_svr,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_svr.fit(X_train_scaled, y_train)
best_svr = grid_svr.best_estimator_

print("\nBest SVR Parameters:", grid_svr.best_params_)


# =============================================================
# 8. Stacking Framework (SF Ensemble)
# =============================================================
print("\n🚀 Training Stacking Ensemble...")

stacking_model = StackingRegressor(
    estimators=[
        ('svr', best_svr),
        ('gbr', GradientBoostingRegressor(random_state=42)),
        ('rf', RandomForestRegressor(random_state=42))
    ],
    final_estimator=GradientBoostingRegressor(random_state=42)
)

stacking_model.fit(X_train_scaled, y_train)
stack_preds = stacking_model.predict(X_test_scaled)


# =============================================================
# 9. Voting Ensemble
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
def evaluate_model(name, y_test, y_pred):
    print("\n" + "="*70)
    print(f"📊 {name} RESULTS")
    print("="*70)
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"Custom Accuracy (±5 marks): {custom_accuracy(y_test, y_pred):.2f}%")
    print("="*70)


# =============================================================
# 11. Show Results
# =============================================================
evaluate_model("Best SVR", y_test, best_svr.predict(X_test_scaled))
evaluate_model("Stacking Ensemble", y_test, stack_preds)
evaluate_model("Voting Ensemble", y_test, voting_preds)
