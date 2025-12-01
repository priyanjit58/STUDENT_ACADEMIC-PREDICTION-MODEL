#research comprising  of major subject computer.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# 📁 Load dataset
df = pd.read_csv(r"D:\research 2025 project\computer_science_students.csv")

# 🧹 Encode categorical features
categorical_cols = [
    'gender', 'major', 'part_time_job', 'diet_quality', 'exercise_frequency',
    'parental_education_level', 'internet_quality', 'mental_health_rating',
    'extracurricular_participation', 'semester', 'dropout_risk',
    'social_activity', 'study_environment', 'access_to_tutoring',
    'family_income_range', 'parental_support_level', 'learning_style'
]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 🎯 Define features and target
X = df_encoded.drop('exam_score', axis=1)
y = df_encoded['exam_score']

# 🔀 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ⚙️ Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🧮 Custom accuracy function
def custom_accuracy(y_true, y_pred, tolerance=5):
    accurate = np.abs(y_true - y_pred) <= tolerance
    return np.mean(accurate) * 100

# 🌲 Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_preds = rf.predict(X_test_scaled)

print("🔍 Random Forest:")
print("R² Score:", r2_score(y_test, rf_preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_preds)))
print(f"Custom Accuracy (±5 marks): {custom_accuracy(y_test, rf_preds):.2f}%")

# 🔥 Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(X_train_scaled, y_train)
gb_preds = gb.predict(X_test_scaled)

print("\n🚀 Gradient Boosting:")
print("R² Score:", r2_score(y_test, gb_preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test, gb_preds)))
print(f"Custom Accuracy (±5 marks): {custom_accuracy(y_test, gb_preds):.2f}%")


    # 🧠 PCA + SVM
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

svm = SVR(kernel='rbf', C=1.0, epsilon=0.2)
svm.fit(X_train_pca, y_train)
svm_preds = svm.predict(X_test_pca)

print("\n📊 SVM with PCA:")
print("R² Score:", r2_score(y_test, svm_preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test, svm_preds)))
print(f"Custom Accuracy (±5 marks): {custom_accuracy(y_test, svm_preds):.2f}%")

