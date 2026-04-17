import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib
import os

# Load data
data = pd.read_csv('CSV.SelvadePrata')

# --- TRATAR DADOS CATEGÓRICOS ---
data = pd.get_dummies(data, drop_first=True)

# Prepare features
X = data.drop(columns=['churn'])
y = data['churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Metrics
metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'f1_score': f1_score(y_test, y_pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, y_proba)
}

print("\n📊 Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/churn_xgboost.pkl')

# Save report
report = classification_report(y_test, y_pred, output_dict=True)

os.makedirs('outputs/reports', exist_ok=True)

with open('outputs/reports/model_metrics.txt', 'w') as f:
    f.write("Model Metrics:\n")
    for metric, value in metrics.items():
        f.write(f"{metric}: {value}\n")

    f.write("\nClassification Report:\n")
    for label, values in report.items():
        f.write(f"{label}: {values}\n")

# Feature importance
importances = model.feature_importances_

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})

importance_df.sort_values(by='Importance', ascending=False, inplace=True)
importance_df.to_csv('outputs/reports/feature_importance.csv', index=False)

print("\n✅ Model training completed and files saved.")
