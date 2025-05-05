# model/train_model.py

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load the dataset
df = pd.read_csv("../producer/creditcard.csv")  # Make sure the CSV is here

# Prepare features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Handle class imbalance
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, "../api/fraud_model.pkl")
