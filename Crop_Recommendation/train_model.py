import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("Crop_Recommendation.csv")

# Rename columns for consistency
df.rename(columns={"pH_Value": "pH"}, inplace=True)

# Encode target variable ('Crop')
label_encoder = LabelEncoder()
df["Crop_Encoded"] = label_encoder.fit_transform(df["Crop"])

# Feature Selection (using correct column names)
features = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']
X = df[features]
y = df["Crop_Encoded"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost Model
model = XGBClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

# Model Evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save Model & Preprocessors
with open("models/crop_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# SHAP Analysis (Global Feature Importance)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train_scaled)  # Get SHAP values

# Check SHAP Shape
print(f"Original SHAP Shape: {np.array(shap_values).shape}")

# If SHAP shape is (num_samples, num_features, num_classes), we need to transpose it
if len(np.array(shap_values).shape) == 3:
    shap_values = np.transpose(shap_values, (2, 0, 1))  # Convert to (num_classes, num_samples, num_features)

# Check Corrected Shape
num_classes, num_samples, num_features = shap_values.shape
print(f"Fixed SHAP Shape: {shap_values.shape}")

# Generate SHAP Summary Plot for Multi-Class Models
plt.figure(figsize=(10, 7))

for i in range(num_classes):
    print(f"Generating SHAP Summary Plot for Class {i}...")
    shap.summary_plot(shap_values[i], X_train_scaled, feature_names=features, show=False)

plt.savefig("static/shap_summary.png")
print("SHAP summary plot saved: static/shap_summary.png")
