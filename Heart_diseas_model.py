"""
Project Name: Heart Disease Prediction Analysis
Author: [Abhay kumar]
Model: Logistic Regression (Optimized with Custom Threshold)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------------------------------
# 1. Data Loading & Cleaning
# ---------------------------------------------------------
data = pd.read_csv("Heart_Disease_Prediction.csv")
df = pd.DataFrame(data)

# Dropping less significant features to reduce noise
df = df.drop(['EKG results', 'FBS over 120', 'Exercise angina', 'Max HR'], axis=1)

# Target Encoding: Presence = 1, Absence = 0
df['Heart Disease'] = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})

# ---------------------------------------------------------
# 2. Exploratory Data Analysis (EDA) & Encoding
# ---------------------------------------------------------
# Identifying categorical columns for One-Hot Encoding
categorical_cols = [col for col in df.columns if 2 < df[col].nunique() < 10]

# Applying One-Hot Encoding
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

print("all columns of df :-\n",df.columns)
# ---------------------------------------------------------
# 3. Data Splitting & Scaling
# ---------------------------------------------------------
X = df.drop('Heart Disease', axis=1)
y = df['Heart Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 4. Model Training (Logistic Regression)
# ---------------------------------------------------------
# Using class_weight='balanced' to handle potential class imbalance
model = LogisticRegression(max_iter=1000, C=0.9, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------
# 5. Evaluation with Custom Threshold (0.3)
# ---------------------------------------------------------
# Getting probabilities for the positive class
y_probs = model.predict_proba(X_test_scaled)[:, 1]

# Setting a sensitive threshold for medical safety
custom_threshold = 0.3
y_pred_final = (y_probs >= custom_threshold).astype(int)

# Metrics Calculation
print(f"--- Final Evaluation (Threshold: {custom_threshold}) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_final):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_final))

# ---------------------------------------------------------
# 6. Visualizations for GitHub Portfolio
# ---------------------------------------------------------

# A. Confusion Matrix Heatmap
plt.figure(figsize=(8, 5))
cm = confusion_matrix(y_test, y_pred_final)
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
            xticklabels=['Healthy', 'Disease'], yticklabels=['Healthy', 'Disease'])
plt.title(f'Confusion Matrix (Threshold {custom_threshold})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# B. Feature Importance Plot
importance = model.coef_[0]
feat_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
feat_importance = feat_importance.sort_values(by='Importance', ascending=True)

import joblib
# Model aur Scaler dono save karein
joblib.dump(model, 'heart_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

#ploting
plt.figure(figsize=(10, 6))
plt.barh(feat_importance['Feature'], feat_importance['Importance'], color='teal')
plt.title('Feature Importance (Logistic Regression Coefficients)')
plt.xlabel('Coefficient Value')
plt.show()


