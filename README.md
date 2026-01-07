❤️ Heart Disease Prediction Analysis
This project develops a Machine Learning model to predict the presence of heart disease using clinical data. The primary focus is on Medical Sensitivity (Recall), ensuring that potential heart patients are correctly identified for further testing.

📊 Project Overview
In medical diagnosis, missing a sick patient (False Negative) is more critical than a false alarm. This model is optimized using a Custom Threshold to minimize the risk of missing heart disease cases.

Algorithm: Logistic Regression

Target Accuracy: ~85%

Optimization: Custom Probability Threshold (0.3) & Balanced Class Weights


🏆 Project Milestone
"This is my first publicly showcased Machine Learning project, achieving a robust test accuracy of 85.1%. The model demonstrates a strong balance between precision and sensitivity, specifically optimized for healthcare data constraints."


🛠️ Tech Stack
Language: Python 3.x

Libraries: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

🏗️ Key Workflow
Feature Selection: Dropped noisy features like EKG results, FBS over 120, and Max HR to improve model stability.

Data Preprocessing: * Categorical columns (e.g., Chest pain type) were encoded using One-Hot Encoding.

Numerical features were scaled using StandardScaler to ensure all features contribute equally.

Threshold Engineering: Instead of the default 0.5, a 0.3 threshold was applied to improve the identification of disease cases (Class 1).