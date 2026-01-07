import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Step 1: Model & Scaler Load Karein ---
try:
    model = joblib.load('heart_model.pkl')
    scaler = joblib.load('scaler.pkl')
except:
    st.error("Error: 'heart_model.pkl' or 'scaler.pkl' not found. Training script run karein!")

# --- Step 2: UI Design ---
st.set_page_config(page_title="Heart Health AI", page_icon="🏥")
st.title("🏥 Heart Disease Prediction System")
st.write("Fill in the patient details to check heart disease risk.")

# --- Step 3: User Inputs (Asaan Inputs) ---
st.sidebar.header("Patient Medical Data")

age = st.sidebar.number_input("Age", 1, 120, 45)
sex = st.sidebar.selectbox("Sex", options=[(1, "Male"), (0, "Female")], format_func=lambda x: x[1])[0]
bp = st.sidebar.number_input("Resting BP", 80, 200, 120)
chol = st.sidebar.number_input("Serum Cholesterol", 100, 600, 240)
st_dep = st.sidebar.number_input("ST Depression", 0.0, 10.0, 1.0)

# Categorical Inputs (Inhe baad mein One-Hot mein convert karenge)
cp_type = st.sidebar.selectbox("Chest Pain Type", [1, 2, 3, 4])
slope = st.sidebar.selectbox("Slope of ST Segment", [1, 2, 3])
vessels = st.sidebar.selectbox("Number of Major Vessels", [0, 1, 2, 3])
thal = st.sidebar.selectbox("Thallium Test Result", [3, 6, 7])

# --- Step 4: Logic to Match Your Columns ---
if st.button("Predict Results"):
    # Pehle ek dictionary banayein sabhi 15 columns ke liye (Heart Disease chhod kar)
    # Sabhi ko 0 se initialize karein
    features = {
        'Age': age, 'Sex': sex, 'BP': bp, 'Cholesterol': chol, 'ST depression': st_dep,
        'Chest pain type_2': 0, 'Chest pain type_3': 0, 'Chest pain type_4': 0,
        'Slope of ST_2': 0, 'Slope of ST_3': 0,
        'Number of vessels fluro_1': 0, 'Number of vessels fluro_2': 0, 'Number of vessels fluro_3': 0,
        'Thallium_6': 0, 'Thallium_7': 0
    }

    # One-Hot Encoding Logic: Jo user ne select kiya use 1 kar do
    if cp_type == 2: features['Chest pain type_2'] = 1
    if cp_type == 3: features['Chest pain type_3'] = 1
    if cp_type == 4: features['Chest pain type_4'] = 1

    if slope == 2: features['Slope of ST_2'] = 1
    if slope == 3: features['Slope of ST_3'] = 1

    if vessels == 1: features['Number of vessels fluro_1'] = 1
    if vessels == 2: features['Number of vessels fluro_2'] = 1
    if vessels == 3: features['Number of vessels fluro_3'] = 1

    if thal == 6: features['Thallium_6'] = 1
    if thal == 7: features['Thallium_7'] = 1

    # DataFrame banayein
    input_df = pd.DataFrame([features])

    # Scaling
    input_scaled = scaler.transform(input_df)

    # Prediction (Threshold = 0.3)
    prob = model.predict_proba(input_scaled)[:, 1][0]
    prediction = 1 if prob >= 0.3 else 0

    # Result Display
    st.divider()
    if prediction == 1:
        st.error(f"⚠️ HIGH RISK: Disease likely present (Probability: {prob*100:.1f}%)")
        st.write("Please consult a cardiologist immediately.")
    else:
        st.success(f"✅ LOW RISK: No disease detected (Probability: {prob*100:.1f}%)")
        st.write("Patient seems healthy based on current parameters.")

    # Visualization: Gauge Chart (Optional visual)
    # st.progress(prob)