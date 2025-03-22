import streamlit as st
import numpy as np
import joblib

# Load the trained model, scaler, and label encoder
rf_model = joblib.load("rf_model.pkl")  # Ensure the file is in the same directory
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Title of the Web App
st.title("🌾 Crop Prediction System")

# Sidebar for user input
st.sidebar.header("Enter Soil & Weather Parameters")

# Create input fields for user input
N = st.sidebar.number_input("Nitrogen (N)", min_value=0, max_value=100, value=50)
P = st.sidebar.number_input("Phosphorus (P)", min_value=0, max_value=100, value=40)
K = st.sidebar.number_input("Potassium (K)", min_value=0, max_value=100, value=30)
temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=22.5)
humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.2)
ph = st.sidebar.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

# Predict button
if st.sidebar.button("Predict Crop"):
    # Prepare input data
    new_sample = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    new_sample_scaled = scaler.transform(new_sample)

    # Predict using the trained model
    predicted_crop = rf_model.predict(new_sample_scaled)
    predicted_crop_name = label_encoder.inverse_transform(predicted_crop)

    # Display the predicted crop
    st.success(f"🌱 Predicted Crop: **{predicted_crop_name[0]}**")
