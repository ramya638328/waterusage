import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Water Usage Prediction")

MODEL_FILE = "waterusage_model.pkl"

st.title("💧 Water Usage Prediction App")

# Check file exists
if not os.path.exists(MODEL_FILE):
    st.error("❌ Model file not found. Upload 'waterusage_model.pkl' to the repo.")
    st.stop()

# Load PKL
with open(MODEL_FILE, "rb") as file:
    loaded_obj = pickle.load(file)

# 🚨 CASE 1: PKL is a DataFrame (WRONG FILE)
if isinstance(loaded_obj, pd.DataFrame):
    st.error("❌ Uploaded PKL contains a dataset, not a trained ML model.")
    st.info("✔ Please re-save ONLY the trained model from Google Colab.")
    st.stop()

# 🚨 CASE 2: PKL is a dictionary
elif isinstance(loaded_obj, dict):
    if "model" not in loaded_obj:
        st.error("❌ Dictionary PKL does not contain a 'model' key.")
        st.stop()
    model = loaded_obj["model"]

# ✅ CASE 3: Proper ML model or Pipeline
else:
    model = loaded_obj

# Inputs
members = st.number_input("Number of Family Members", min_value=1)
water_today = st.number_input("Water Used Today (Liters)", min_value=0.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0)

# Prediction
if st.button("Predict Tomorrow's Usage"):
    try:
        input_data = np.array([[members, water_today, temperature]])
        prediction = model.predict(input_data)
        st.success(f"✅ Estimated Water Usage: {prediction[0]:.2f} Liters")
    except Exception as e:
        st.error("❌ Prediction failed. Model input shape mismatch.")
