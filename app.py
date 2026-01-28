import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Water Usage Prediction")

with open("waterusage_pre.pkl", "rb") as file:
    data = pickle.load(file)

model = data["model"]   # ✅ extract model

st.title("💧 Water Usage Prediction")

members = st.number_input("Family Members", min_value=1)
water_today = st.number_input("Water Used Today (Liters)", min_value=0.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0)

if st.button("Predict"):
    input_data = np.array([[members, water_today, temperature]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Water Usage: {prediction[0]:.2f} Liters")
