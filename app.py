import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Water Usage Prediction")

# Load trained model
with open("waterusage_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("💧 Water Usage Prediction App")

members = st.number_input("Number of Family Members", min_value=1)
water_today = st.number_input("Water Used Today (Liters)", min_value=0.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0)

if st.button("Predict Tomorrow's Usage"):
    input_data = np.array([[members, water_today, temperature]])
    prediction = model.predict(input_data)
    st.success(f"Estimated Water Usage: {prediction[0]:.2f} Liters")
