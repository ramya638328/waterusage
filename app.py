import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(page_title="Water Usage Prediction", layout="centered")

# Load model
with open("waterusage_pre.pkl", "rb") as file:
    model = pickle.load(file)

st.title("💧 Water Usage Prediction App")
st.write("Predict tomorrow's household water usage using ML")

st.markdown("---")

# Example input fields (change names if your dataset is different)
members = st.number_input("Number of Family Members", min_value=1, step=1)
water_today = st.number_input("Water Used Today (Liters)", min_value=0.0, step=10.0)
temperature = st.number_input("Today's Temperature (°C)", min_value=0.0, step=0.5)

if st.button("Predict Tomorrow's Water Usage"):
    input_data = np.array([[members, water_today, temperature]])
    prediction = model.predict(input_data)

    st.success(f"💡 Estimated Water Usage Tomorrow: **{prediction[0]:.2f} Liters**")
