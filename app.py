import streamlit as st
import pickle
import numpy as np

st.title("Traffic Flow Prediction")

# Debug message
st.write("App started successfully...")

# Load model safely
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    st.success("Model and Scaler loaded successfully")
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")
    st.stop()

# Inputs
coded_day = st.number_input("Coded Day", min_value=0)
zone = st.number_input("Zone", min_value=0)
weather = st.number_input("Weather", min_value=0)
temperature = st.number_input("Temperature")
day = st.number_input("Day of Month")
month = st.number_input("Month")
year = st.number_input("Year")
dayofweek = st.number_input("Day of Week (0=Mon)")

if st.button("Predict Traffic"):
    features = np.array([[coded_day, zone, weather, temperature,
                          day, month, year, dayofweek]])

    features = scaler.transform(features)
    prediction = model.predict(features)

    st.success(f"Predicted Traffic: {round(prediction[0], 2)}")