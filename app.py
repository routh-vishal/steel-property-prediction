import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from utils.feature_engineering import enhanced_features
# Page Configuration
st.set_page_config(
    page_title="Steel Property Predictor",
    page_icon="⚙️",
    layout="centered"
)

# Load Pre-trained Models
ts_model = joblib.load("./models/Ridge_TS_poly_final.joblib")
elongation_model = joblib.load("./models/XGB_Elongation_final.joblib")

# App Header
st.title("⚙️ Steel Property Predictor")
st.markdown("""
### Predict Mechanical Properties of Low-Alloy Steels
Provide the **composition** and **temperature** to estimate:
- **Tensile Strength (MPa)**
- **Elongation (%)**
""")

# Input Section
col1, col2 = st.columns(2)

with col1:
    temperature = st.slider("Temperature (°C)", 0, 1200, 25)
    carbon = st.slider("Carbon (C) %", 0.0, 1.0, 0.2, 0.01)
    manganese = st.slider("Manganese (Mn) %", 0.0, 2.0, 0.5, 0.01)

with col2:
    chromium = st.slider("Chromium (Cr) %", 0.0, 2.0, 0.5, 0.01)
    nickel = st.slider("Nickel (Ni) %", 0.0, 2.0, 0.3, 0.01)
    aluminum = st.slider("Aluminum (Al) %", 0.0, 1.0, 0.1, 0.01)

import pandas as pd
import numpy as np

# Prediction
if st.button("🔍 Predict Properties"):
    # Make sure to pass values in the correct order
    input_values = [[carbon, manganese, chromium, nickel, aluminum, temperature]]
    feature_columns = ['C','Mn','Cr','Ni','Al','Temperature_C'] 
    # Convert to DataFrame
    input_features = pd.DataFrame(input_values, columns=feature_columns)

    # Predict using both models
    ts_pred = ts_model.predict(input_features)[0]
    el_pred = elongation_model.predict(input_features)[0]

    # Display Results
    st.subheader("🧾 Predicted Mechanical Properties")
    colA, colB = st.columns(2)
    colA.metric("Tensile Strength (MPa)", f"{ts_pred:.2f}")
    colB.metric("Elongation (%)", f"{el_pred:.2f}")

    st.success("✅ Prediction complete!")
