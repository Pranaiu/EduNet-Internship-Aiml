import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Title and Description
st.title("💧 Water Pollutants Predictor")
st.write("📊 Predict common water pollutant levels based on Year and Station ID using a pre-trained machine learning model.")

# Load the trained model and feature structure
model = joblib.load("/content/afa2e701598d20110228.csv")
model_cols = joblib.load("/content/afa2e701598d20110228.csv")

# List of pollutant labels
pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

# User Input Section
st.sidebar.header("📥 Input Parameters")
year_input = st.sidebar.number_input("Enter Year", min_value=2000, max_value=2100, value=2024)
station_id = st.sidebar.text_input("Enter Station ID (e.g. '1', '5', '22')", value='1')

# Predict Button
if st.sidebar.button('🔍 Predict'):
    if not station_id.strip():
        st.warning('⚠️ Please enter a valid Station ID')
    else:
        # Step 1: Prepare Input Data
        input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])

        # Step 2: Align with model's training columns
        missing_cols = set(model_cols) - set(input_encoded.columns)
        for col in missing_cols:
            input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]  # reorder to match training

        # Step 3: Predict pollutant levels
        predicted_pollutants = model.predict(input_encoded)[0]

        # Step 4: Display results
        st.subheader(f"📡 Predicted pollutant levels for Station ID '{station_id}' in {year_input}:")
        result_dict = {p: round(val, 2) for p, val in zip(pollutants, predicted_pollutants)}
        st.json(result_dict)
