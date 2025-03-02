# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 12:28:05 2025

@author: abasu
"""

import streamlit as st
import pickle
import pandas as pd


# --- Page Configuration ---
st.set_page_config(
    page_title="Bank Deposit Prediction :bank:",
    page_icon=":money_with_wings:",
    layout="wide"
)

# --- Custom CSS for Aesthetics ---
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3, h4 {
        color: #333333;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5em 1em;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with open("logistic_regression_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)
with open("label_encoder.pkl", "rb") as le_file:
    label_encoder = pickle.load(le_file)
with open("columns.pkl", "rb") as col_file:
    training_columns = pickle.load(col_file)

# --- App Title and Description ---
st.title("Bank Deposit Prediction :sparkles:")
st.write("### Predict if a customer will subscribe to a term deposit :moneybag:")
st.markdown("Fill in the customer details below and click **Predict Deposit Outcome**.")

# --- Input Form Container ---
with st.container():
    col1, col2 = st.columns(2)

    # Numerical Inputs in Column 1
    with col1:
        st.subheader("Numerical Features :1234:")
        age = st.number_input("Age", min_value=0, value=30)
        balance = st.number_input("Balance", value=1000)
        day = st.number_input("Day of Month", min_value=1, max_value=31, value=15)
        duration = st.number_input("Call Duration (sec)", value=300)
        campaign = st.number_input("Contacts during Campaign", value=1)
        pdays = st.number_input("Days Passed After Last Contact", value=-1)
        previous = st.number_input("Contacts Before Campaign", value=0)

    # Categorical Inputs in Column 2
    with col2:
        st.subheader("Categorical Features :speech_balloon:")
        job = st.selectbox("Job", options=["admin.", "unknown", "technician", "services", "management", 
                                            "retired", "blue-collar", "unemployed", "entrepreneur", 
                                            "housemaid", "self-employed", "student"])
        marital = st.selectbox("Marital Status", options=["married", "single", "divorced"])
        education = st.selectbox("Education", options=["secondary", "tertiary", "primary", "unknown"])
        default = st.selectbox("Credit in Default?", options=["no", "yes"])
        housing = st.selectbox("Housing Loan?", options=["yes", "no"])
        loan = st.selectbox("Personal Loan?", options=["no", "yes"])
        contact = st.selectbox("Contact Type", options=["unknown", "cellular", "telephone"])
        month = st.selectbox("Last Contact Month", options=["jan", "feb", "mar", "apr", "may", "jun", 
                                                            "jul", "aug", "sep", "oct", "nov", "dec"])
        poutcome = st.selectbox("Previous Campaign Outcome", options=["unknown", "other", "failure", "success"])

# --- Prepare Input Data ---
input_data = {
    "age": age,
    "balance": balance,
    "day": day,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "month": month,
    "poutcome": poutcome
}
input_df = pd.DataFrame([input_data])

# Preprocess numerical features using the saved scaler
num_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
input_df[num_cols] = scaler.transform(input_df[num_cols])

# One-hot encode categorical features
cat_cols = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]
input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

# Reindex to match the training columns (fill missing columns with zeros)
input_df = input_df.reindex(columns=training_columns, fill_value=0)

st.markdown("---")
# --- Prediction Button ---
if st.button("Predict Deposit Outcome :crystal_ball:"):
    prediction = model.predict(input_df)
    prediction_label = label_encoder.inverse_transform(prediction)[0]
    st.success(f"Predicted deposit outcome: **{prediction_label}**")
    st.balloons()



st.markdown("---")
st.write("Thanks for using our app! Have a great day :smiley:")  
