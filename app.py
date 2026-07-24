import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("bank_marketing_model.pkl")

# Page settings
st.set_page_config(
    page_title="Bank Term Deposit Prediction",
    page_icon="🏦",
    layout="centered"
)

# App title
st.title("🏦 Bank Term Deposit Prediction App")

st.write("""
This app predicts whether a bank customer is likely to subscribe to a term deposit
based on customer profile, financial details, and marketing campaign information.
""")

st.divider()

st.header("Enter Customer Details")

# User inputs
age = st.number_input("Age", min_value=18, max_value=100, value=35)

job = st.selectbox(
    "Job",
    [
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown"
    ]
)

marital = st.selectbox(
    "Marital Status",
    ["single", "married", "divorced"]
)

education = st.selectbox(
    "Education",
    ["primary", "secondary", "tertiary", "unknown"]
)

default = st.selectbox(
    "Has Credit in Default?",
    ["no", "yes"]
)

balance = st.number_input("Account Balance", value=1500)

housing = st.selectbox(
    "Has Housing Loan?",
    ["no", "yes"]
)

loan = st.selectbox(
    "Has Personal Loan?",
    ["no", "yes"]
)

contact = st.selectbox(
    "Contact Communication Type",
    ["cellular", "telephone", "unknown"]
)

day = st.number_input("Last Contact Day of Month", min_value=1, max_value=31, value=15)

month = st.selectbox(
    "Last Contact Month",
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"]
)

campaign = st.number_input(
    "Number of Contacts During This Campaign",
    min_value=1,
    max_value=50,
    value=2
)

pdays = st.number_input(
    "Days Since Previous Contact",
    min_value=-1,
    max_value=999,
    value=-1
)

previous = st.number_input(
    "Number of Previous Contacts",
    min_value=0,
    max_value=50,
    value=0
)

poutcome = st.selectbox(
    "Previous Campaign Outcome",
    ["unknown", "failure", "other", "success"]
)

# Create input dataframe
input_data = pd.DataFrame({
    "age": [age],
    "job": [job],
    "marital": [marital],
    "education": [education],
    "default": [default],
    "balance": [balance],
    "housing": [housing],
    "loan": [loan],
    "contact": [contact],
    "day": [day],
    "month": [month],
    "campaign": [campaign],
    "pdays": [pdays],
    "previous": [previous],
    "poutcome": [poutcome]
})

st.divider()

# Prediction button
if st.button("Predict Subscription"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success("Prediction: The customer is likely to subscribe to a term deposit.")
    else:
        st.error("Prediction: The customer is unlikely to subscribe to a term deposit.")

    st.write(f"Probability of subscribing: **{probability:.2%}**")

    st.subheader("Customer Data Used for Prediction")
    st.dataframe(input_data)

st.divider()

st.caption("Machine Learning Model: Tuned Random Forest Classifier")