import streamlit as st
import pandas as pd
import joblib

# Page setup must be near the top
st.set_page_config(
    page_title="Bank Term Deposit Prediction",
    page_icon="🏦",
    layout="wide"
)

# Load trained model
model = joblib.load("bank_marketing_model.pkl")

# Custom CSS to make the app fit the browser better
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    h1 {
        text-align: center;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏦 Bank Term Deposit Prediction App")

st.markdown(
    """
    <p class="subtitle">
    Predict whether a bank customer is likely to subscribe to a term deposit
    based on customer profile, financial details, and marketing campaign information.
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

st.header("Enter Customer Details")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Customer Profile")

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35
        )

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

    with col2:
        st.subheader("Financial Details")

        balance = st.number_input(
            "Account Balance",
            value=1500
        )

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

        poutcome = st.selectbox(
            "Previous Campaign Outcome",
            ["unknown", "failure", "other", "success"]
        )

    with col3:
        st.subheader("Campaign Details")

        day = st.number_input(
            "Last Contact Day of Month",
            min_value=1,
            max_value=31,
            value=15
        )

        month = st.selectbox(
            "Last Contact Month",
            [
                "jan", "feb", "mar", "apr", "may", "jun",
                "jul", "aug", "sep", "oct", "nov", "dec"
            ]
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

    submitted = st.form_submit_button("Predict Subscription", use_container_width=True)

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

# Apply the same feature engineering used during model training
input_data["was_contacted_before"] = input_data["pdays"].apply(
    lambda x: 0 if x == -1 else 1
)

input_data["has_any_loan"] = input_data.apply(
    lambda row: 1 if row["housing"] == "yes" or row["loan"] == "yes" else 0,
    axis=1
)

input_data["campaign_intensity"] = pd.cut(
    input_data["campaign"],
    bins=[0, 1, 3, 10, 100],
    labels=["single_contact", "low_contact", "medium_contact", "high_contact"]
)

st.divider()

if submitted:
    prediction = model.predict(input_data)[0]

    try:
        probability = model.predict_proba(input_data)[0][1]
    except:
        probability = None

    result_col1, result_col2 = st.columns([1, 1])

    with result_col1:
        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("The customer is likely to subscribe to a term deposit.")
        else:
            st.error("The customer is unlikely to subscribe to a term deposit.")

        if probability is not None:
            st.metric(
                label="Probability of Subscribing",
                value=f"{probability:.2%}"
            )

    with result_col2:
        st.subheader("Customer Data Used")
        st.dataframe(input_data, use_container_width=True)

st.caption("Machine Learning Model: Tuned Random Forest Classifier")