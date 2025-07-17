import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("best_model_random_forest.pkl")
scaler = joblib.load("scaler.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")
contract_encoder = joblib.load("contract_encoder.pkl")
expected_columns = joblib.load("feature_columns.pkl")

st.title("Customer Churn Prediction")
st.markdown("""
This app allows you to:
- Enter a single customer's information manually.
- Or upload a CSV file with multiple customers.
The model will predict whether each customer is likely to churn.
""")

option = st.radio("Choose input method:", ("Single Input", "Upload CSV File"))

if option == "Single Input":
    age = st.number_input("Age", 18, 100, 30)
    gender = st.selectbox("Gender", gender_encoder.classes_)
    tenure = st.number_input("Tenure (months)", 0, 120, 12)
    usage = st.number_input("Usage Frequency", 0.0, 100.0, 10.0)
    support = st.number_input("Support Calls", 0, 100, 2)
    delay = st.number_input("Payment Delay", 0.0, 100.0, 0.0)
    contract = st.selectbox("Contract Length", contract_encoder.classes_)
    spend = st.number_input("Total Spend", 0.0, 10000.0, 500.0)
    last_interaction = st.number_input("Last Interaction (days)", 0, 365, 30)
    subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])

    if st.button("Predict"):
        row = {
            "Age": age,
            "Gender": gender_encoder.transform([gender])[0],
            "Tenure": tenure,
            "Usage Frequency": usage,
            "Support Calls": support,
            "Payment Delay": delay,
            "Contract Length": contract_encoder.transform([contract])[0],
            "Total Spend": spend,
            "Last Interaction": last_interaction,
            "Subscription Type_Standard": 1 if subscription == "Standard" else 0,
            "Subscription Type_Premium": 1 if subscription == "Premium" else 0
        }

        df_input = pd.DataFrame([row])

        for col in expected_columns:
            if col not in df_input.columns:
                df_input[col] = 0

        df_input = df_input[expected_columns]
        df_scaled = scaler.transform(df_input)
        pred = model.predict(df_scaled)
        st.success("Prediction: {}".format("Churn" if pred[0] == 1 else "Not Churn"))

elif option == "Upload CSV File":
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Encoding
        df['Gender'] = gender_encoder.transform(df['Gender'])
        df['Contract Length'] = contract_encoder.transform(df['Contract Length'])
        df = pd.get_dummies(df, columns=['Subscription Type'], drop_first=True)

        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        df = df[expected_columns]
        df_scaled = scaler.transform(df)
        predictions = model.predict(df_scaled)

        df_result = df.copy()
        df_result["Prediction"] = ["Churn" if p == 1 else "Not Churn" for p in predictions]
        st.dataframe(df_result)
