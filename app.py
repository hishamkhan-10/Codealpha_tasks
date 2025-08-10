import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

model = pickle.load(open("XGBoost_credit_risk_model.pkl", "rb"))


st.set_page_config(page_title="Credit Risk Prediction App", layout="centered")

st.title("Credit Risk Prediction App")
st.write("Predict whether a customer has **Good Credit** or **Bad Credit** using an XGBoost model.")


option = st.sidebar.radio("Choose Prediction Mode:", ["Single Prediction", "Batch Prediction (CSV)"])


if option == "Single Prediction":
    st.subheader("Enter Customer Details")

    income = st.number_input("Annual Income", min_value=0, value=50000)
    debt = st.number_input("Debt Amount", min_value=0, value=2000)
    credit_history = st.number_input("Credit History (Years)", min_value=0, value=5)
    employed = st.selectbox("Currently Employed?", ["Yes", "No"])

    employed_val = 1 if employed == "Yes" else 0

    input_data = np.array([[income, debt, credit_history, employed_val]])


    if st.button("Predict Credit Risk"):
        prediction = model.predict(input_data)
        result = "Good Credit" if prediction[0] == 1 else "Bad Credit"
        st.success(f"Prediction: {result}")

elif option == "Batch Prediction (CSV)":
    st.subheader("Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:", data.head())


        predictions = model.predict(data)
        data["Prediction"] = ["Good Credit" if pred == 1 else "Bad Credit" for pred in predictions]

        st.write("Predictions:", data)

        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", data=csv, file_name="credit_predictions.csv", mime="text/csv")
