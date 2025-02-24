import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open("loan_approval_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the trained scaler
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Load feature names used during training
with open("feature_names.pkl", "rb") as feature_file:
    feature_names = pickle.load(feature_file)

# Streamlit UI
st.title("💳 Loan Approval Prediction System")
st.markdown("Enter your details below to check if your loan will be approved.")

st.sidebar.header("User Input Parameters")

def user_input():
    person_age = st.sidebar.slider("Person Age", 18, 100, 30)

    person_gender = st.sidebar.radio("Gender", ["Male", "Female"])
    person_gender = 1 if person_gender == "Male" else 0  

    person_education = st.sidebar.selectbox("Education Level", ["Master", "High School", "Bachelor", "Associate", "Doctorate"])
    education_map = {"Master": 1, "High School": 2, "Bachelor": 3, "Associate": 4, "Doctorate": 5}
    person_education = education_map[person_education]

    person_income = st.sidebar.number_input("Annual Income (USD)", 10000, 1000000, 50000)
    
    person_emp_exp = st.sidebar.slider("Employment Experience (Years)", 0, 40, 5)
    
    home_ownership = st.sidebar.selectbox("Home Ownership", ["Rent", "Mortgage", "Own", "Other"])
    home_ownership_map = {"Rent": 1, "Mortgage": 2, "Own": 3, "Other": 4}
    person_home_ownership = home_ownership_map[home_ownership]

    loan_amnt = st.sidebar.number_input("Loan Amount (USD)", 500, 50000, 10000)

    loan_intent = st.sidebar.selectbox("Loan Purpose", ["Education", "Medical", "Venture", "Personal", "Debt Consolidation", "Home Improvement"])
    loan_intent_map = {"Education": 1, "Medical": 2, "Venture": 3, "Personal": 4, "Debt Consolidation": 5, "Home Improvement": 6}
    loan_intent = loan_intent_map[loan_intent]

    loan_int_rate = st.sidebar.slider("Loan Interest Rate (%)", 0.0, 30.0, 10.0)
    loan_percent_income = st.sidebar.slider("Loan Percent Income", 0.0, 1.0, 0.2)

    cb_person_cred_hist_length = st.sidebar.slider("Credit History Length (Years)", 1, 30, 10)
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)

    previous_defaults = st.sidebar.radio("Previous Loan Defaults", ["Yes", "No"])
    previous_loan_defaults_on_file = 1 if previous_defaults == "Yes" else 0

    # Create DataFrame
    data = {
        "person_age": person_age,
        "person_gender": person_gender,
        "person_education": person_education,
        "person_income": person_income,
        "person_emp_exp": person_emp_exp,
        "person_home_ownership": person_home_ownership,
        "loan_amnt": loan_amnt,
        "loan_intent": loan_intent,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_cred_hist_length": cb_person_cred_hist_length,
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file
    }
    
    return pd.DataFrame([data])

df = user_input()

# Ensure feature order matches training
df = df[feature_names]  # ✅ Fix: Match feature names exactly

# Apply Scaling (Use the same scaler from training)
df_scaled = scaler.transform(df)

# Display user input
st.subheader("🔍 User Input")
st.write(df)

# Predict when button is clicked
if st.button("🔮 Predict Loan Status"):
    prediction = model.predict(df_scaled)
    prediction_proba = model.predict_proba(df_scaled)[:, 1]

    st.subheader("📊 Prediction Result")
    if prediction[0] == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Denied.")
    
    st.write(f"Confidence Score: {prediction_proba[0]:.2f}")
