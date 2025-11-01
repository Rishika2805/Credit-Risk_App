import streamlit as st
import pandas as pd
import joblib

# Load the trained model and preprocessing pipeline
model = joblib.load("credit_risk_model.pkl")
pipeline = joblib.load("credit_risk_pipeline.pkl")
expected_columns = joblib.load("credit_risk_features.pkl")

num_attribs = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
               'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
                'cb_person_default_on_file' 
               ]

cat_attribs = ['person_home_ownership', 'loan_intent', 'loan_grade']

# Streamlit App
st.title("💳 Credit Card Risk Prediction")

st.write("""
          This application predicts the credit risk of a loan applicant based on their 
          financial and personal information. 
          Please fill in the details below to get a prediction.
          """)

# Personal Information Section

st.header("Personal Information")

# Age
person_age = st.slider("Age", 18, 100, 30)

# Annual Income
person_income = st.number_input("Annual Income ($)", min_value=0)

# Home Ownership
person_home_ownership = st.selectbox("Home Ownership", ["OWN", "RENT", "MORTGAGE", "OTHER"])

# Employment Information Section
person_emp_length = st.slider("Years of Employment", 0, 50, 5)

# Loan Information Section
st.subheader("💵 Loan Details")

# Loan Amount
loan_amnt = st.number_input("Loan Amount ($)", min_value=0)

# Loan Purpose
loan_intent = st.selectbox("Loan Intent", [
    "EDUCATION", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT", 
    "MEDICAL", "PERSONAL", "VENTURE"
])

# Interest Rate
loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.1)

# Loan Grade
loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])

loan_percent_income = loan_amnt / (person_income + 1) * 100  # Avoid division by zero

st.subheader("📊 Credit History")
cb_person_default_on_file = st.selectbox(
    "Previous Default on File?",
    options=["Yes", "No"],        # What user sees
    index=1
)

# Map to 1/0 internally
cb_person_default_on_file_val = 1 if cb_person_default_on_file == "Yes" else 0


cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0)


# Prepare input data for prediction

input_data = pd.DataFrame([[
    person_age, person_income, person_home_ownership, person_emp_length, loan_intent, loan_grade,
    loan_amnt, loan_int_rate, 
    loan_percent_income, cb_person_default_on_file_val, cb_person_cred_hist_length,
]], columns=expected_columns) 

# Ensure all expected columns are present
for col in expected_columns:
    if col not in input_data.columns:
        input_data[col] = 0  # or appropriate default value

# Reorder columns to match training data
input_data = input_data[expected_columns]

# Preprocess the input data

if st.button("Predict Credit Risk"):
    input_prepared = pipeline.transform(input_data)

    # Make prediction
    prediction = model.predict(input_prepared)[0]
    prediction_proba = model.predict_proba(input_prepared)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Default! (Confidence: {prediction_proba:.2%})")
    else:
        st.success(f"✅ Low Risk of Default. (Confidence: {prediction_proba:.2%})")

