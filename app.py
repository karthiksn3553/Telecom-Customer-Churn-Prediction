import streamlit as st
import pandas as pd
import pickle

# 1. Load the ONE model file
model = pickle.load(open('churn_xgb_model.pkl', 'rb'))

# The model remembers its own columns from training!
expected_columns = model.feature_names_in_

# Title
st.title("📶 Telecom Customer Churn Predictor")
st.write("Predict whether a customer is at risk of canceling their subscription.")

# 2. Collect User Input
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ['Female', 'Male'])
    senior_citizen = st.selectbox("Senior Citizen?", [0, 1])
    partner = st.selectbox("Has Partner?", ['Yes', 'No'])
    dependents = st.selectbox("Has Dependents?", ['Yes', 'No'])
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)

with col2:
    contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    payment_method = st.selectbox("Payment Method", [
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=120.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=600.0)

# 3. Create DataFrame from input
input_dict = {
    'gender': [gender],
    'SeniorCitizen': [senior_citizen],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'Contract': [contract],
    'InternetService': [internet_service],
    'PaymentMethod': [payment_method],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
}
input_df = pd.DataFrame(input_dict)

if st.button("Predict Churn Risk"):
    # 4. Encode input exactly like the training data
    input_encoded = pd.get_dummies(input_df)

    # 5. Align columns with the model's expected columns (fills missing with 0)
    input_aligned = input_encoded.reindex(columns=expected_columns, fill_value=0)

    # 6. Predict directly (No scaling needed!)
    prediction = model.predict(input_aligned)[0]
    probability = model.predict_proba(input_aligned)[0][1]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ High Risk of Churn! (Probability of leaving: {probability:.2%})")
    else:
        st.success(f"✅ Customer is likely to stay. (Probability of leaving: {probability:.2%})")