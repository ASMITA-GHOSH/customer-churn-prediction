import streamlit as st
import pickle
import pandas as pd

# Load model and features
model = pickle.load(open('churn_model.pkl', 'rb'))
features = pickle.load(open('features.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Churn Predictor", layout="centered")

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict churn risk")

# 👉 User Inputs (important ones only)
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

contract = st.selectbox("Contract Type", [0, 1, 2])
internet_service = st.selectbox("Internet Service", [0, 1, 2])

# 👉 Create full input with all features
input_dict = {feature: 0 for feature in features}

# Fill selected values (must match column names exactly)
input_dict['tenure'] = tenure
input_dict['MonthlyCharges'] = monthly_charges
input_dict['TotalCharges'] = total_charges

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Churn ({probability:.2f})")
    else:
        st.success(f"✅ Low Risk of Churn ({probability:.2f})")