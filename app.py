import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Telco Churn Prediction", page_icon="📱")

# --- 2. LOAD MODEL & ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    # Pastikan file 'churn_model.pkl' ada di folder yang sama
    try:
        artifacts = joblib.load('churn_model.pkl')
        return artifacts['model'], artifacts['scaler'], artifacts['features']
    except FileNotFoundError:
        return None, None, None

model, scaler, feature_columns = load_artifacts()

if model is None:
    st.error("Error: File 'churn_model.pkl' not found. Please make sure the file is uploaded to GitHub in the same folder as app.py.")
    st.stop()

# --- 3. UI: TITLE & DESCRIPTION ---
st.title("📱 Telco Customer Churn Prediction")
st.write("""
This app uses Machine Learning to predict whether a customer will **Churn** (stop subscribing) or **Not**.
Please fill in the customer parameters below.
""")

# --- 4. INPUT FORM ---
with st.form("churn_form"):
    st.subheader("Customer Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Numeric Features (Matched with CSV columns: tenure, MonthlyCharges, TotalCharges)
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0)
        
        # Demographic Features
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    with col2:
        # Service Features
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        
        # Additional Features (Default values for features present in CSV but not in form)
        device_protection = "No" 
        online_backup = "No"
        streaming_tv = "No"
        streaming_movies = "No"

    submitted = st.form_submit_button("Predict Now")

# --- 5. PREDICTION LOGIC ---
if submitted:
    # A. Create DataFrame from Input
    raw_data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    input_df = pd.DataFrame([raw_data])

    # B. Preprocessing - Scaling Numeric
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    try:
        input_df[num_cols] = scaler.transform(input_df[num_cols])
    except Exception as e:
        st.error(f"An error occurred during data scaling: {e}")
        st.stop()

    # C. Preprocessing - Encoding
    input_encoded = pd.get_dummies(input_df, drop_first=False)

    # D. Align Columns (IMPORTANT!)
    # Match input columns with training columns
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    # E. Prediction
    try:
        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0][1]

        # --- 6. DISPLAY RESULT ---
        st.markdown("---")
        st.subheader("Prediction Result")
        
        if prediction == 1:
            st.error(f"⚠️ **CHURN DETECTED** (Probability: {probability:.2%})")
            st.write("This customer is at high risk of churning.")
        else:
            st.success(f"✅ **NOT CHURN** (Probability: {probability:.2%})")
            st.write("This customer is predicted to stay.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")