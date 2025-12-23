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

# --- 4. INPUT FORM (YANG SUDAH DIPERBAIKI) ---
with st.form("churn_form"):
    st.subheader("Data Pelanggan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fitur Numerik
        tenure = st.number_input("Lama Berlangganan (Bulan)", min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input("Biaya Bulanan ($)", min_value=0.0, value=50.0)
        total_charges = st.number_input("Total Biaya ($)", min_value=0.0, value=500.0)
        
        # Fitur Demografis
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Memiliki Partner", ["Yes", "No"])
        dependents = st.selectbox("Memiliki Tanggungan", ["Yes", "No"])

    with col2:
        # Fitur Layanan (LENGKAP)
        phone_service = st.selectbox("Layanan Telepon", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"]) # DITAMBAHKAN
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"]) # DITAMBAHKAN
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"]) # DITAMBAHKAN
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"]) # DITAMBAHKAN
        
        contract = st.selectbox("Kontrak", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Metode Pembayaran", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    submitted = st.form_submit_button("Prediksi Sekarang")

# --- 5. PREDICTION LOGIC (YANG SUDAH DIPERBAIKI) ---
if submitted:
    # A. Buat DataFrame
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
        'OnlineBackup': online_backup,        # Nilai dari form
        'DeviceProtection': device_protection, # Nilai dari form
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,          # Nilai dari form
        'StreamingMovies': streaming_movies,  # Nilai dari form
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    input_df = pd.DataFrame([raw_data])

    # B. Scaling
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    try:
        input_df[num_cols] = scaler.transform(input_df[num_cols])
    except Exception as e:
        st.error(f"Error scaling: {e}")
        st.stop()

    # C. Encoding - PERBAIKAN PENTING DI SINI
    # Gunakan drop_first=False agar data tidak hilang saat prediksi single row
    input_encoded = pd.get_dummies(input_df, drop_first=False)

    # D. Align Columns
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    # E. Prediksi
    try:
        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0][1]

        st.markdown("---")
        st.subheader("Hasil Prediksi")
        
        if prediction == 1:
            st.error(f"⚠️ **CHURN DETECTED** (Probabilitas: {probability:.2%})")
            st.write("Pelanggan berisiko churn.")
        else:
            st.success(f"✅ **NOT CHURN** (Probabilitas: {probability:.2%})")
            st.write("Pelanggan diprediksi aman.")
            
    except Exception as e:
        st.error(f"Error prediction: {e}")