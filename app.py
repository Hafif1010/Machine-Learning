import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# === Load Model ===
model = joblib.load("fraud_model.pkl")
st.set_page_config(page_title="Fraud Detection", layout="centered")

# === UI Header ===
st.title("💳 Fraud Transaction Detector")
st.markdown("Masukkan data transaksi atau upload file untuk mendeteksi apakah transaksi mengandung penipuan atau tidak.")

# === Input Mode ===
mode = st.radio("Pilih metode input data:", ["Manual", "Upload CSV"])

# === Type Encoding Map (harus sama seperti training) ===
type_map = {'PAYMENT': 0, 'TRANSFER': 1, 'CASH_OUT': 2, 'DEBIT': 3, 'CASH_IN': 4}

# === Manual Input ===
if mode == "Manual":
    st.subheader("🔧 Input Manual")
    
    type_input = st.selectbox("Tipe Transaksi", list(type_map.keys()))
    amount = st.number_input("Jumlah Transaksi")
    oldbalanceOrg = st.number_input("Saldo Awal Pengirim")
    newbalanceOrig = st.number_input("Saldo Akhir Pengirim")
    oldbalanceDest = st.number_input("Saldo Awal Penerima")
    newbalanceDest = st.number_input("Saldo Akhir Penerima")

    if st.button("🔍 Prediksi"):
        input_data = pd.DataFrame([[
            type_map[type_input], amount, 0, oldbalanceOrg,
            newbalanceOrig, 0, oldbalanceDest, newbalanceDest
        ]], columns=[
            'type', 'amount', 'nameOrig', 'oldbalanceOrg',
            'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest'
        ])
        
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if pred == 1:
            st.error(f"🚨 Transaksi ini kemungkinan **Penipuan** dengan probabilitas {prob:.2%}")
        else:
            st.success(f"✅ Transaksi ini **Aman** dengan probabilitas {1 - prob:.2%}")

# === CSV Upload Input ===
else:
    st.subheader("📁 Upload File CSV")
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Preview Data", df.head())

        # Kolom yang dipakai oleh model (tanpa 'step' dan 'isFlaggedFraud')
        expected_cols = ['type', 'amount', 'nameOrig', 'oldbalanceOrg',
                         'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']

        df['type'] = df['type'].map(type_map)
        df['nameOrig'] = pd.factorize(df['nameOrig'])[0]
        df['nameDest'] = pd.factorize(df['nameDest'])[0]
        
        X = df[expected_cols]
        predictions = model.predict(X)
        proba = model.predict_proba(X)[:, 1]

        df['Predicted_isFraud'] = predictions
        df['Probability_Fraud'] = proba

        st.subheader("📊 Hasil Prediksi")
        st.write(df[['amount', 'Predicted_isFraud', 'Probability_Fraud']].head(10))

        fraud_count = df['Predicted_isFraud'].value_counts()
        labels = fraud_count.index.map({0: 'Aman', 1: 'Penipuan'}).tolist()
        colors = [ '#00cc99' if i == 0 else '#ff4d4d' for i in fraud_count.index ]

        fig, ax = plt.subplots()
        ax.pie(fraud_count, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error saat memproses file: {e}")
