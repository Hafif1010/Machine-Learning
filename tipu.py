import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# 1. Load data dan model
df = pd.read_csv("hapip.csv")
model = joblib.load("fraud_model.pkl")

# 2. Encode kolom kategori (harus sama dengan saat training)
df['type'] = LabelEncoder().fit_transform(df['type'])
df['nameOrig'] = LabelEncoder().fit_transform(df['nameOrig'])
df['nameDest'] = LabelEncoder().fit_transform(df['nameDest'])

# 3. Siapkan fitur sesuai dengan model (hilangkan 'step' dan target)
X = df.drop(columns=['isFraud', 'step'])

# 4. Prediksi probabilitas fraud
df['fraud_probability'] = model.predict_proba(X)[:, 1]
df['predicted_isFraud'] = model.predict(X)

# 5. Ambil 100 transaksi dengan probabilitas fraud tertinggi
top_fraud_100 = df.sort_values(by='fraud_probability', ascending=False).head(100)

# 6. Simpan ke CSV (opsional)
top_fraud_100.to_csv("top_100_fraud_transactions.csv", index=False)
print("✅ Disimpan ke 'top_100_fraud_transactions.csv'")
