import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

X_test = pd.read_csv(os.path.join(BASE_DIR, 'data', 'test', 'X_test.csv'))
model = joblib.load(os.path.join(BASE_DIR, 'models', 'model_regresi.pkl'))

prediksi = model.predict(X_test)

if prediksi.ndim > 1:
    prediksi = prediksi.flatten()

hasil = pd.DataFrame({'prediksi_harga': prediksi})
hasil.to_csv(os.path.join(BASE_DIR, 'data', 'predictions', 'hasil_prediksi.csv'), index=False)

print(f"Predictions saved: {len(hasil)} rows")
