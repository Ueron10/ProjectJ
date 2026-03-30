import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

y_test = pd.read_csv(os.path.join(BASE_DIR, 'data', 'test', 'y_test.csv'))['harga'].values
prediksi = pd.read_csv(os.path.join(BASE_DIR, 'data', 'predictions', 'hasil_prediksi.csv'))['prediksi_harga'].values

mae = mean_absolute_error(y_test, prediksi)
r2 = r2_score(y_test, prediksi)

print(f"MAE: Rp {mae:,.0f}")
print(f"R2 Score: {r2:.4f}")
