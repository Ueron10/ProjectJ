import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'data_konversi.csv'))

X = data[['luas_tanah', 'luas_bangunan', 'kamar_tidur', 'kamar_mandi']]
y = data[['harga']]

X.to_csv(os.path.join(BASE_DIR, 'data', 'processed', 'fitur_rumah.csv'), index=False)
y.to_csv(os.path.join(BASE_DIR, 'data', 'processed', 'target_harga.csv'), index=False)

print(f"Features X: {X.shape}, Target y: {y.shape}")
