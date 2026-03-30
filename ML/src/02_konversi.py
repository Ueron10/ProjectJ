import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'data_bersih.csv'))

if data['harga'].dtype == 'object':
    data['harga'] = data['harga'].replace('[^0-9]', '', regex=True).astype(float)
else:
    data['harga'] = data['harga'].astype(float)

data.to_csv(os.path.join(BASE_DIR, 'data', 'processed', 'data_konversi.csv'), index=False)
print(f"Data converted: {len(data)} rows saved to data_konversi.csv")
