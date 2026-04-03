import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data = pd.read_csv(os.path.join(BASE_DIR, 'data', 'raw', 'jabodetabek_house_price.csv'))

# Clean whitespace from city and district
data['city'] = data['city'].str.strip()
data['district'] = data['district'].str.strip()

data = data[['price_in_rp', 'city', 'district', 'land_size_m2', 'building_size_m2', 'bedrooms', 'bathrooms']]
data.columns = ['harga', 'kota', 'kecamatan', 'luas_tanah', 'luas_bangunan', 'kamar_tidur', 'kamar_mandi']
data = data.dropna()

data.to_csv(os.path.join(BASE_DIR, 'data', 'processed', 'data_bersih.csv'), index=False)
print(f"Data cleaned: {len(data)} rows saved to data_bersih.csv")

