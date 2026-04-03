import pandas as pd
import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'data_konversi.csv'))

# Create city-district mapping for later use
city_district_map = {}
for city in sorted(data['kota'].unique()):
    districts = sorted(data[data['kota'] == city]['kecamatan'].unique().tolist())
    city_district_map[city] = districts

# Save mapping to JSON
with open(os.path.join(BASE_DIR, 'models', 'city_district_mapping.json'), 'w', encoding='utf-8') as f:
    json.dump(city_district_map, f, ensure_ascii=False, indent=2)

# One-hot encode kota and kecamatan
data_encoded = pd.get_dummies(data, columns=['kota', 'kecamatan'], prefix=['kota', 'kecamatan'])

# Split features and target
feature_cols = [col for col in data_encoded.columns if col != 'harga']
X = data_encoded[feature_cols]
y = data_encoded[['harga']]

X.to_csv(os.path.join(BASE_DIR, 'data', 'processed', 'fitur_rumah.csv'), index=False)
y.to_csv(os.path.join(BASE_DIR, 'data', 'processed', 'target_harga.csv'), index=False)

# Save feature columns for later use
with open(os.path.join(BASE_DIR, 'models', 'feature_columns.json'), 'w', encoding='utf-8') as f:
    json.dump(feature_cols, f, ensure_ascii=False, indent=2)

print(f"Features X: {X.shape}, Target y: {y.shape}")
print(f"City-district mapping saved: {len(city_district_map)} cities")
print(f"Total districts: {sum(len(d) for d in city_district_map.values())}")
