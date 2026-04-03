import joblib
import json
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model
model = joblib.load(os.path.join(BASE_DIR, 'models', 'model_regresi.pkl'))

# Load city-district mapping
with open(os.path.join(BASE_DIR, 'models', 'city_district_mapping.json'), 'r', encoding='utf-8') as f:
    city_district_map = json.load(f)

# Load feature columns
with open(os.path.join(BASE_DIR, 'models', 'feature_columns.json'), 'r', encoding='utf-8') as f:
    feature_columns = json.load(f)

def input_float(prompt):
    while True:
        try:
            value = float(input(prompt))
            if value <= 0:
                print("Nilai harus lebih dari 0.")
                continue
            return value
        except ValueError:
            print("Input tidak valid. Masukkan angka.")

def input_int(prompt):
    while True:
        try:
            value = int(input(prompt))
            if value < 0:
                print("Nilai tidak boleh negatif.")
                continue
            return value
        except ValueError:
            print("Input tidak valid. Masukkan angka bulat.")

def pilih_kota():
    print("\n=== PILIH KOTA ===")
    cities = sorted(city_district_map.keys())
    for i, city in enumerate(cities, 1):
        print(f"{i}. {city}")
    
    while True:
        try:
            pilihan = int(input(f"\nPilih kota (1-{len(cities)}): "))
            if 1 <= pilihan <= len(cities):
                return cities[pilihan - 1]
            print(f"Pilihan harus antara 1-{len(cities)}")
        except ValueError:
            print("Input tidak valid. Masukkan angka.")

def pilih_kecamatan(kota):
    print(f"\n=== PILIH KECAMATAN DI {kota.upper()} ===")
    districts = city_district_map[kota]
    for i, district in enumerate(districts, 1):
        print(f"{i}. {district}")
    
    while True:
        try:
            pilihan = int(input(f"\nPilih kecamatan (1-{len(districts)}): "))
            if 1 <= pilihan <= len(districts):
                return districts[pilihan - 1]
            print(f"Pilihan harus antara 1-{len(districts)}")
        except ValueError:
            print("Input tidak valid. Masukkan angka.")

def buat_input_data(kota, kecamatan, luas_tanah, luas_bangunan, kamar_tidur, kamar_mandi):
    # Create a dictionary with all features set to 0
    input_data = {col: 0 for col in feature_columns}
    
    # Set numeric features
    input_data['luas_tanah'] = luas_tanah
    input_data['luas_bangunan'] = luas_bangunan
    input_data['kamar_tidur'] = kamar_tidur
    input_data['kamar_mandi'] = kamar_mandi
    
    # Set one-hot encoded city and district
    kota_col = f"kota_{kota}"
    kecamatan_col = f"kecamatan_{kecamatan}"
    
    if kota_col in input_data:
        input_data[kota_col] = 1
    if kecamatan_col in input_data:
        input_data[kecamatan_col] = 1
    
    return pd.DataFrame([input_data])

print("=" * 50)
print("PREDIKSI HARGA RUMAH JABODETABEK")
print("=" * 50)

rumah_list = []

while True:
    print(f"\n--- Data Rumah {len(rumah_list) + 1} ---")
    
    # Pilih kota dan kecamatan
    kota = pilih_kota()
    kecamatan = pilih_kecamatan(kota)
    
    # Input data rumah
    print(f"\n--- Spesifikasi Rumah di {kecamatan}, {kota} ---")
    luas_tanah = input_float("Luas Tanah (m2): ")
    luas_bangunan = input_float("Luas Bangunan (m2): ")
    kamar_tidur = input_int("Jumlah Kamar Tidur: ")
    kamar_mandi = input_int("Jumlah Kamar Mandi: ")
    
    rumah_list.append({
        'kota': kota,
        'kecamatan': kecamatan,
        'luas_tanah': luas_tanah,
        'luas_bangunan': luas_bangunan,
        'kamar_tidur': kamar_tidur,
        'kamar_mandi': kamar_mandi
    })
    
    lanjut = input("\nIngin menambahkan rumah lain? (y/n): ").strip().lower()
    if lanjut != 'y':
        break

# Prediksi untuk semua rumah
print("\n" + "=" * 50)
print("HASIL PREDIKSI")
print("=" * 50)

for i, rumah in enumerate(rumah_list, start=1):
    # Buat input data
    input_df = buat_input_data(
        rumah['kota'], 
        rumah['kecamatan'], 
        rumah['luas_tanah'], 
        rumah['luas_bangunan'], 
        rumah['kamar_tidur'], 
        rumah['kamar_mandi']
    )
    
    # Prediksi
    prediksi = model.predict(input_df)
    prediksi_value = prediksi[0] if len(prediksi) == 1 else prediksi.flatten()[0]
    
    print(f"\nRumah {i}:")
    print(f"  Lokasi: {rumah['kecamatan']}, {rumah['kota']}")
    print(f"  Luas Tanah: {rumah['luas_tanah']} m2")
    print(f"  Luas Bangunan: {rumah['luas_bangunan']} m2")
    print(f"  Kamar Tidur: {rumah['kamar_tidur']}")
    print(f"  Kamar Mandi: {rumah['kamar_mandi']}")
    print(f"  Harga Prediksi: Rp {prediksi_value:,.0f}")
    print("-" * 50)