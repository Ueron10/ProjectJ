import joblib
import os
import pandas as pd

# Load model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, 'models', 'model_regresi.pkl'))

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

print("Prediksi Harga Rumah - Multi Input")
print("-" * 40)

rumah_list = []

while True:
    print("\nMasukkan data rumah:")
    luas_tanah = input_float("Luas Tanah (m2): ")
    luas_bangunan = input_float("Luas Bangunan (m2): ")
    kamar_tidur = input_int("Jumlah Kamar Tidur: ")
    kamar_mandi = input_int("Jumlah Kamar Mandi: ")

    rumah_list.append({
        'luas_tanah': luas_tanah,
        'luas_bangunan': luas_bangunan,
        'kamar_tidur': kamar_tidur,
        'kamar_mandi': kamar_mandi
    })

    lanjut = input("Ingin menambahkan rumah lain? (y/n): ").strip().lower()
    if lanjut != 'y':
        break

# Konversi list ke DataFrame
df_input = pd.DataFrame(rumah_list)

# Prediksi
prediksi = model.predict(df_input)
prediksi = prediksi.flatten()  # pastikan 1D array

print("\nHasil Prediksi:")
print("-" * 40)
for i, row in enumerate(df_input.itertuples(), start=1):
    print(f"Rumah {i}:")
    print(f"  Luas Tanah: {row.luas_tanah} m2")
    print(f"  Luas Bangunan: {row.luas_bangunan} m2")
    print(f"  Kamar Tidur: {row.kamar_tidur}")
    print(f"  Kamar Mandi: {row.kamar_mandi}")
    print(f"  Harga Prediksi: Rp {prediksi[i-1]:,.0f}")
    print("-" * 40)