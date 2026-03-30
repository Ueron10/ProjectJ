# LAPORAN PENELITIAN
## Prediksi Harga Rumah di Jabodetabek Menggunakan Regresi Linear

---

## BAB 1: PENDAHULUAN

### 1.1 Latar Belakang
Properti merupakan salah satu sektor investasi yang penting di Indonesia. Harga rumah dipengaruhi oleh berbagai faktor seperti lokasi, luas tanah, luas bangunan, jumlah kamar, dan fasilitas lainnya. Penentuan harga rumah yang tepat menjadi tantangan bagi penjual dan pembeli.

Machine Learning (ML) adalah cabang dari Artificial Intelligence (AI) yang memungkinkan komputer untuk belajar dari data tanpa diprogram secara eksplisit. ML menganalisis pola-pola dalam data historis untuk membuat prediksi atau keputusan pada data baru.

Dalam konteks properti, ML dapat digunakan untuk memprediksi harga rumah berdasarkan fitur-fitur seperti luas tanah, luas bangunan, jumlah kamar, dan lokasi. Dengan menganalisis data transaksi rumah yang sudah terjadi, model ML dapat mempelajari hubungan antara fitur-fitur tersebut dengan harga jual.

### 1.2 Rumusan Masalah
Bagaimana membangun model prediksi harga rumah menggunakan Regresi Linear berdasarkan fitur luas tanah, luas bangunan, jumlah kamar tidur, dan jumlah kamar mandi?

### 1.3 Tujuan Penelitian
1. Membangun pipeline Machine Learning untuk prediksi harga rumah
2. Melakukan preprocessing data harga rumah Jabodetabek
3. Melatih model Regresi Linear
4. Mengevaluasi performa model

---

## BAB 2: TINJAUAN PUSTAKA

### 2.1 Machine Learning
Machine Learning adalah metode analisis data yang mengotomatisasi pembangunan model analitik. ML menggunakan algoritma yang iteratif belajar dari data, memungkinkan komputer menemukan insight tanpa diprogram secara eksplisit untuk menemukan pengetahuan dari data.

Jenis-jenis Machine Learning:
- **Supervised Learning**: Model dilatih dengan data yang memiliki label/target
- **Unsupervised Learning**: Model menemukan pola dalam data tanpa label
- **Reinforcement Learning**: Model belajar melalui trial and error dengan reward/penalty

### 2.2 Regresi Linear
Regresi Linear adalah salah satu algoritma ML yang paling sederhana dan sering digunakan untuk prediksi nilai kontinu (seperti harga). Model ini mencoba menemukan garis lurus (linear) yang paling sesuai dengan data training.

**Rumus Regresi Linear:**
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

Untuk prediksi harga rumah:
```
harga = intercept + (w₁ × luas_tanah) + (w₂ × luas_bangunan) + (w₃ × kamar_tidur) + (w₄ × kamar_mandi)
```

Dimana:
- `intercept (β₀)`: Nilai dasar harga ketika semua fitur bernilai nol
- `β₁, β₂, β₃, β₄`: Koefisien/bobot yang menunjukkan pengaruh masing-masing fitur
- `ε`: Error/residual

### 2.3 Metrik Evaluasi
**Mean Absolute Error (MAE)**
```
MAE = (1/n) × Σ|yᵢ - ŷᵢ|
```
Rata-rata selisih absolut antara nilai aktual dan prediksi.

**R-squared (R²)**
```
R² = 1 - (SSres / SStot)
```
Mengukur seberapa baik model menjelaskan variasi dalam data (0-1, semakin tinggi semakin baik).

---

## BAB 3: METODOLOGI

### 3.1 Data
Dataset yang digunakan adalah data harga rumah di area Jabodetabek dari website rumah123.com. Dataset terdiri dari 3,553 baris data dengan kolom:

| Kolom | Deskripsi |
|-------|-----------|
| price_in_rp | Harga rumah dalam Rupiah |
| land_size_m2 | Luas tanah dalam meter persegi |
| building_size_m2 | Luas bangunan dalam meter persegi |
| bedrooms | Jumlah kamar tidur |
| bathrooms | Jumlah kamar mandi |

### 3.2 Alat dan Library
- **Bahasa Pemrograman**: Python 3.x
- **Library Utama**:
  - pandas: Manipulasi dan analisis data
  - scikit-learn: Algoritma Machine Learning
  - joblib: Serialisasi model
  - matplotlib: Visualisasi data

### 3.3 Langkah Penelitian

#### 3.3.1 Data Cleaning
- Memilih kolom relevan: harga, luas_tanah, luas_bangunan, kamar_tidur, kamar_mandi
- Menghapus data kosong (dropna)

#### 3.3.2 Konversi Data
- Konversi kolom harga ke tipe numerik (float64)

#### 3.3.3 Feature Engineering
- X (fitur): luas_tanah, luas_bangunan, kamar_tidur, kamar_mandi
- y (target): harga

#### 3.3.4 Data Splitting
- Train set: 80% (2,811 sampel)
- Test set: 20% (703 sampel)
- Random state: 42

#### 3.3.5 Model Training
- Algoritma: Linear Regression (sklearn.linear_model.LinearRegression)
- Training dengan data train

#### 3.3.6 Evaluasi Model
- Mean Absolute Error (MAE)
- R-squared (R²)

---

## BAB 4: HASIL DAN PEMBAHASAN

### 4.1 Implementasi Pipeline
Pipeline dibangun dalam 8 langkah modular yang dirancang untuk memudahkan reproduksi dan pemeliharaan sistem prediksi harga rumah. Setiap langkah menghasilkan output yang menjadi input bagi langkah berikutnya, menciptakan alur kerja yang terstruktur dan transparan.

| No | Script | Fungsi | Input | Output |
|----|--------|--------|-------|--------|
| 1 | `01_cleaning.py` | Memilih kolom relevan, rename, dan hapus missing values | `jabodetabek_house_price.csv` | `data_bersih.csv` (3,514 baris) |
| 2 | `02_konversi.py` | Konversi tipe data harga ke numerik | `data_bersih.csv` | `data_konversi.csv` |
| 3 | `03_fitur_target.py` | Memisahkan fitur (X) dan target (y) | `data_konversi.csv` | `fitur_rumah.csv`, `target_harga.csv` |
| 4 | `04_split.py` | Membagi data train (80%) dan test (20%) | `fitur_rumah.csv`, `target_harga.csv` | `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` |
| 5 | `05_training.py` | Melatih model Linear Regression | `X_train.csv`, `y_train.csv` | `model_regresi.pkl` |
| 6 | `06_prediksi.py` | Melakukan prediksi pada data test | `X_test.csv`, `model_regresi.pkl` | `hasil_prediksi.csv` |
| 7 | `07_evaluasi.py` | Menghitung metrik MAE dan R² | `y_test.csv`, `hasil_prediksi.csv` | Metrik evaluasi di console |
| 8 | `08_tes_manual.py` | Prediksi interaktif berdasarkan input user | Input manual user | Prediksi harga di console |

Keuntungan pendekatan modular ini adalah kemudahan debugging, kemampuan untuk mengganti algoritma pada langkah tertentu tanpa mengubah seluruh sistem, serta dokumentasi yang jelas untuk setiap tahap proses.

### 4.2 Visualisasi Data
Gambar 4.1 menunjukkan scatter plot yang menggambarkan hubungan antara luas bangunan (sumbu x) dengan harga rumah (sumbu y). Dari visualisasi ini, terlihat beberapa pola penting:

![Gambar 4.1](gambar_4_1_luas_bangunan_harga.png)
*Gambar 4.1: Scatter Plot Luas Bangunan vs Harga Rumah*

**Analisis Visualisasi:**
1. **Tren Positif**: Terdapat korelasi positif yang jelas antara luas bangunan dan harga. Semakin besar luas bangunan, harga cenderung meningkat. Hal ini sesuai dengan ekspektasi logis di pasar properti.

2. **Dispersi Data**: Titik-titik data menyebar cukup lebar, menunjukkan bahwa selain luas bangunan, terdapat faktor lain yang mempengaruhi harga (seperti lokasi, fasilitas, kondisi rumah).

3. **Outlier**: Beberapa titik berada jauh di atas garis tren, kemungkinan rumah mewah dengan fasilitas premium. Sebaliknya, titik di bawah tren mungkin rumah di lokasi kurang strategis.

4. **Rentang Data**: Luas bangunan berkisar 0-500 m² dengan harga terkonsentrasi di bawah 10 miliar rupiah.

### 4.3 Koefisien Model

Model Regresi Linear menghitung hubungan antara fitur dan target dengan rumus:

**Rumus Matematis:**
```
harga = β₀ + β₁X₁ + β₂X₂ + β₃X₃ + β₄X₄ + ε
```

Dimana:
- `β₀` = Intercept (nilai konstan)
- `β₁, β₂, β₃, β₄` = Koefisien masing-masing fitur
- `X₁` = Luas Tanah, `X₂` = Luas Bangunan, `X₃` = Kamar Tidur, `X₄` = Kamar Mandi
- `ε` = Error term

**Formula Perhitungan Koefisien (Ordinary Least Squares):**
```
β = (XᵀX)⁻¹Xᵀy
```

Hasil training model Linear Regression menghasilkan koefisien yang merepresentasikan kontribusi masing-masing fitur terhadap harga rumah:

| Fitur | Koefisien | Interpretasi |
|-------|-----------|--------------|
| Luas Tanah | Rp 5,235,380 per m² | Setiap tambahan 1 m² tanah meningkatkan harga sekitar Rp 5.2 juta |
| Luas Bangunan | Rp 25,567,700 per m² | Setiap tambahan 1 m² bangunan meningkatkan harga sekitar Rp 25.6 juta |
| Kamar Tidur | -Rp 739,623,901 | Koefisien negatif, menunjukkan efek yang perlu dianalisis lebih lanjut |
| Kamar Mandi | Rp 143,515,905 | Setiap tambahan 1 kamar mandi meningkatkan harga sekitar Rp 143.5 juta |
| Intercept | Rp 440,669,031 | Nilai dasar harga ketika semua fitur bernilai nol |

**Pembahasan Koefisien:**

1. **Luas Bangunan Dominan**: Koefisien luas bangunan (Rp 25.6 juta/m²) jauh lebih besar daripada luas tanah (Rp 5.2 juta/m²). Ini menunjukkan bahwa luas bangunan memiliki pengaruh terbesar terhadap harga rumah. Hal ini masuk akal karena biaya konstruksi dan material untuk bangunan biasanya lebih signifikan daripada nilai tanah mentah.

2. **Kamar Tidur Negatif**: Koefisien negatif pada kamar tidur mungkin disebabkan oleh:
   - Multikolinearitas dengan luas bangunan (rumah dengan lebih banyak kamar biasanya lebih besar)
   - Rumah kecil dengan banyak kamar mungkin dianggap tidak efisien
   - Perlu analisis lebih lanjut untuk memvalidasi hasil ini

3. **Kamar Mandi Positif**: Kamar mandi memiliki koefisien positif yang signifikan, menunjukkan fasilitas ini menjadi nilai tambah yang dihargai pembeli.

4. **Intercept**: Nilai intercept Rp 440 juta bisa diartikan sebagai nilai minimum atau "harga lahan dasar" meskipun dalam praktiknya rumah dengan fitur nol tidak realistis.

### 4.4 Evaluasi Model
Model dievaluasi menggunakan data test yang terdiri dari 703 sampel (20% dari total data). Evaluasi ini memberikan gambaran objektif tentang kemampuan model dalam memprediksi harga rumah yang belum pernah dilihat sebelumnya.

**Rumus Metrik Evaluasi:**

**Mean Absolute Error (MAE):**
```
MAE = (1/n) × Σ|yᵢ - ŷᵢ|
```

**R-squared (R²):**
```
R² = 1 - (Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²)
```

Dimana:
- `n` = jumlah sampel
- `yᵢ` = harga aktual
- `ŷᵢ` = harga prediksi
- `ȳ` = rata-rata harga aktual

Hasil evaluasi model:

| Metrik | Nilai | Keterangan |
|--------|-------|------------|
| MAE | Rp 1,661,393,109 | Rata-rata selisih prediksi dengan aktual |
| R² Score | 0.5992 | Proporsi variasi yang dapat dijelaskan model |

**Interpretasi Detail:**

1. **R² Score (0.5992)**: Nilai R² sebesar 0.5992 atau 59.92% menunjukkan bahwa model dapat menjelaskan hampir 60% variasi harga rumah dalam dataset. Dalam konteks prediksi harga properti yang dipengaruhi banyak faktor kompleks (lokasi, kondisi pasar, fasilitas umum), nilai ini tergolong moderat hingga baik. Faktor lokasi yang tidak dimasukkan dalam model kemungkinan besar menjadi penyebab 40% variasi yang belum terjelaskan.

2. **MAE (Rp 1.66 Miliar)**: Mean Absolute Error sebesar Rp 1.66 miliar berarti rata-rata prediksi model meleset sekitar 1.66 miliar rupiah dari harga aktual. Untuk rumah dengan harga rata-rata sekitar Rp 2-5 miliar di Jabodetabek, margin error ini masih dalam batas wajar meskipun idealnya lebih rendah.

3. **Konteks Properti Jabodetabek**: Dengan rentang harga rumah di Jabodetabek yang sangat luas (dari ratusan juta hingga puluhan miliar), MAE 1.66 miliar masih dapat diterima untuk prediksi kasar, namun perlu perbaikan untuk keputusan investasi yang lebih akurat.

### 4.5 Prediksi Manual dan Validasi Model

Untuk memvalidasi kegunaan praktis model, dilakukan prediksi manual menggunakan script `08_tes_manual.py`. Script ini menerima input interaktif dari user dengan fitur multi-input (dapat memasukkan beberapa rumah sekaligus) dan menampilkan hasil prediksi harga berdasarkan model yang telah dilatih.

**Cara Penggunaan Script:**
1. Jalankan `python 08_tes_manual.py`
2. Masukkan data rumah: luas tanah, luas bangunan, kamar tidur, kamar mandi
3. Pilih 'y' untuk menambahkan rumah lain, atau 'n' untuk melihat hasil
4. Script menampilkan prediksi harga untuk semua rumah yang diinput

**Contoh Output:**
```
Prediksi Harga Rumah - Multi Input
----------------------------------------

Masukkan data rumah:
Luas Tanah (m2): 100
Luas Bangunan (m2): 120
Jumlah Kamar Tidur: 3
Jumlah Kamar Mandi: 2
Ingin menambahkan rumah lain? (y/n): y

Masukkan data rumah:
Luas Tanah (m2): 200
Luas Bangunan (m2): 250
Jumlah Kamar Tidur: 4
Jumlah Kamar Mandi: 3
Ingin menambahkan rumah lain? (y/n): n

Hasil Prediksi:
----------------------------------------
Rumah 1:
  Luas Tanah: 100 m2
  Luas Bangunan: 120 m2
  Kamar Tidur: 3
  Kamar Mandi: 2
  Harga Prediksi: Rp 2,100,491,170
----------------------------------------
Rumah 2:
  Luas Tanah: 200 m2
  Luas Bangunan: 250 m2
  Kamar Tidur: 4
  Kamar Mandi: 3
  Harga Prediksi: Rp 5,351,722,206
----------------------------------------
```

Berikut hasil prediksi untuk tiga profil rumah yang diuji:

| Profil Rumah | Input | Prediksi | Analisis |
|--------------|-------|----------|----------|
| Rumah Standar | 100 m² tanah, 120 m² bangunan, 3 KT, 2 KM | Rp 2,100,491,170 | Harga sesuai untuk rumah menengah di pinggiran Jakarta |
| Rumah Besar | 200 m² tanah, 250 m² bangunan, 4 KT, 3 KM | Rp 5,351,722,206 | Harga premium, cocok untuk kawasan elit |
| Rumah Kompak | 60 m² tanah, 80 m² bangunan, 2 KT, 1 KM | Rp 1,464,475,953 | Harga terjangkau untuk rumah starter |

**Validasi Hasil Prediksi:**

1. **Konsistensi dengan Pasar**: Harga prediksi untuk ketiga profil berada dalam rentang yang wajar untuk pasar properti Jabodetabek saat ini. Rumah dengan luas bangunan 120 m² dihargai sekitar Rp 2.1 miliar, yang masuk akal untuk lokasi suburban.

2. **Pengaruh Luas Bangunan**: Perbandingan Rumah Standar (120 m² bangunan) dan Rumah Besar (250 m² bangunan) menunjukkan selisih harga Rp 3.25 miliar. Sesuai koefisien luas bangunan Rp 25.6 juta/m², perbedaan 130 m² seharusnya menambah Rp 3.32 miliar, yang konsisten dengan hasil prediksi.

3. **Limitasi Model**: Model tidak memperhitungkan lokasi spesifik (Jakarta pusat vs Bekasi), sehingga prediksi mungkin kurang akurat untuk rumah di lokasi premium atau terpencil. Ini menjelaskan mengapa metrik evaluasi menunjukkan margin error yang cukup besar.

---

## BAB 5: KESIMPULAN

### 5.1 Kesimpulan
1. Model Regresi Linear berhasil dibangun untuk prediksi harga rumah di Jabodetabek
2. Model mencapai R² = 0.5992 yang menunjukkan performa moderat
3. Luas bangunan memiliki pengaruh terbesar terhadap harga (koefisien Rp 25.5 juta/m²)
4. Pipeline modular memudahkan reproduksi dan pengembangan lebih lanjut

### 5.2 Saran
1. Menambahkan fitur lokasi (kota/kecamatan) untuk meningkatkan akurasi
2. Mencoba algoritma lain seperti Random Forest atau XGBoost
3. Melakukan feature scaling untuk data dengan rentang nilai berbeda
4. Mengumpulkan lebih banyak data untuk meningkatkan generalisasi model

---

## DAFTAR PUSTAKA

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
2. Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. O'Reilly Media.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
4. McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 51-56.

---

## LAMPIRAN

### Struktur Folder Project
```
ML/
├── data/
│   ├── raw/              # Dataset asli
│   ├── processed/        # Data hasil preprocessing
│   ├── train/            # Data training
│   ├── test/             # Data testing
│   └── predictions/      # Hasil prediksi
├── models/               # Model yang sudah di-train
│   └── model_regresi.pkl
├── src/                  # Source code pipeline
│   ├── 01_cleaning.py
│   ├── 02_konversi.py
│   ├── 03_fitur_target.py
│   ├── 04_split.py
│   ├── 05_training.py
│   ├── 06_prediksi.py
│   ├── 07_evaluasi.py
│   └── 08_tes_manual.py
└── documentation/        # Dokumentasi dan gambar
    ├── README.md
    └── gambar_4_1_luas_bangunan_harga.png
```

### Cara Menjalankan
```bash
# Run dari folder src
cd src
python 01_cleaning.py
python 02_konversi.py
python 03_fitur_target.py
python 04_split.py
python 05_training.py
python 06_prediksi.py
python 07_evaluasi.py
python 08_tes_manual.py
```

### Dependensi
- Python 3.x
- pandas
- scikit-learn
- joblib
- matplotlib

Install dengan:
```bash
pip install pandas scikit-learn joblib matplotlib
```
