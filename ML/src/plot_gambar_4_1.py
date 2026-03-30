import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'data_konversi.csv'))

plt.figure(figsize=(10, 6))
plt.scatter(data['luas_bangunan'], data['harga'], alpha=0.5, s=20)
plt.xlabel('Luas Bangunan (m²)')
plt.ylabel('Harga (Rp)')
plt.title('Hubungan Luas Bangunan dan Harga Rumah')

# Format y-axis to show in billions
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'Rp {x/1e9:.1f}B'))

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(os.path.join(BASE_DIR, 'documentation', 'gambar_4_1_luas_bangunan_harga.png'), dpi=150)
print("gambar_4_1_luas_bangunan_harga.png")
