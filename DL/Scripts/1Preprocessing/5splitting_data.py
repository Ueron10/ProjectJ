import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# Setup path relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
data_dir = os.path.join(project_root, "Data")
outputs_dir = os.path.join(project_root, "Outputs")

# 1. Load CSV (label)
df = pd.read_csv(os.path.join(data_dir, "imdb_reviews_labeled.csv"))  # pastikan file ada

# 2. Load X (hasil token + padding)
X = np.load(os.path.join(outputs_dir, "X_tokenized.npy"))

# 3. Ambil label dan convert ke numeric
y_text = df['sentiment'].values  # pastikan kolom 'sentiment' ada di CSV
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)  # Convert text to numeric labels

# 4. Tentukan rasio train/test
train_ratio = 0.8
total_samples = X.shape[0]
train_size = int(total_samples * train_ratio)

# 5. Shuffle
indices = np.arange(total_samples)
np.random.seed(42)
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

# 6. Split
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

# 7. Cek shape
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# 8. (Opsional) Simpan
np.save(os.path.join(outputs_dir, "X_train.npy"), X_train)
np.save(os.path.join(outputs_dir, "X_test.npy"), X_test)
np.save(os.path.join(outputs_dir, "y_train.npy"), y_train)
np.save(os.path.join(outputs_dir, "y_test.npy"), y_test)

# 9. Simpan label encoder mapping
print("\nLabel Encoding Mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"  {i} = {label}")
np.save(os.path.join(outputs_dir, "label_encoding.npy"), label_encoder.classes_)