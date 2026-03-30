import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Setup path relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
data_dir = os.path.join(project_root, "Data")
outputs_dir = os.path.join(project_root, "Outputs")

# 1. LOAD DATA
df = pd.read_csv(os.path.join(data_dir, "imdb_reviews_labeled.csv"))

# Ambil clean_text (handle NaN)
texts = df['clean_text'].fillna("").astype(str)

print("Jumlah data:", len(texts))

# Tokenization
vocab_size = 5000  # jumlah kata yang dipakai

tokenizer = Tokenizer(
    num_words=vocab_size,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(texts)

# Ubah ke sequence angka
sequences = tokenizer.texts_to_sequences(texts)

# 4. Padding
max_length = 100

X = pad_sequences(
    sequences,
    maxlen=max_length,
    padding='post',     # tambahkan 0 di belakang
    truncating='post'   # potong di belakang
)

# 5. Cek hasil
print("\nContoh teks asli:")
print(texts.iloc[0])

print("\nSequence:")
print(sequences[0])

print("\nPadded:")
print(X[0])

print("\nShape data:", X.shape)

# 6. Simpan (opsional tapi disarankan)
np.save(os.path.join(outputs_dir, "X_tokenized.npy"), X)

# Simpan tokenizer (biar bisa dipakai lagi)
import pickle
with open(os.path.join(outputs_dir, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

print("\nTokenization & Padding selesai!")
print("File tersimpan: X_tokenized.npy & tokenizer.pkl")