import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pickle

# Setup path relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
data_dir = os.path.join(project_root, "Data")
outputs_dir = os.path.join(project_root, "Outputs")
os.makedirs(outputs_dir, exist_ok=True)

# Load data
df = pd.read_csv(os.path.join(data_dir, "imdb_reviews_labeled.csv"))
texts = df['clean_text'].fillna("").astype(str)

print(f"Tokenizing {len(texts)} texts...")

# Tokenization
vocab_size = 5000
max_length = 100

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Padding
X = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

print(f"\nTokenization selesai!")
print(f"  Vocabulary size: {vocab_size}")
print(f"  Max sequence length: {max_length}")
print(f"  Data shape: {X.shape}")

# Simpan
np.save(os.path.join(outputs_dir, "X_tokenized.npy"), X)
with open(os.path.join(outputs_dir, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

print("\nOutput files:")
print("  - X_tokenized.npy")
print("  - tokenizer.pkl")
