import pandas as pd
import os

# Setup path relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
data_dir = os.path.join(project_root, "Data")

# Load data
df = pd.read_csv(os.path.join(data_dir, "imdb_reviews_clean.csv"))

# Pastikan rating numeric
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Fungsi labeling
def label_sentiment(rating):
    if rating >= 7:
        return "positive"
    elif rating >= 5:
        return "neutral"
    else:
        return "negative"

# Terapkan
df['sentiment'] = df['rating'].apply(label_sentiment)

# Simpan hasil
df.to_csv(os.path.join(data_dir, "imdb_reviews_labeled.csv"), index=False, encoding='utf-8-sig')

# Preview
df.head()