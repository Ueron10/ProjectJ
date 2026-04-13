import pandas as pd
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon', quiet=True)

# Setup path relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
data_dir = os.path.join(project_root, "Data")

# Load data
df = pd.read_csv(os.path.join(data_dir, "imdb_reviews_clean.csv"))

# Inisialisasi VADER
sia = SentimentIntensityAnalyzer()

# Fungsi labeling menggunakan VADER
def label_sentiment_vader(text):
    if pd.isna(text) or text == "":
        return "neutral"
    
    scores = sia.polarity_scores(str(text))
    compound = scores['compound']
    
    # Threshold berdasarkan compound score VADER
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"

# Terapkan VADER ke clean_text (bukan rating)
df['sentiment'] = df['clean_text'].apply(label_sentiment_vader)

# Simpan hasil
df.to_csv(os.path.join(data_dir, "imdb_reviews_labeled.csv"), index=False, encoding='utf-8-sig')

# Preview distribusi label
print("Distribusi Label (VADER):")
print(df['sentiment'].value_counts())
print(f"\nTotal data: {len(df)}")

# Preview
df[['content', 'clean_text', 'sentiment']].head()