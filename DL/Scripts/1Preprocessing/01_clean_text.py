import pandas as pd
import re
import nltk
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Setup path relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
data_dir = os.path.join(project_root, "Data")

# Download resource NLTK (hanya jika belum ada)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# Load data
df = pd.read_csv(os.path.join(data_dir, "IMDB Dataset.csv"))

# Inisialisasi
std_stopwords = set(stopwords.words('english'))
sentiment_important = {
    'no', 'not', 'nor', 'don', 'don\'t', 'doesn', 'doesn\'t', 'didn', 'didn\'t',
    'hasn', 'hasn\'t', 'haven', 'haven\'t', 'isn', 'isn\'t', 'aren', 'aren\'t', 
    'wasn', 'wasn\'t', 'weren', 'weren\'t', 'be', 'been', 'being', 'have', 'has', 
    'had', 'does', 'did', 'will', 'would', 'could', 'ought', 'i', 'you', 'he', 
    'she', 'it', 'we', 'they', 'what', 'which', 'who', 'whom', 'very', 'so', 'too', 
    'just', 'more', 'most', 'such', 'only', 'own', 'same', 'and', 'or', 'if', 'then', 
    'because', 'as', 'is', 'are'
}
stop_words = std_stopwords - sentiment_important
stemmer = PorterStemmer()

def preprocess_text(text):
    if pd.isna(text):
        return ""
    
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Hapus HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hapus karakter non-huruf
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    
    return " ".join(tokens)

# Terapkan preprocessing
print("Preprocessing text data...")
df['clean_text'] = df['review'].apply(preprocess_text)

# Simpan hasil
df.to_csv(os.path.join(data_dir, "imdb_reviews_clean.csv"), index=False, encoding='utf-8-sig')

print("Preprocessing selesai!")
print(f"  Total samples: {len(df)}")
print(f"  Output: imdb_reviews_clean.csv")
print("\nSample:")
print(df[['review', 'clean_text', 'sentiment']].head())
