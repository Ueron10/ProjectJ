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

# Download resource NLTK
nltk.download('punkt')
nltk.download('punkt_tab') 
nltk.download('stopwords')

# Load data
df = pd.read_csv(os.path.join(data_dir, "imdb_reviews_full.csv"))

# Inisialisasi
# Custom stopwords - KEEP sentiment-important words
std_stopwords = set(stopwords.words('english'))
# Remove sentiment-important words from stopwords
sentiment_important = {
    'no', 'not', 'nor', 'don', 'don\'t', 'doesn', 'doesn\'t', 'didn', 'didn\'t', 'didn\'t',
    'hasn', 'hasn\'t', 'haven', 'haven\'t', 'isn', 'isn\'t', 'aren', 'aren\'t', 'wasn', 'wasn\'t', 
    'weren', 'weren\'t', 'be', 'been', 'being', 'have', 'has', 'had', 'does', 'did', 'will', 'would',
    'could', 'ought', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'whom',
    'very', 'so', 'too', 'just', 'more', 'most', 'such', 'only', 'own', 'same',
    'and', 'or', 'if', 'then', 'because', 'as', 'is', 'are'
}
stop_words = std_stopwords - sentiment_important
stemmer = PorterStemmer()

# Fungsi preprocessing
def preprocess_text(text):
    if pd.isna(text):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Hapus HTML / tag
    text = re.sub(r'<.*?>', '', text)

    # 3. Hapus karakter selain huruf
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 4. Tokenizing
    tokens = word_tokenize(text)

    # 5. Stopword removal
    tokens = [word for word in tokens if word not in stop_words]

    # 6. Stemming
    tokens = [stemmer.stem(word) for word in tokens]

    # Gabungkan kembali
    return " ".join(tokens)

# Terapkan ke kolom content
df['clean_text'] = df['content'].apply(preprocess_text)

# Simpan hasil
df.to_csv(os.path.join(data_dir, "imdb_reviews_clean.csv"), index=False, encoding='utf-8-sig')

# Preview
df[['content', 'clean_text']].head()