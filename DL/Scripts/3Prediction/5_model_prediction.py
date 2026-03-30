import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Setup path relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
models_dir = os.path.join(project_root, "Models")
outputs_dir = os.path.join(project_root, "Outputs")

# Initialize stemmer and custom stopwords
stemmer = PorterStemmer()
std_stopwords = set(stopwords.words('english'))
sentiment_important = {
    'no', 'not', 'nor', 'don', 'don\'t', 'doesn', 'doesn\'t', 'didn', 'didn\'t', 'didn\'t',
    'hasn', 'hasn\'t', 'haven', 'haven\'t', 'isn', 'isn\'t', 'aren', 'aren\'t', 'wasn', 'wasn\'t', 
    'weren', 'weren\'t', 'be', 'been', 'being', 'have', 'has', 'had', 'does', 'did', 'will', 'would',
    'could', 'ought', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'whom',
    'very', 'so', 'too', 'just', 'more', 'most', 'such', 'only', 'own', 'same',
    'and', 'or', 'if', 'then', 'because', 'as', 'is', 'are'
}
stop_words = std_stopwords - sentiment_important

# Lexicon for short text
POSITIVE_WORDS = {
    'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome',
    'love', 'best', 'perfect', 'beautiful', 'brilliant', 'nice', 'lovely',
    'splendid', 'marvelous', 'glorious', 'phenomenal', 'fabulous', 'exceptional',
    'wow', 'yay', 'hurray', 'terrific', 'superb', 'outstanding', 'wonderful',
    'like', 'enjoy', 'entertaining', 'delightful', 'impressive', 'wonderful'
}

NEGATIVE_WORDS = {
    'bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'poor', 'weak',
    'stupid', 'dumb', 'boring', 'annoying', 'lame', 'pathetic', 'useless',
    'waste', 'disappointing', 'disgusting', 'atrocious', 'vile', 'abominable',
    'despicable', 'ugly', 'gross', 'nasty', 'crappy', 'sucks', 'unwatchable'
}

# Helper Functions
def preprocess_text(text):
    """Preprocess teks sesuai pipeline training"""
    if not text:
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

# Lexicon-based sentiment (for short text)
def lexicon_sentiment(text):
    """
    Deteksi sentiment menggunakan lexicon matching untuk teks pendek
    Lebih akurat untuk teks < 5 tokens
    
    Returns:
        dict: {'label': 'positive'|'negative'|'neutral', 'confidence': float, 'method': 'lexicon'}
              or None jika lexicon tidak bisa menentukan
    """
    cleaned = preprocess_text(text)
    tokens = cleaned.split()
    
    if not tokens or len(tokens) == 0:
        return None
    
    positive_count = 0
    negative_count = 0
    
    for token in tokens:
        # Match exact atau stemmed version
        if token in POSITIVE_WORDS:
            positive_count += 1
        elif token in NEGATIVE_WORDS:
            negative_count += 1
        else:
            # Check if stemmed version matches
            stem = stemmer.stem(token)
            for pos_word in POSITIVE_WORDS:
                if stem.startswith(pos_word[:3]) or pos_word.startswith(stem[:3]):
                    positive_count += 1
                    break
            for neg_word in NEGATIVE_WORDS:
                if stem.startswith(neg_word[:3]) or neg_word.startswith(stem[:3]):
                    negative_count += 1
                    break
    
    total_matches = positive_count + negative_count
    
    # Jika ditemukan matches
    if total_matches > 0:
        if positive_count > negative_count:
            confidence = min(0.95, 0.7 + (positive_count / len(tokens)) * 0.25)
            return {
                'label': 'positive',
                'confidence': confidence,
                'method': 'lexicon'
            }
        elif negative_count > positive_count:
            confidence = min(0.95, 0.7 + (negative_count / len(tokens)) * 0.25)
            return {
                'label': 'negative',
                'confidence': confidence,
                'method': 'lexicon'
            }
    
    # Jika tidak ada matches yang jelas
    return None

# Load Model
print("\n" + "="*70)
print("LOADING MODEL AND ARTIFACTS...")
print("="*70)

try:
    model = load_model(os.path.join(models_dir, "model_sentiment_cnn.keras"))
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model file not found!")
    print("  Jalankan 4_model_save.py terlebih dahulu")
    exit(1)

# Load tokenizer
try:
    with open(os.path.join(outputs_dir, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    print("Tokenizer loaded successfully!")
except FileNotFoundError:
    print("Tokenizer file not found!")
    exit(1)

# Load label classes
try:
    label_classes = np.load(os.path.join(outputs_dir, "label_encoding.npy"), allow_pickle=True)
    print("Label encoding loaded successfully!")
    print(f"  Classes: {label_classes}")
except FileNotFoundError:
    print("Label encoding file not found!")
    print(f"  Looking for: {os.path.join(outputs_dir, 'label_encoding.npy')}")
    exit(1)

# Prediction Function (Hybrid)
def predict_sentiment(text, top_n=3):
    """
    Prediksi sentiment dari teks input menggunakan HYBRID APPROACH:
    - Teks < 5 tokens: Lexicon-based (akurat untuk teks pendek)
    - Teks >= 5 tokens: Neural network CNN (akurat untuk teks panjang)
    
    Args:
        text (str): Input teks untuk diprediksi
        top_n (int): Jumlah top predictions yang ditampilkan
    
    Returns:
        dict: Hasil prediksi dengan probabilities dan label
    """
    
    # Preprocess teks
    cleaned_text = preprocess_text(text)
    
    if not cleaned_text:
        print("Teks kosong setelah preprocessing!")
        return None
    
    # Hitung jumlah tokens
    token_count = len(cleaned_text.split())
    
    # Untuk teks pendek (< 5 tokens): gunakan lexicon
    if token_count < 5:
        lexicon_result = lexicon_sentiment(text)
        
        if lexicon_result:
            # Lexicon berhasil mendeteksi sentiment
            predicted_label = lexicon_result['label']
            confidence = lexicon_result['confidence']
            
            # Buat probabilities dengan preference terhadap lexicon result
            if predicted_label == 'positive':
                probs = {
                    'positive': confidence,
                    'negative': (1 - confidence) * 0.5,
                    'neutral': (1 - confidence) * 0.5
                }
            elif predicted_label == 'negative':
                probs = {
                    'negative': confidence,
                    'positive': (1 - confidence) * 0.5,
                    'neutral': (1 - confidence) * 0.5
                }
            else:
                probs = {
                    'neutral': confidence,
                    'positive': (1 - confidence) * 0.5,
                    'negative': (1 - confidence) * 0.5
                }
            
            # Normalize probabilities
            total = sum(probs.values())
            probs = {k: v/total for k, v in probs.items()}
            
            # Urutkan untuk top predictions
            sorted_labels = sorted(probs.items(), key=lambda x: x[1], reverse=True)
            
            result = {
                'text': text,
                'cleaned_text': cleaned_text,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'probabilities': probs,
                'method': 'lexicon',
                'token_count': token_count,
                'top_predictions': [
                    {
                        'label': label,
                        'probability': prob
                    }
                    for label, prob in sorted_labels[:top_n]
                ]
            }
            
            return result
    
    # Untuk teks panjang atau lexicon gagal: gunakan neural network
    # Tokenize
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    
    # Pad sequences
    padded = pad_sequences(sequences, maxlen=100)
    
    # Prediksi
    prediction_probs = model.predict(padded, verbose=0)[0]
    predicted_idx = np.argmax(prediction_probs)
    predicted_label = label_classes[predicted_idx]
    
    # Urutkan probabilities dari terbesar ke terkecil
    sorted_indices = np.argsort(prediction_probs)[::-1]
    
    # Hasil
    result = {
        'text': text,
        'cleaned_text': cleaned_text,
        'predicted_label': predicted_label,
        'confidence': float(prediction_probs[predicted_idx]),
        'probabilities': {
            label_classes[i]: float(prediction_probs[i]) 
            for i in range(len(label_classes))
        },
        'method': 'neural' if token_count >= 5 else 'neural_fallback',
        'token_count': token_count,
        'top_predictions': [
            {
                'label': label_classes[idx],
                'probability': float(prediction_probs[idx])
            }
            for idx in sorted_indices[:top_n]
        ]
    }
    
    return result

# Interactive Prediction
print("\n" + "="*70)
print("SENTIMENT PREDICTION")
print("="*70)
print("\nMasukkan teks untuk diprediksi (atau 'quit' untuk keluar)\n")

while True:
    # Input teks
    user_input = input("Teks: ").strip()
    
    # Check if user wants to quit
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\nProgram selesai!")
        break
    
    if not user_input:
        print("Masukkan teks terlebih dahulu!\n")
        continue
    
    # Prediksi
    print("\nProcessing...")
    result = predict_sentiment(user_input)
    
    if result:
        print("\n" + "-"*70)
        print(f"Input: {result['text']}")
        print(f"Cleaned: {result['cleaned_text']}")
        print(f"Tokens: {result['token_count']} | Method: {result['method'].upper()}")
        print("-"*70)
        print(f"\nHASIL PREDIKSI:")
        print(f"   Label: {result['predicted_label'].upper()}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"\nSEMUA PROBABILITAS:")
        for label, prob in result['probabilities'].items():
            bar = "█" * int(prob * 30)
            print(f"   {label:10} [{bar:<30}] {prob:.4f}")
        print("\nTOP PREDICTIONS:")
        for i, pred in enumerate(result['top_predictions'], 1):
            print(f"   {i}. {pred['label']:10} - {pred['probability']:.4f} ({pred['probability']:.2%})")
    
    print()

# Batch Prediction (Optional)
print("\n" + "="*70)
print("BATCH PREDICTION EXAMPLE")
print("="*70)

# Contoh prediksi multiple texts
test_texts = [
    "This movie is absolutely amazing and wonderful!",
    "The film was okay, nothing special about it.",
    "Worst movie ever, completely terrible and boring!"
]

print("\nContoh prediksi multiple texts:\n")

for text in test_texts:
    result = predict_sentiment(text)
    if result:
        print(f"[{result['predicted_label'].upper()}] {result['confidence']:.2%} - {text}")

print("\n" + "="*70)
