import numpy as np
import pickle
import re
import os
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Setup path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
models_dir = os.path.join(project_root, "Models")
outputs_dir = os.path.join(project_root, "Outputs")

# Load artifacts
model = load_model(os.path.join(models_dir, "model_final.keras"))
with open(os.path.join(outputs_dir, "tokenizer.pkl"), "rb") as f:
    tokenizer = pickle.load(f)
label_classes = np.load(os.path.join(outputs_dir, "label_encoding.npy"), allow_pickle=True)

# Preprocessing setup
stemmer = PorterStemmer()
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

def preprocess(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def predict(text):
    cleaned = preprocess(text)
    if not cleaned:
        return None
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
    probs = model.predict(padded, verbose=0)[0]
    labels = [str(label).lower() for label in label_classes]
    pred_idx = int(np.argmax(probs))
    return {
        'label': labels[pred_idx],
        'confidence': float(probs[pred_idx]),
        'all_probs': {labels[i]: float(probs[i]) for i in range(len(labels))},
        'cleaned': cleaned
    }

# Interactive mode
print("Sentiment Analysis Prediction")
print("="*50)
print("Enter text to analyze (or 'quit' to exit)\n")

while True:
    text = input("> ").strip()
    if text.lower() in ['quit', 'exit', 'q']:
        break
    if not text:
        continue
    
    result = predict(text)
    if result is None:
        print("\n  Result: unable to predict (cleaned text is empty)\n")
        continue
    print(f"\n  Result: {result['label'].upper()}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print()

print("Goodbye!")
