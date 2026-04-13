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
stop_words = set(stopwords.words('english')) - {
    'no', 'not', 'nor', 'don', 'don\'t', 'very', 'so', 'too', 'just', 'only'
}

def preprocess(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

def predict(text):
    cleaned = preprocess(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=100)
    probs = model.predict(padded, verbose=0)[0]
    pred_idx = np.argmax(probs)
    return {
        'label': label_classes[pred_idx],
        'confidence': probs[pred_idx],
        'all_probs': {label_classes[i]: probs[i] for i in range(len(label_classes))}
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
    print(f"\n  Result: {result['label'].upper()}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print()

print("Goodbye!")
