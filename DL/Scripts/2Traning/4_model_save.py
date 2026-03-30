from tensorflow.keras.models import load_model
import os

# Setup path relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
models_dir = os.path.join(project_root, "Models")

# Create Models directory if doesn't exist
os.makedirs(models_dir, exist_ok=True)

print("\n" + "="*70)
print("SAVING MODEL AND ARTIFACTS")
print("="*70)

# Load Model
print("\nLoading trained model...")
try:
    model = load_model(os.path.join(models_dir, "model_sentiment_cnn_trained.keras"))
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model file not found!")
    print("  Jalankan 2_model_training.py terlebih dahulu")
    exit(1)

# Save Model
print("\nSaving model...")
output_model_path = os.path.join(models_dir, "model_sentiment_cnn.keras")

model.save(output_model_path)
print(f"Model saved to: {output_model_path}")

# Create README
print("\nCreating README...")
readme_content = """# Sentiment Analysis Model

## Model Information
- **Model Type**: CNN (Convolutional Neural Network)
- **Framework**: TensorFlow/Keras
- **Input**: Tokenized text (max_length=100)
- **Output**: 3 classes (negative, neutral, positive)

## Files
- `model_sentiment_cnn.keras`: Trained model
- `tokenizer.pkl`: Text tokenizer (pickle format)
- `label_encoding.npy`: Label encoding mapping

## Usage

### Load Model and Make Predictions
```python
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model('model_sentiment_cnn.keras')

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load label classes
label_classes = np.load('label_encoding.npy')

# Prepare text
text = "This movie is amazing!"
sequences = tokenizer.texts_to_sequences([text])
padded = pad_sequences(sequences, maxlen=100)

# Predict
prediction = model.predict(padded)
predicted_class = np.argmax(prediction)
sentiment = label_classes[predicted_class]

print(f"Sentiment: {sentiment}")
print(f"Confidence: {prediction[0][predicted_class]:.2f}")
```

## Model Architecture
- Embedding Layer (vocab_size=5000, embedding_dim=100)
- Conv1D Layer 1 (128 filters, kernel_size=5)
- Conv1D Layer 2 (64 filters, kernel_size=3)
- GlobalMaxPooling1D
- Dense Layer (128 units, ReLU)
- Dropout (0.5)
- Dense Layer (64 units, ReLU)
- Dropout (0.3)
- Output Layer (3 units, Softmax)

## Performance
- Test Accuracy: ~69.42%
- Classes: Negative (0), Neutral (1), Positive (2)
"""

readme_path = os.path.join(models_dir, "README.md")
with open(readme_path, "w") as f:
    f.write(readme_content)
print("README.md created")

# Summary
print("\n" + "="*70)
print("SAVE SUMMARY")
print("="*70)
print(f"\nFiles saved in '{models_dir}' directory:")
print("  - model_sentiment_cnn.keras")
print("  - tokenizer.pkl - stored in Outputs/")
print("  - label_encoding.npy - stored in Outputs/")
print("  - README.md")
print("\nModel is ready for deployment!")
print("="*70)
