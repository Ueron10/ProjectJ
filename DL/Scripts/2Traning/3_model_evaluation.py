import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tensorflow.keras.models import load_model
import seaborn as sns
import os

# Setup path relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
models_dir = os.path.join(project_root, "Models")
outputs_dir = os.path.join(project_root, "Outputs")

# Load Model
print("\n" + "="*70)
print("LOADING TRAINED MODEL...")
print("="*70)

try:
    model = load_model(os.path.join(models_dir, "model_sentiment_cnn_trained.keras"))
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model file not found!")
    print("  Jalankan 2_model_training.py terlebih dahulu")
    exit(1)

# Load Data
print("\nLoading test data...")
try:
    X_test = np.load(os.path.join(outputs_dir, "X_test.npy"))
    y_test = np.load(os.path.join(outputs_dir, "y_test.npy"))
    label_classes = np.load(os.path.join(outputs_dir, "label_encoding.npy"), allow_pickle=True)
    
    print("Data loaded successfully!")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")
    print(f"  Classes: {label_classes}")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("  Pastikan splitting_data.py sudah dijalankan")
    exit(1)

# Evaluate Model
print("\n" + "="*70)
print("EVALUATING MODEL ON TEST SET...")
print("="*70)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Prediksi
print("\nGenerating predictions...")
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion Matrix
print("\n" + "="*70)
print("CONFUSION MATRIX")
print("="*70)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (Raw):")
print(cm)

# Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_classes, yticklabels=label_classes,
            cbar_kws={'label': 'Count'}, ax=ax)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('Confusion Matrix - Sentiment Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\nConfusion matrix plot saved as 'confusion_matrix.png'")
plt.show()

# Classification Report
print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)

print("\n")
print(classification_report(y_test, y_pred, target_names=label_classes))

# Metrics Detail
print("="*70)
print("DETAILED METRICS")
print("="*70)

from sklearn.metrics import precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nOverall Metrics (Weighted Average):")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print("="*70)

# Save Evaluation Results
print("\nEvaluation complete!")
print("  - confusion_matrix.png")
print("  - evaluation results printed above")
