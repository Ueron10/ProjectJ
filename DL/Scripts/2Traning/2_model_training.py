import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import os

# Setup path relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
models_dir = os.path.join(project_root, "Models")
outputs_dir = os.path.join(project_root, "Outputs")

# Load Model
print("\n" + "="*70)
print("LOADING MODEL...")
print("="*70)

try:
    model = load_model(os.path.join(models_dir, "model_architecture_only.keras"))
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model file not found!")
    print("  Jalankan 1_model_architecture.py terlebih dahulu")
    exit(1)

# Load Data
print("\nLoading training data...")
try:
    X_train = np.load(os.path.join(outputs_dir, "X_train.npy"))
    X_test = np.load(os.path.join(outputs_dir, "X_test.npy"))
    y_train = np.load(os.path.join(outputs_dir, "y_train.npy"))
    y_test = np.load(os.path.join(outputs_dir, "y_test.npy"))
    
    print("Data loaded successfully!")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  y_test shape: {y_test.shape}")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("  Jalankan splitting_data.py terlebih dahulu")
    exit(1)

# Compute Class Weights
print("\n" + "="*70)
print("COMPUTING CLASS WEIGHTS...")
print("="*70)

unique, counts = np.unique(y_train, return_counts=True)
print("\nClass Distribution:")
for idx, count in zip(unique, counts):
    label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    pct = (count / len(y_train)) * 100
    print(f"  {label_map[idx]:10} ({idx}): {count:4d} samples ({pct:5.1f}%)")

# Compute balanced weights
class_weights = compute_class_weight('balanced', 
    classes=np.unique(y_train), y=y_train)

# Boost negative and neutral class weights
class_weights[0] = class_weights[0] * 1.5
class_weights[1] = class_weights[1] * 1.3

class_weight_dict = dict(enumerate(class_weights))

print("\nClass Weights (Boosted):")
for idx, weight in class_weight_dict.items():
    label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    orig_weight = compute_class_weight('balanced', 
        classes=np.unique(y_train), y=y_train)[idx]
    boost = weight / orig_weight if orig_weight > 0 else 1
    print(f"  {label_map[idx]:10} ({idx}): {weight:.4f} ({boost:.1f}x boost)")

# Parameter Training
batch_size = 32
epochs = 30
learning_rate = 0.0005

print(f"\nTraining Parameters:")
print(f"  Batch size: {batch_size}")
print(f"  Epochs: {epochs}")
print(f"  Learning rate: {learning_rate}")

# Recompile with correct learning rate
from tensorflow.keras.optimizers import Adam
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
print("\n" + "="*70)
print("TRAINING MODEL...")
print("="*70)

early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=7,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)

# Visualisasi Training History
print("\n" + "="*70)
print("GENERATING TRAINING HISTORY PLOTS...")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Accuracy
axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Model Accuracy During Training', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot 2: Loss
axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Model Loss During Training', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
print("Training history plot saved as 'training_history.png'")
plt.close()

# Simpan Model
model.save(os.path.join(models_dir, "model_sentiment_cnn_trained.keras"))
print("\nTrained model saved as 'model_sentiment_cnn_trained.keras'")

# Print Summary
print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)
print(f"Total Epochs Run: {len(history.history['loss'])}")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
print("\nImprovements Applied:")
print("- Multi-kernel CNN (kernel 3,4,5)")
print("- Aggressive dropout (0.3-0.4)")
print("- Lower learning rate (0.0005)")
print("- Aggressive class weighting (negative/neutral 1.5x-1.3x boost)")
print("- More epochs (30) with patient early stopping (patience=7)")
print("="*70)
