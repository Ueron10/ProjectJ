import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Embedding, Conv1D, Dense, Dropout, GlobalMaxPooling1D, 
    BatchNormalization, concatenate, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Setup path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
models_dir = os.path.join(project_root, "Models")
outputs_dir = os.path.join(project_root, "Outputs")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

print("="*70)
print("BUILD & TRAIN CNN MODEL")
print("="*70)

# ============ BUILD MODEL ============
print("\n[1/4] Building CNN architecture...")

vocab_size = 5000
embedding_dim = 128
max_length = 100
num_filters = 128

# Multi-kernel CNN Architecture
inputs = Input(shape=(max_length,))
x = Embedding(vocab_size, embedding_dim, input_length=max_length)(inputs)

# Multiple Conv1D kernels (3, 4, 5)
conv3 = Conv1D(num_filters, 3, activation='relu')(x)
conv3 = BatchNormalization()(conv3)
conv3 = GlobalMaxPooling1D()(conv3)

conv4 = Conv1D(num_filters, 4, activation='relu')(x)
conv4 = BatchNormalization()(conv4)
conv4 = GlobalMaxPooling1D()(conv4)

conv5 = Conv1D(num_filters, 5, activation='relu')(x)
conv5 = BatchNormalization()(conv5)
conv5 = GlobalMaxPooling1D()(conv5)

# Concatenate
concat = concatenate([conv3, conv4, conv5], axis=1)

# Dense layers
x = Dense(128, activation='relu')(concat)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(64, activation='relu')(x)
x = Dropout(0.4)(x)

# Output layer (binary: positive/negative)
outputs = Dense(2, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model architecture built!")
model.summary()

# ============ LOAD DATA ============
print("\n[2/4] Loading training data...")

X_train = np.load(os.path.join(outputs_dir, "X_train.npy"))
X_test = np.load(os.path.join(outputs_dir, "X_test.npy"))
y_train = np.load(os.path.join(outputs_dir, "y_train.npy"))
y_test = np.load(os.path.join(outputs_dir, "y_test.npy"))

print(f"  Training: {X_train.shape}")
print(f"  Testing:  {X_test.shape}")

# ============ TRAIN MODEL ============
print("\n[3/4] Training model...")

batch_size = 32
epochs = 50

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True,
    mode='min',
    verbose=1
)

lr_reducer = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=0.00001,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, lr_reducer],
    verbose=1
)

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], label='Training')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], label='Training')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Model Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(outputs_dir, 'training_history.png'), dpi=300)
print("\nTraining plot saved!")

# ============ SAVE MODEL ============
print("\n[4/4] Saving model...")

model.save(os.path.join(models_dir, "model_final.keras"))

best_val_acc = max(history.history['val_accuracy'])
final_epoch = len(history.history['loss'])

print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)
print(f"Epochs completed: {final_epoch}")
print(f"Best val accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print(f"Final train accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"\nModel saved to: Models/model_final.keras")
print("="*70)
