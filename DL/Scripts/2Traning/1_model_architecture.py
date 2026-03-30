import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, Dropout, Dense, GlobalMaxPooling1D, Concatenate
)
from tensorflow.keras.optimizers import Adam
import os

# Setup path relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
models_dir = os.path.join(project_root, "Models")

# Create Models directory if doesn't exist
os.makedirs(models_dir, exist_ok=True)

print("Building CNN Model")
print("="*70)

# Multi-kernel CNN Architecture
inputs = Input(shape=(100,))

# Embedding layer
x = Embedding(input_dim=10000, output_dim=128)(inputs)
x = Dropout(0.2)(x)

# Multiple parallel convolutional layers with different kernel sizes
conv1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
conv1 = GlobalMaxPooling1D()(conv1)
conv1 = Dropout(0.3)(conv1)

conv2 = Conv1D(filters=64, kernel_size=4, padding='same', activation='relu')(x)
conv2 = GlobalMaxPooling1D()(conv2)
conv2 = Dropout(0.3)(conv2)

conv3 = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu')(x)
conv3 = GlobalMaxPooling1D()(conv3)
conv3 = Dropout(0.3)(conv3)

# Concatenate all parallel outputs
merged = Concatenate()([conv1, conv2, conv3])

# Dense layers with dropout
dense1 = Dense(128, activation='relu')(merged)
dense1 = Dropout(0.4)(dense1)

dense2 = Dense(64, activation='relu')(dense1)
dense2 = Dropout(0.3)(dense2)

# Output layer
outputs = Dense(3, activation='softmax')(dense2)

# Create model
model = Model(inputs=inputs, outputs=outputs)

# Compile with lower learning rate for stability
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Print model summary
print("\nModel Architecture Created:")
print("-" * 70)
model.summary()
print("-" * 70)

# Save architecture
output_model_path = os.path.join(models_dir, "model_architecture_only.keras")
model.save(output_model_path)
print(f"\nModel saved to: {output_model_path}")

print("\n" + "="*70)
print("Key Improvements:")
print("="*70)
print("1. Multi-kernel architecture: kernel_size [3,4,5] captures different patterns")
print("2. Parallel processing: Each kernel size gets its own conv + pooling")
print("3. Lower learning rate (0.0005): More stable training for sparse inputs")
print("4. Aggressive dropout (0.3-0.4): Prevents overfitting on sparse sequences")
print("5. GlobalMaxPooling on each kernel: Extracts most important feature per kernel")
print("="*70 + "\n")

