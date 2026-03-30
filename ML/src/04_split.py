import pandas as pd
from sklearn.model_selection import train_test_split
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

X = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'fitur_rumah.csv'))
y = pd.read_csv(os.path.join(BASE_DIR, 'data', 'processed', 'target_harga.csv'))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.to_csv(os.path.join(BASE_DIR, 'data', 'train', 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(BASE_DIR, 'data', 'test', 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(BASE_DIR, 'data', 'train', 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(BASE_DIR, 'data', 'test', 'y_test.csv'), index=False)

print(f"Train: X{X_train.shape}, y{y_train.shape} | Test: X{X_test.shape}, y{y_test.shape}")
