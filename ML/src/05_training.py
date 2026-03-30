import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

X_train = pd.read_csv(os.path.join(BASE_DIR, 'data', 'train', 'X_train.csv'))
y_train = pd.read_csv(os.path.join(BASE_DIR, 'data', 'train', 'y_train.csv'))

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, os.path.join(BASE_DIR, 'models', 'model_regresi.pkl'))

print(f"Model trained: {X_train.shape[0]} samples, R2 will be calculated in evaluation")
