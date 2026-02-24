import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATASET_FILE = 'alphabet_dataset.csv'
MODEL_FILE = 'alphabet_model.pkl'

def train_alphabet_model():
    print("Loading alphabet dataset...")
    try:
        df = pd.read_csv(DATASET_FILE)
    except FileNotFoundError:
        print(f"Error: {DATASET_FILE} not found.")
        return

    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    
    print(f"Dataset shape: {X.shape}")
    print(f"Classes ({len(np.unique(y))}): {np.unique(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier (A-Z)...")
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    print(f"Saving model to {MODEL_FILE}...")
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    print("Done.")

if __name__ == "__main__":
    train_alphabet_model()
