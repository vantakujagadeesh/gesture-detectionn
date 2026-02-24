import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATASET_FILE = 'gestures_dataset.csv'
MODEL_FILE = 'static_gesture_model.pkl'

def train_static_gesture_model():
    print(f"Loading static gesture dataset from {DATASET_FILE}...")
    try:
        df = pd.read_csv(DATASET_FILE)
    except FileNotFoundError:
        print(f"Error: {DATASET_FILE} not found.")
        return

    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    
    print(f"Dataset shape: {X.shape}")
    print(f"Classes ({len(np.unique(y))}): {np.unique(y)}")
    
    # Data Augmentation: Flip X coordinates to support both hands
    print("Performing data augmentation (X-axis flipping)...")
    X_reshaped = X.reshape(-1, 21, 3)
    X_flipped = X_reshaped.copy()
    X_flipped[:, :, 0] = -X_flipped[:, :, 0]
    X_flipped = X_flipped.reshape(-1, 63)
    
    X = np.concatenate([X, X_flipped])
    y = np.concatenate([y, y])
    
    print(f"Augmented dataset shape: {X.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier for Static Gestures...")
    # Using more estimators and max_depth for better performance
    model = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    print(f"Saving model to {MODEL_FILE}...")
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    print("Done.")

if __name__ == "__main__":
    train_static_gesture_model()
