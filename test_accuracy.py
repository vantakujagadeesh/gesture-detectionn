import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Load vocabulary
try:
    with open('actions.txt', 'r') as f:
        actions = np.array([line.strip() for line in f.readlines()])
except FileNotFoundError:
    print("actions.txt not found.")
    exit()

def test_model():
    print("Loading test data...")
    try:
        X_test = np.load('X_test.npy')
        y_test = np.load('y_test.npy')
    except FileNotFoundError:
        print("Test data (X_test.npy, y_test.npy) not found. Run train_model.py first.")
        return

    print(f"Test Set Size: {len(X_test)} samples")
    
    print("Loading model...")
    try:
        model = load_model('action.h5')
    except:
        print("Model action.h5 not found.")
        return

    # Latency Testing
    print("\n--- Latency Test (Single Inference) ---")
    start_time = time.time()
    _ = model.predict(np.expand_dims(X_test[0], axis=0), verbose=0)
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    print(f"Inference Latency: {latency_ms:.2f} ms")
    if latency_ms < 50:
        print("Result: PASS (Real-time capable)")
    else:
        print("Result: WARNING (May lag in real-time)")

    # Accuracy Testing
    print("\n--- Accuracy Test ---")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    acc = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {acc*100:.2f}%")
    
    if acc >= 0.9:
        print("Result: PASS (>= 90% accuracy)")
    else:
        print("Result: WARNING (< 90% accuracy - Consider more data or larger model)")

    # Per-Class Report (only for classes present in test set)
    present_classes = np.unique(np.concatenate((y_true, y_pred)))
    target_names = [actions[i] for i in present_classes]
    
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, labels=present_classes, target_names=target_names))

    # Confusion Matrix (optional visualization)
    # cm = confusion_matrix(y_true, y_pred)
    # plt.figure(figsize=(10,7))
    # sns.heatmap(cm, annot=True, xticklabels=target_names, yticklabels=target_names)
    # plt.savefig('confusion_matrix.png')
    # print("Confusion matrix saved to confusion_matrix.png")

if __name__ == "__main__":
    test_model()
