"""
Train Alphabet Model using real ASL hand images from asl_dataset/
This script:
1. Reads images from asl_dataset/a through asl_dataset/z
2. Uses MediaPipe Hands to extract 21 normalized hand landmarks
3. Adds synthetic SPACE and DELETE gesture samples
4. Trains a RandomForestClassifier
5. Saves the model as alphabet_model.pkl
"""

import csv
import cv2
import numpy as np
import os
import pickle
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Configuration
ASL_DATASET_DIR = 'asl_dataset'
OUTPUT_CSV = 'asl_real_dataset.csv'
MODEL_FILE = 'alphabet_model.pkl'
BACKUP_MODEL = 'alphabet_model_backup.pkl'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

def extract_landmarks_from_image(image_path):
    """Extract normalized hand landmarks from an image file."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if not results.multi_hand_landmarks:
        return None
    
    hand_landmarks = results.multi_hand_landmarks[0].landmark
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    
    # Normalize: center on wrist, scale by max distance
    wrist = coords[0, :]
    coords = coords - wrist
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords = coords / max_dist
    
    return coords.flatten()

def generate_synthetic_space_delete(num_samples=500):
    """Generate synthetic SPACE and DELETE landmark data."""
    samples = []
    
    for _ in range(num_samples):
        # SPACE = Open flat palm (all fingers extended, spread)
        lm = np.zeros((21, 3))
        # Wrist at origin
        # Thumb extended to the side
        lm[1] = [0.1, -0.05, 0]; lm[2] = [0.2, -0.1, 0]; lm[3] = [0.3, -0.15, 0]; lm[4] = [0.4, -0.2, 0]
        # Index finger straight up
        lm[5] = [-0.05, -0.15, 0]; lm[6] = [-0.05, -0.3, 0]; lm[7] = [-0.05, -0.45, 0]; lm[8] = [-0.05, -0.55, 0]
        # Middle finger straight up
        lm[9] = [0.0, -0.15, 0]; lm[10] = [0.0, -0.3, 0]; lm[11] = [0.0, -0.48, 0]; lm[12] = [0.0, -0.58, 0]
        # Ring finger straight up
        lm[13] = [0.05, -0.15, 0]; lm[14] = [0.05, -0.28, 0]; lm[15] = [0.05, -0.42, 0]; lm[16] = [0.05, -0.52, 0]
        # Pinky straight up
        lm[17] = [0.1, -0.12, 0]; lm[18] = [0.1, -0.24, 0]; lm[19] = [0.1, -0.36, 0]; lm[20] = [0.1, -0.46, 0]
        
        # Normalize
        max_dist = np.max(np.linalg.norm(lm, axis=1))
        if max_dist > 0: lm = lm / max_dist
        # Add noise
        lm += np.random.normal(0, 0.025, lm.shape)
        samples.append(('SPACE', lm.flatten()))
    
    for _ in range(num_samples):
        # DELETE = Thumbs up (fist with thumb extended up)
        lm = np.zeros((21, 3))
        # Thumb extended upward
        lm[1] = [0.08, -0.05, 0]; lm[2] = [0.1, -0.15, 0]; lm[3] = [0.1, -0.3, 0]; lm[4] = [0.1, -0.42, 0]
        # Index curled into fist
        lm[5] = [-0.02, -0.1, 0]; lm[6] = [-0.02, -0.15, 0]; lm[7] = [-0.02, -0.08, -0.08]; lm[8] = [-0.02, -0.02, -0.1]
        # Middle curled
        lm[9] = [0.02, -0.1, 0]; lm[10] = [0.02, -0.15, 0]; lm[11] = [0.02, -0.08, -0.08]; lm[12] = [0.02, -0.02, -0.1]
        # Ring curled
        lm[13] = [0.05, -0.1, 0]; lm[14] = [0.05, -0.14, 0]; lm[15] = [0.05, -0.08, -0.08]; lm[16] = [0.05, -0.02, -0.1]
        # Pinky curled
        lm[17] = [0.08, -0.08, 0]; lm[18] = [0.08, -0.12, 0]; lm[19] = [0.08, -0.06, -0.08]; lm[20] = [0.08, 0.0, -0.1]
        
        # Normalize
        max_dist = np.max(np.linalg.norm(lm, axis=1))
        if max_dist > 0: lm = lm / max_dist
        # Add noise
        lm += np.random.normal(0, 0.025, lm.shape)
        samples.append(('DELETE', lm.flatten()))
    
    return samples

def process_asl_dataset():
    """Process all images in asl_dataset/ and extract landmarks."""
    all_samples = []
    letters = sorted([d for d in os.listdir(ASL_DATASET_DIR) 
                      if os.path.isdir(os.path.join(ASL_DATASET_DIR, d)) and len(d) == 1])
    
    print(f"Found letter directories: {letters}")
    
    for letter in letters:
        letter_dir = os.path.join(ASL_DATASET_DIR, letter)
        images = [f for f in os.listdir(letter_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        success_count = 0
        for img_file in images:
            img_path = os.path.join(letter_dir, img_file)
            landmarks = extract_landmarks_from_image(img_path)
            if landmarks is not None:
                label = letter.upper()
                all_samples.append((label, landmarks))
                success_count += 1
        
        print(f"  Letter '{letter.upper()}': {success_count}/{len(images)} images processed")
    
    return all_samples

def save_csv(samples, output_file):
    """Save samples to CSV."""
    header = ['label'] + [f'{c}{i}' for i in range(21) for c in ['x', 'y', 'z']]
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for label, features in samples:
            writer.writerow([label] + features.tolist())
    print(f"Saved {len(samples)} samples to {output_file}")

def main():
    print("=" * 60)
    print("ASL Dataset Alphabet Model Training")
    print("=" * 60)
    
    # Step 1: Process real images
    print("\n[1/4] Processing ASL dataset images...")
    real_samples = process_asl_dataset()
    print(f"Total real samples extracted: {len(real_samples)}")
    
    # Step 2: Generate synthetic SPACE and DELETE
    print("\n[2/4] Generating synthetic SPACE and DELETE samples...")
    synthetic_samples = generate_synthetic_space_delete(num_samples=500)
    print(f"Synthetic samples: {len(synthetic_samples)}")
    
    # Combine
    all_samples = real_samples + synthetic_samples
    print(f"Total combined samples: {len(all_samples)}")
    
    # Save CSV
    save_csv(all_samples, OUTPUT_CSV)
    
    # Step 3: Train model
    print("\n[3/4] Training RandomForest model...")
    labels = [s[0] for s in all_samples]
    features = np.array([s[1] for s in all_samples])
    
    unique_labels = sorted(set(labels))
    print(f"Classes ({len(unique_labels)}): {unique_labels}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Step 4: Save model (backup old one first)
    print(f"\n[4/4] Saving model...")
    if os.path.exists(MODEL_FILE):
        os.rename(MODEL_FILE, BACKUP_MODEL)
        print(f"Old model backed up to {BACKUP_MODEL}")
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    print(f"New model saved to {MODEL_FILE}")
    print("\nDone! ✅")

if __name__ == "__main__":
    main()
