import cv2
import mediapipe as mp
import numpy as np
import os
import csv
from camera import MediapipeHelper

def process_asl_dataset():
    helper = MediapipeHelper()
    dataset_path = 'asl_dataset'
    output_file = 'alphabet_dataset.csv'
    
    # Get labels from folder names
    labels = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    
    print(f"Processing labels: {labels}")
    
    # Prepare CSV header
    header = ['label'] + [f'{c}{i}' for i in range(21) for c in ['x', 'y', 'z']]
    
    data_rows = []
    
    with helper.hands as hands:
        for label in labels:
            label_path = os.path.join(dataset_path, label)
            images = [f for f in os.listdir(label_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"Processing {label} ({len(images)} images)...")
            
            for img_name in images:
                img_path = os.path.join(label_path, img_name)
                image = cv2.imread(img_path)
                if image is None: continue
                
                # Detect landmarks
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                
                if results.multi_hand_landmarks:
                    # Use the first hand detected
                    hand_landmarks = results.multi_hand_landmarks[0].landmark
                    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
                    
                    # Normalize (same as in camera.py)
                    wrist = coords[0, :]
                    coords = coords - wrist
                    max_dist = np.max(np.linalg.norm(coords, axis=1))
                    if max_dist > 0:
                        coords = coords / max_dist
                    
                    data_rows.append([label.upper()] + coords.flatten().tolist())
                    
                    # DATA AUGMENTATION: Flip X for left hand support
                    flipped_coords = coords.copy()
                    flipped_coords[:, 0] = -flipped_coords[:, 0]
                    data_rows.append([label.upper()] + flipped_coords.flatten().tolist())

    # Now add SPACE and DELETE from synthetic generation to ensure they are included
    # (Since they are not in the image dataset)
    # Reduced samples to match the image dataset (around 70-100)
    print("Adding balanced synthetic SPACE and DELETE samples...")
    import generate_alphabet_dataset
    for label in ['SPACE', 'DELETE']:
        for _ in range(100): # Balanced with others
            sample = generate_alphabet_dataset.generate_sample(label)
            data_rows.append([label] + sample.tolist())
            
            # Flip X for left hand support
            sample_reshaped = sample.reshape(21, 3).copy()
            sample_reshaped[:, 0] = -sample_reshaped[:, 0]
            data_rows.append([label] + sample_reshaped.flatten().tolist())

    print(f"Total samples collected: {len(data_rows)}")
    
    # Save to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_rows)
    
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    process_asl_dataset()
