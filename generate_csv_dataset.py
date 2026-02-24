import csv
import numpy as np
import os
import random

# Define gestures and their synthetic logic
GESTURES = ['thumbs_up', 'peace', 'wave', 'fist', 'point', 'ok']
SAMPLES_PER_CLASS = 600 # 6 classes * 600 = 3600 samples
OUTPUT_FILE = 'gestures_dataset.csv'

def generate_hand_landmarks(gesture_name):
    # 21 landmarks, 3 coordinates (x, y, z)
    # MediaPipe Hands: 0=Wrist, 4=ThumbTip, 8=IndexTip, 12=MidTip, 16=RingTip, 20=PinkyTip
    
    # Base hand (flat open palm)
    landmarks = np.zeros((21, 3))
    
    # Simulate wrist at bottom center
    landmarks[0] = [0.5, 0.9, 0] 
    
    # Helper to set finger states (Open/Closed)
    def set_finger(indices, is_open):
        # Simple linear interpolation for visual approximation in data space
        base_x = 0.5 + (indices[-1]/20 - 0.5) * 0.4 # Spread fingers
        if is_open:
            landmarks[indices[0]] = [base_x, 0.7, 0] # Joint 1
            landmarks[indices[1]] = [base_x, 0.5, 0] # Joint 2
            landmarks[indices[2]] = [base_x, 0.3, 0] # Joint 3
            landmarks[indices[3]] = [base_x, 0.1, 0] # Tip (Up)
        else:
            landmarks[indices[0]] = [base_x, 0.7, 0]
            landmarks[indices[1]] = [base_x, 0.75, -0.05]
            landmarks[indices[2]] = [base_x, 0.8, -0.1]
            landmarks[indices[3]] = [base_x, 0.85, -0.15] # Tip (Curled down)

    thumb = [1, 2, 3, 4]
    index = [5, 6, 7, 8]
    middle = [9, 10, 11, 12]
    ring = [13, 14, 15, 16]
    pinky = [17, 18, 19, 20]

    if gesture_name == 'thumbs_up':
        set_finger(thumb, True)
        landmarks[4] = [0.8, 0.5, 0] # Thumb sticking out to side/up
        set_finger(index, False)
        set_finger(middle, False)
        set_finger(ring, False)
        set_finger(pinky, False)
        
    elif gesture_name == 'peace':
        set_finger(thumb, False)
        set_finger(index, True)
        set_finger(middle, True)
        set_finger(ring, False)
        set_finger(pinky, False)
        # Spread V
        landmarks[8][0] -= 0.1 # Index left
        landmarks[12][0] += 0.1 # Middle right
        
    elif gesture_name == 'wave':
        set_finger(thumb, True)
        set_finger(index, True)
        set_finger(middle, True)
        set_finger(ring, True)
        set_finger(pinky, True)
        
    elif gesture_name == 'fist':
        set_finger(thumb, False)
        set_finger(index, False)
        set_finger(middle, False)
        set_finger(ring, False)
        set_finger(pinky, False)
        
    elif gesture_name == 'point':
        set_finger(thumb, False)
        set_finger(index, True)
        set_finger(middle, False)
        set_finger(ring, False)
        set_finger(pinky, False)
        
    elif gesture_name == 'ok':
        set_finger(thumb, True)
        set_finger(index, True)
        # Connect thumb and index
        landmarks[4] = [0.6, 0.4, 0]
        landmarks[8] = [0.6, 0.4, 0]
        set_finger(middle, True)
        set_finger(ring, True)
        set_finger(pinky, True)

    # Add noise to make it robust
    noise = np.random.normal(0, 0.02, landmarks.shape)
    landmarks += noise
    
    return landmarks.flatten()

def main():
    print(f"Generating {len(GESTURES) * SAMPLES_PER_CLASS} samples...")
    
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header: label, x0, y0, z0, ...
        header = ['label'] + [f'{c}{i}' for i in range(21) for c in ['x', 'y', 'z']]
        writer.writerow(header)
        
        for gesture in GESTURES:
            print(f"Generating {gesture}...")
            for _ in range(SAMPLES_PER_CLASS):
                landmarks = generate_hand_landmarks(gesture)
                row = [gesture] + landmarks.tolist()
                writer.writerow(row)
                
    print(f"Dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
