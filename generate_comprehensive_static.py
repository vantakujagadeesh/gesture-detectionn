import csv
import numpy as np
import os
import random

# Extended Static Vocabulary
# We will generate heuristics for a subset of ASL Alphabet + Numbers + Utility gestures
GESTURES = [
    # Alphabet (Subset of distinct static shapes)
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
    # Numbers
    'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE',
    # Utility / Emojis
    'THUMBS_UP', 'THUMBS_DOWN', 'PEACE', 'OK', 'FIST', 'OPEN_PALM', 'POINT'
]

SAMPLES_PER_CLASS = 500
OUTPUT_FILE = 'comprehensive_static_dataset.csv'

def get_base_hand():
    # Wrist at 0,0,0
    landmarks = np.zeros((21, 3))
    landmarks[0] = [0.5, 0.9, 0] # Wrist
    return landmarks

def set_finger(landmarks, finger_idx, state, spread=0.0):
    # finger_idx: 0=Thumb, 1=Index, 2=Middle, 3=Ring, 4=Pinky
    # state: 'OPEN', 'CLOSED', 'HALF', 'CURVED'
    
    indices = [
        [1, 2, 3, 4],     # Thumb
        [5, 6, 7, 8],     # Index
        [9, 10, 11, 12],  # Middle
        [13, 14, 15, 16], # Ring
        [17, 18, 19, 20]  # Pinky
    ]
    
    idx_list = indices[finger_idx]
    base_x = 0.5 + (finger_idx - 2) * 0.1 + spread
    
    if finger_idx == 0: # Thumb logic is special
        if state == 'OPEN':
            landmarks[idx_list[0]] = [base_x + 0.1, 0.8, 0]
            landmarks[idx_list[1]] = [base_x + 0.15, 0.7, 0]
            landmarks[idx_list[2]] = [base_x + 0.2, 0.6, 0]
            landmarks[idx_list[3]] = [base_x + 0.25, 0.5, 0]
        elif state == 'CLOSED':
            landmarks[idx_list[0]] = [base_x + 0.05, 0.8, 0]
            landmarks[idx_list[1]] = [base_x + 0.1, 0.75, -0.05]
            landmarks[idx_list[2]] = [base_x + 0.08, 0.7, -0.05]
            landmarks[idx_list[3]] = [base_x + 0.05, 0.65, -0.05] # Tucked
            
    else: # Fingers
        if state == 'OPEN':
            landmarks[idx_list[0]] = [base_x, 0.7, 0]
            landmarks[idx_list[1]] = [base_x, 0.5, 0]
            landmarks[idx_list[2]] = [base_x, 0.3, 0]
            landmarks[idx_list[3]] = [base_x, 0.1, 0]
        elif state == 'CLOSED':
            landmarks[idx_list[0]] = [base_x, 0.7, 0]
            landmarks[idx_list[1]] = [base_x, 0.75, -0.05]
            landmarks[idx_list[2]] = [base_x, 0.8, -0.1]
            landmarks[idx_list[3]] = [base_x, 0.85, -0.1]
        elif state == 'CURVED': # Like 'C' or 'E'
            landmarks[idx_list[0]] = [base_x, 0.7, 0]
            landmarks[idx_list[1]] = [base_x, 0.6, -0.02]
            landmarks[idx_list[2]] = [base_x, 0.65, -0.05]
            landmarks[idx_list[3]] = [base_x, 0.75, -0.08]

def generate_sample(gesture):
    lm = get_base_hand()
    
    # Defaults
    t, i, m, r, p = 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED'
    
    # Logic Map
    if gesture == 'A': t, i, m, r, p = 'OPEN', 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED' # Thumb alongside
    elif gesture == 'B': t, i, m, r, p = 'CLOSED', 'OPEN', 'OPEN', 'OPEN', 'OPEN' # Thumb tucked
    elif gesture == 'C': t, i, m, r, p = 'OPEN', 'CURVED', 'CURVED', 'CURVED', 'CURVED'
    elif gesture == 'D': t, i, m, r, p = 'CLOSED', 'OPEN', 'CURVED', 'CURVED', 'CURVED'
    elif gesture == 'E': t, i, m, r, p = 'CLOSED', 'CURVED', 'CURVED', 'CURVED', 'CURVED'
    elif gesture == 'F': t, i, m, r, p = 'CLOSED', 'CLOSED', 'OPEN', 'OPEN', 'OPEN' # OK sign actually
    elif gesture == 'G': t, i, m, r, p = 'CLOSED', 'OPEN', 'CLOSED', 'CLOSED', 'CLOSED' # Pointing side
    elif gesture == 'H': t, i, m, r, p = 'CLOSED', 'OPEN', 'OPEN', 'CLOSED', 'CLOSED' # Horizontal
    elif gesture == 'I': t, i, m, r, p = 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED', 'OPEN'
    elif gesture == 'K': t, i, m, r, p = 'OPEN', 'OPEN', 'OPEN', 'CLOSED', 'CLOSED' # V with thumb in between
    elif gesture == 'L': t, i, m, r, p = 'OPEN', 'OPEN', 'CLOSED', 'CLOSED', 'CLOSED'
    elif gesture == 'M': t, i, m, r, p = 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED' # 3 fingers over thumb
    elif gesture == 'N': t, i, m, r, p = 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED' # 2 fingers over thumb
    elif gesture == 'O': t, i, m, r, p = 'CLOSED', 'CURVED', 'CURVED', 'CURVED', 'CURVED' # Tips touch thumb
    elif gesture == 'P': t, i, m, r, p = 'OPEN', 'OPEN', 'OPEN', 'CLOSED', 'CLOSED' # Downward K
    elif gesture == 'Q': t, i, m, r, p = 'OPEN', 'OPEN', 'CLOSED', 'CLOSED', 'CLOSED' # Downward G
    elif gesture == 'R': t, i, m, r, p = 'CLOSED', 'OPEN', 'OPEN', 'CLOSED', 'CLOSED' # Crossed
    elif gesture == 'S': t, i, m, r, p = 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED' # Fist thumb over
    elif gesture == 'T': t, i, m, r, p = 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED' # Thumb between index/mid
    elif gesture == 'U': t, i, m, r, p = 'CLOSED', 'OPEN', 'OPEN', 'CLOSED', 'CLOSED' # Together
    elif gesture == 'V': t, i, m, r, p = 'CLOSED', 'OPEN', 'OPEN', 'CLOSED', 'CLOSED' # Spread
    elif gesture == 'W': t, i, m, r, p = 'CLOSED', 'OPEN', 'OPEN', 'OPEN', 'CLOSED'
    elif gesture == 'X': t, i, m, r, p = 'CLOSED', 'CURVED', 'CLOSED', 'CLOSED', 'CLOSED'
    elif gesture == 'Y': t, i, m, r, p = 'OPEN', 'CLOSED', 'CLOSED', 'CLOSED', 'OPEN'
    
    elif gesture == 'ONE': t, i, m, r, p = 'CLOSED', 'OPEN', 'CLOSED', 'CLOSED', 'CLOSED'
    elif gesture == 'TWO': t, i, m, r, p = 'CLOSED', 'OPEN', 'OPEN', 'CLOSED', 'CLOSED'
    elif gesture == 'THREE': t, i, m, r, p = 'OPEN', 'OPEN', 'OPEN', 'CLOSED', 'CLOSED' # ASL 3 uses thumb
    elif gesture == 'FOUR': t, i, m, r, p = 'CLOSED', 'OPEN', 'OPEN', 'OPEN', 'OPEN'
    elif gesture == 'FIVE': t, i, m, r, p = 'OPEN', 'OPEN', 'OPEN', 'OPEN', 'OPEN'
    
    elif gesture == 'THUMBS_UP': t, i, m, r, p = 'OPEN', 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED'
    elif gesture == 'THUMBS_DOWN': t, i, m, r, p = 'OPEN', 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED' # Rotation handled by noise? No, need logic.
    elif gesture == 'PEACE': t, i, m, r, p = 'CLOSED', 'OPEN', 'OPEN', 'CLOSED', 'CLOSED'
    elif gesture == 'OK': t, i, m, r, p = 'OPEN', 'CURVED', 'OPEN', 'OPEN', 'OPEN' # Thumb index touch
    elif gesture == 'FIST': t, i, m, r, p = 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED'
    elif gesture == 'OPEN_PALM': t, i, m, r, p = 'OPEN', 'OPEN', 'OPEN', 'OPEN', 'OPEN'
    elif gesture == 'POINT': t, i, m, r, p = 'CLOSED', 'OPEN', 'CLOSED', 'CLOSED', 'CLOSED'

    # Apply
    set_finger(lm, 0, t)
    set_finger(lm, 1, i)
    set_finger(lm, 2, m)
    set_finger(lm, 3, r)
    set_finger(lm, 4, p)
    
    # Specific tweaks for collisions/distinct shapes
    if gesture == 'A': lm[4] = [0.45, 0.6, -0.05] # Thumb on side
    if gesture == 'S': lm[4] = [0.5, 0.65, -0.1] # Thumb over fingers
    if gesture == 'T': lm[4] = [0.55, 0.65, -0.1] # Thumb between
    if gesture == 'M': lm[4] = [0.6, 0.8, -0.05] # Thumb under 3
    if gesture == 'N': lm[4] = [0.55, 0.8, -0.05] # Thumb under 2
    
    if gesture == 'V': 
        lm[8][0] -= 0.05 # Spread Index
        lm[12][0] += 0.05 # Spread Middle
        
    if gesture == 'U':
        lm[8][0] = 0.5 # Together
        lm[12][0] = 0.52 
        
    if gesture == 'THUMBS_DOWN':
        # Rotate 180 (Simple flip of Y)
        lm[:, 1] = 1.0 - lm[:, 1]

    # Add Random Noise for Robustness
    noise = np.random.normal(0, 0.015, lm.shape)
    lm += noise
    
    return lm.flatten()

def main():
    print(f"Generating {len(GESTURES) * SAMPLES_PER_CLASS} samples...")
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['label'] + [f'{c}{i}' for i in range(21) for c in ['x', 'y', 'z']]
        writer.writerow(header)
        
        for gesture in GESTURES:
            # print(f"Generating {gesture}...")
            for _ in range(SAMPLES_PER_CLASS):
                row = [gesture] + generate_sample(gesture).tolist()
                writer.writerow(row)
    print("Done.")

if __name__ == "__main__":
    main()
