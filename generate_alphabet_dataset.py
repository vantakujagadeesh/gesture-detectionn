import csv
import numpy as np
import os
import random

# Full ASL Alphabet + Space/Delete
LETTERS = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'SPACE', 'DELETE'
]

SAMPLES_PER_CLASS = 1000
OUTPUT_FILE = 'alphabet_dataset.csv'

def get_base_hand():
    return np.zeros((21, 3))

def set_finger(landmarks, finger_idx, state, spread=0.0):
    indices = [
        [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]
    ]
    idx_list = indices[finger_idx]
    angle = (finger_idx - 2) * 0.2 + spread
    
    if finger_idx == 0: # Thumb
        if state == 'OPEN':
            landmarks[idx_list[0]] = [0.1, -0.1, 0]
            landmarks[idx_list[3]] = [0.4, -0.2, 0]
        elif state == 'CLOSED': # Tucked side
            landmarks[idx_list[3]] = [0.15, -0.1, -0.05]
        elif state == 'FIST': # Across fingers
             landmarks[idx_list[3]] = [0.0, 0.0, -0.1]
    else: # Fingers
        if state == 'OPEN':
            base_x = np.sin(angle) * 0.2
            base_y = -np.cos(angle) * 0.2 
            landmarks[idx_list[0]] = [base_x, base_y, 0]
            landmarks[idx_list[3]] = [base_x * 2.8, base_y * 2.8, 0]
        elif state == 'CLOSED': # Fist
            base_x = np.sin(angle) * 0.1
            base_y = -np.cos(angle) * 0.1
            landmarks[idx_list[0]] = [base_x, base_y, 0]
            landmarks[idx_list[3]] = [base_x, base_y + 0.1, -0.1]
        elif state == 'CURVED': # C/E/O
             base_x = np.sin(angle) * 0.15
             base_y = -np.cos(angle) * 0.15
             landmarks[idx_list[3]] = [base_x, base_y, -0.1] # Tips forward

def generate_sample(letter):
    lm = get_base_hand()
    t, i, m, r, p = 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED'
    
    # Logic for A-Z
    if letter == 'A': t, i, m, r, p = 'OPEN', 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED' # Thumb side
    elif letter == 'B': t, i, m, r, p = 'FIST', 'OPEN', 'OPEN', 'OPEN', 'OPEN' # Thumb tucked
    elif letter == 'C': t, i, m, r, p = 'OPEN', 'CURVED', 'CURVED', 'CURVED', 'CURVED'
    elif letter == 'D': t, i, m, r, p = 'CLOSED', 'OPEN', 'CURVED', 'CURVED', 'CURVED'
    elif letter == 'E': t, i, m, r, p = 'FIST', 'CURVED', 'CURVED', 'CURVED', 'CURVED' # Tight
    elif letter == 'F': t, i, m, r, p = 'OPEN', 'CURVED', 'OPEN', 'OPEN', 'OPEN' # OK sign
    elif letter == 'G': t, i, m, r, p = 'CLOSED', 'OPEN', 'CLOSED', 'CLOSED', 'CLOSED' # Point side
    elif letter == 'H': t, i, m, r, p = 'CLOSED', 'OPEN', 'OPEN', 'CLOSED', 'CLOSED' # Side
    elif letter == 'I': t, i, m, r, p = 'FIST', 'CLOSED', 'CLOSED', 'CLOSED', 'OPEN'
    elif letter == 'J': t, i, m, r, p = 'FIST', 'CLOSED', 'CLOSED', 'CLOSED', 'OPEN' # Motion handled by logic? Static J is I with twist
    elif letter == 'K': t, i, m, r, p = 'OPEN', 'OPEN', 'OPEN', 'CLOSED', 'CLOSED' # Thumb between
    elif letter == 'L': t, i, m, r, p = 'OPEN', 'OPEN', 'CLOSED', 'CLOSED', 'CLOSED'
    elif letter == 'M': t, i, m, r, p = 'FIST', 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED' # Thumb under 3
    elif letter == 'N': t, i, m, r, p = 'FIST', 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED' # Thumb under 2
    elif letter == 'O': t, i, m, r, p = 'OPEN', 'CURVED', 'CURVED', 'CURVED', 'CURVED' # Tips touch
    elif letter == 'P': t, i, m, r, p = 'OPEN', 'OPEN', 'OPEN', 'CLOSED', 'CLOSED' # Down
    elif letter == 'Q': t, i, m, r, p = 'OPEN', 'OPEN', 'CLOSED', 'CLOSED', 'CLOSED' # Down
    elif letter == 'R': t, i, m, r, p = 'FIST', 'OPEN', 'OPEN', 'CLOSED', 'CLOSED' # Crossed
    elif letter == 'S': t, i, m, r, p = 'FIST', 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED' # Thumb over
    elif letter == 'T': t, i, m, r, p = 'FIST', 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED' # Thumb between
    elif letter == 'U': t, i, m, r, p = 'FIST', 'OPEN', 'OPEN', 'CLOSED', 'CLOSED' # Together
    elif letter == 'V': t, i, m, r, p = 'FIST', 'OPEN', 'OPEN', 'CLOSED', 'CLOSED' # Spread
    elif letter == 'W': t, i, m, r, p = 'FIST', 'OPEN', 'OPEN', 'OPEN', 'CLOSED'
    elif letter == 'X': t, i, m, r, p = 'FIST', 'CURVED', 'CLOSED', 'CLOSED', 'CLOSED'
    elif letter == 'Y': t, i, m, r, p = 'OPEN', 'CLOSED', 'CLOSED', 'CLOSED', 'OPEN'
    elif letter == 'Z': t, i, m, r, p = 'FIST', 'OPEN', 'CLOSED', 'CLOSED', 'CLOSED' # Point logic
    elif letter == 'SPACE': t, i, m, r, p = 'OPEN', 'OPEN', 'OPEN', 'OPEN', 'OPEN' # Flat hand
    elif letter == 'DELETE': t, i, m, r, p = 'OPEN', 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED' # Thumb out? Or Fist? Let's use Open Palm Backwards

    set_finger(lm, 0, t)
    set_finger(lm, 1, i)
    set_finger(lm, 2, m)
    set_finger(lm, 3, r)
    set_finger(lm, 4, p)
    
    # Specifics
    if letter == 'M': lm[4] = [0.0, 0.0, -0.05] # Thumb under ring
    if letter == 'N': lm[4] = [0.0, 0.0, -0.05] # Thumb under mid
    if letter == 'T': lm[4] = [0.0, 0.0, -0.05] # Thumb under index
    if letter == 'S': lm[4] = [0.0, 0.0, -0.15] # Thumb OVER fingers
    
    if letter == 'G': 
        # Rotate 90
        tmp = lm[:, 0].copy(); lm[:, 0] = -lm[:, 1]; lm[:, 1] = tmp
    if letter == 'H': 
        # Rotate 90
        tmp = lm[:, 0].copy(); lm[:, 0] = -lm[:, 1]; lm[:, 1] = tmp
        
    if letter == 'P' or letter == 'Q':
        lm[:, 1] = -lm[:, 1] # Point down

    # Normalization
    max_dist = np.max(np.linalg.norm(lm, axis=1))
    if max_dist > 0: lm = lm / max_dist
    
    # Noise
    lm += np.random.normal(0, 0.02, lm.shape)
    
    return lm.flatten()

def main():
    print("Generating Alphabet Dataset...")
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['label'] + [f'{c}{i}' for i in range(21) for c in ['x', 'y', 'z']]
        writer.writerow(header)
        for letter in LETTERS:
            for _ in range(SAMPLES_PER_CLASS):
                row = [letter] + generate_sample(letter).tolist()
                writer.writerow(row)
    print(f"Done: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
