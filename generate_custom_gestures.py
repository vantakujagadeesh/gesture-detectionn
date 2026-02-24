import csv
import numpy as np
import os
import random

GESTURES = [
    'ONE_FINGER', 'TWO_FINGERS', 'OPEN_PALM', 'CLOSED_FIST',
    'THUMBS_UP', 'THUMBS_DOWN', 'FIVE_FINGERS', 'THREE_FINGERS',
    'FOUR_FINGERS', 'POINT_FORWARD', 'POINT_DOWN', 'POINT_SIDE',
    'OK_SIGN', 'FINGERS_CROSSED', 'ROCK_SIGN', 'THREE_FINGERS_THUMB',
    'FOUR_FINGERS_FOLD', 'TWO_FINGERS_FWD', 'TAP_TWICE', 'WAVE'
]

SAMPLES_PER_CLASS = 1200 
OUTPUT_FILE = 'custom_gestures_normalized.csv'

def get_base_hand():
    return np.zeros((21, 3))

def rotate_z(landmarks, angle_deg):
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    x = landmarks[:, 0].copy()
    y = landmarks[:, 1].copy()
    landmarks[:, 0] = x * cos_a - y * sin_a
    landmarks[:, 1] = x * sin_a + y * cos_a

def set_finger(landmarks, finger_idx, state, spread=0.0):
    indices = [
        [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]
    ]
    idx_list = indices[finger_idx]
    angle = (finger_idx - 2) * 0.2 + spread
    
    if finger_idx == 0: # Thumb
        if state == 'OPEN':
            landmarks[idx_list[0]] = [0.1, -0.1, 0]
            landmarks[idx_list[1]] = [0.2, -0.2, 0]
            landmarks[idx_list[2]] = [0.3, -0.25, 0]
            landmarks[idx_list[3]] = [0.4, -0.2, 0]
        elif state == 'CLOSED':
            landmarks[idx_list[0]] = [0.1, -0.05, 0]
            landmarks[idx_list[3]] = [0.15, 0.0, -0.05]
        elif state == 'TUCKED':
             landmarks[idx_list[0]] = [0.1, -0.05, 0]
             landmarks[idx_list[3]] = [0.0, 0.0, -0.08] # Across palm
    else: # Fingers
        if state == 'OPEN':
            base_x = np.sin(angle) * 0.2
            base_y = -np.cos(angle) * 0.2 
            landmarks[idx_list[0]] = [base_x, base_y, 0]
            landmarks[idx_list[1]] = [base_x * 1.8, base_y * 1.8, 0]
            landmarks[idx_list[2]] = [base_x * 2.4, base_y * 2.4, 0]
            landmarks[idx_list[3]] = [base_x * 2.8, base_y * 2.8, 0]
        elif state == 'CLOSED':
            base_x = np.sin(angle) * 0.1
            base_y = -np.cos(angle) * 0.1
            landmarks[idx_list[0]] = [base_x, base_y, 0]
            landmarks[idx_list[3]] = [base_x, base_y + 0.1, -0.1]
        elif state == 'FORWARD': 
             landmarks[idx_list[0]] = [np.sin(angle)*0.2, -0.2, 0]
             landmarks[idx_list[3]] = [np.sin(angle)*0.2, -0.2, -0.3]

def generate_sample(gesture):
    lm = get_base_hand()
    t, i, m, r, p = 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED'
    
    # Defaults
    spread_val = 0.0 # Standard spread
    
    if gesture == 'ONE_FINGER': t, i, m, r, p = 'CLOSED', 'OPEN', 'CLOSED', 'CLOSED', 'CLOSED'
    elif gesture == 'TWO_FINGERS': t, i, m, r, p = 'CLOSED', 'OPEN', 'OPEN', 'CLOSED', 'CLOSED'
    elif gesture == 'OPEN_PALM': t, i, m, r, p = 'OPEN', 'OPEN', 'OPEN', 'OPEN', 'OPEN'
    elif gesture == 'CLOSED_FIST': t, i, m, r, p = 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED'
    elif gesture == 'THUMBS_UP': t, i, m, r, p = 'OPEN', 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED'
    elif gesture == 'THUMBS_DOWN': t, i, m, r, p = 'OPEN', 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED'
    elif gesture == 'FIVE_FINGERS': t, i, m, r, p = 'OPEN', 'OPEN', 'OPEN', 'OPEN', 'OPEN'
    elif gesture == 'THREE_FINGERS': t, i, m, r, p = 'CLOSED', 'OPEN', 'OPEN', 'OPEN', 'CLOSED'
    
    # 9. Four Fingers (Spread, Thumb Closed)
    elif gesture == 'FOUR_FINGERS': 
        t, i, m, r, p = 'CLOSED', 'OPEN', 'OPEN', 'OPEN', 'OPEN'
        spread_val = 0.15 # Splayed
        
    elif gesture == 'POINT_FORWARD': t, i, m, r, p = 'CLOSED', 'FORWARD', 'CLOSED', 'CLOSED', 'CLOSED'
    elif gesture == 'POINT_DOWN': t, i, m, r, p = 'CLOSED', 'OPEN', 'CLOSED', 'CLOSED', 'CLOSED'
    elif gesture == 'POINT_SIDE': t, i, m, r, p = 'CLOSED', 'OPEN', 'CLOSED', 'CLOSED', 'CLOSED'
    elif gesture == 'OK_SIGN': t, i, m, r, p = 'OPEN', 'CLOSED', 'OPEN', 'OPEN', 'OPEN'
    elif gesture == 'FINGERS_CROSSED': t, i, m, r, p = 'CLOSED', 'OPEN', 'OPEN', 'CLOSED', 'CLOSED'
    elif gesture == 'ROCK_SIGN': t, i, m, r, p = 'CLOSED', 'OPEN', 'CLOSED', 'CLOSED', 'OPEN'
    elif gesture == 'THREE_FINGERS_THUMB': t, i, m, r, p = 'OPEN', 'OPEN', 'OPEN', 'CLOSED', 'CLOSED'
    
    # 17. Four Fingers Folded (Fingers Together, Thumb Tucked)
    elif gesture == 'FOUR_FINGERS_FOLD': 
        t, i, m, r, p = 'TUCKED', 'OPEN', 'OPEN', 'OPEN', 'OPEN'
        spread_val = -0.05 # Fingers squeezed together (B-Hand)
        
    elif gesture == 'TWO_FINGERS_FWD': t, i, m, r, p = 'CLOSED', 'FORWARD', 'FORWARD', 'CLOSED', 'CLOSED'
    elif gesture == 'TAP_TWICE': t, i, m, r, p = 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED', 'CLOSED'
    elif gesture == 'WAVE': t, i, m, r, p = 'OPEN', 'OPEN', 'OPEN', 'OPEN', 'OPEN'

    set_finger(lm, 0, t, spread=0.0)
    set_finger(lm, 1, i, spread=spread_val)
    set_finger(lm, 2, m, spread=spread_val)
    set_finger(lm, 3, r, spread=spread_val)
    set_finger(lm, 4, p, spread=spread_val)
    
    # --- Geometric Tweaks ---
    if gesture == 'THUMBS_UP': lm[4] = [0.3, -0.3, 0]
    if gesture == 'THUMBS_DOWN': lm[4] = [0.3, 0.3, 0]
    if gesture == 'POINT_SIDE':
        tmp = lm[:, 0].copy()
        lm[:, 0] = -lm[:, 1]
        lm[:, 1] = tmp
    if gesture == 'POINT_DOWN': lm[:, 1] = -lm[:, 1]
    if gesture == 'OK_SIGN':
        lm[4] = [0.1, -0.15, 0]
        lm[8] = [0.1, -0.15, 0]
    if gesture == 'FINGERS_CROSSED': lm[12][0] = lm[8][0] - 0.05
    if gesture == 'TAP_TWICE': lm[8] = [0.05, -0.1, -0.05]
    if gesture == 'WAVE':
        rot = random.uniform(20, 45) * random.choice([-1, 1])
        rotate_z(lm, rot)
        
    max_dist = np.max(np.linalg.norm(lm, axis=1))
    if max_dist > 0: lm = lm / max_dist
    noise = np.random.normal(0, 0.02, lm.shape)
    lm += noise
    
    return lm.flatten()

def main():
    print(f"Generating enhanced normalized dataset...")
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['label'] + [f'{c}{i}' for i in range(21) for c in ['x', 'y', 'z']]
        writer.writerow(header)
        for gesture in GESTURES:
            for _ in range(SAMPLES_PER_CLASS):
                row = [gesture] + generate_sample(gesture).tolist()
                writer.writerow(row)
    print(f"Done: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
