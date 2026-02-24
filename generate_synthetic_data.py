import numpy as np
import os

DATA_PATH = os.path.join('model_data') 
actions = ['hello', 'thanks', 'iloveyou', 'yes', 'no']
no_sequences = 30
sequence_length = 30

def create_folder_structure():
    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

def generate_landmarks(action):
    frames = []
    for i in range(sequence_length):
        # Base: All zeros
        # 1662 values: Pose(132) + LH(63) + RH(63) = 258 used in our model
        # We need to match the shape expected by model.py (which uses 258)
        # But wait, camera.py extracts:
        # pose (33*4) + lh (21*3) + rh (21*3)
        # = 132 + 63 + 63 = 258.
        
        data = np.zeros(258)
        
        # Simulate movement
        t = i / sequence_length
        
        # Right Hand (indices 132+63 to end) -> 195:258
        rh_start = 195
        
        if action == 'hello':
            # Hand moves from head (y small) to forward
            data[rh_start] = 0.5 + t * 0.1 # x
            data[rh_start+1] = 0.2 + t * 0.2 # y (down)
            
        elif action == 'thanks':
            # Hand moves from chin (y medium) down
            data[rh_start] = 0.5
            data[rh_start+1] = 0.4 + t * 0.3 # y (down)
            
        elif action == 'iloveyou':
            # Static
            data[rh_start] = 0.6
            data[rh_start+1] = 0.5
            
        elif action == 'yes':
            # Nodding (pose nose y)
            # Pose nose is index 0-3 (x,y,z,v)
            data[1] = 0.2 + np.sin(t * 10) * 0.05
            
        elif action == 'no':
            # Shaking (pose nose x)
            data[0] = 0.5 + np.sin(t * 10) * 0.05
            
        # Add some noise
        data += np.random.normal(0, 0.01, 258)
        frames.append(data)
        
    return frames

def main():
    create_folder_structure()
    
    print("Generating synthetic data...")
    for action in actions:
        print(f"Generating {action}...")
        for sequence in range(no_sequences):
            frames = generate_landmarks(action)
            for frame_num, frame in enumerate(frames):
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, frame)
                
    # Update actions.txt
    with open('actions.txt', 'w') as f:
        for action in actions:
            f.write(action + '\n')
    print("Synthetic data generated.")

if __name__ == "__main__":
    main()
