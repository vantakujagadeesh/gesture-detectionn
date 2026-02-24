import cv2
import numpy as np
import os
import time
import sys
from camera import MediapipeHelper

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('model_data') 

# Load actions from file
try:
    with open('actions.txt', 'r') as f:
        actions = np.array([line.strip() for line in f.readlines()])
except FileNotFoundError:
    print("actions.txt not found. Please run generate_vocab.py first.")
    exit()

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

def collect_data_for_action(action_name):
    if action_name not in actions:
        print(f"Error: '{action_name}' is not in actions.txt")
        return

    # Create folder for this action
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action_name, str(sequence)))
        except:
            pass

    helper = MediapipeHelper()
    cap = cv2.VideoCapture(0)
    
    # Set mediapipe model 
    with helper.holistic as holistic:
        
        # Loop through sequences aka videos
        for sequence in range(no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()
                if not ret:
                    print("Camera not detected.")
                    break

                # Make detections
                image, results = helper.detect_landmarks(frame)
                
                # Draw landmarks
                helper.draw_styled_landmarks(image, results)
                
                # Apply wait logic
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action_name, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action_name, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # Export keypoints
                keypoints = helper.extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action_name, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python collect_data.py <action_name>")
        print("Example: python collect_data.py hello")
        print(f"Available actions (first 10 of {len(actions)}): {actions[:10]}")
    else:
        collect_data_for_action(sys.argv[1])
