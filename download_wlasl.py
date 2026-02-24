import json
import os
import urllib.request
import cv2
import numpy as np
from camera import MediapipeHelper
import ssl

# Bypass SSL errors
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Constants
JSON_PATH = 'WLASL_v0.3.json'
RAW_VIDEO_DIR = 'raw_videos'
MODEL_DATA_DIR = 'model_data'
SEQUENCE_LENGTH = 30
MAX_VIDEOS_PER_CLASS = 10  # Limit to save disk space/time, increase for better accuracy

def download_video(url, output_path):
    try:
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def process_video(video_path, action, sequence_num):
    helper = MediapipeHelper()
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    with helper.holistic as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract landmarks
            image, results = helper.detect_landmarks(frame)
            keypoints = helper.extract_keypoints(results)
            frames.append(keypoints)
            
    cap.release()
    
    if len(frames) == 0:
        return False
        
    # Normalize to SEQUENCE_LENGTH
    frames = np.array(frames)
    if len(frames) < SEQUENCE_LENGTH:
        diff = SEQUENCE_LENGTH - len(frames)
        last_frame = frames[-1]
        padding = np.tile(last_frame, (diff, 1))
        frames = np.concatenate([frames, padding])
    elif len(frames) > SEQUENCE_LENGTH:
        indices = np.linspace(0, len(frames)-1, SEQUENCE_LENGTH).astype(int)
        frames = frames[indices]
        
    # Save
    save_dir = os.path.join(MODEL_DATA_DIR, action, str(sequence_num))
    os.makedirs(save_dir, exist_ok=True)
    
    for i, frame in enumerate(frames):
        np.save(os.path.join(save_dir, f"{i}.npy"), frame)
        
    return True

def main():
    if not os.path.exists(JSON_PATH):
        print(f"{JSON_PATH} not found.")
        return

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    os.makedirs(RAW_VIDEO_DIR, exist_ok=True)
    os.makedirs(MODEL_DATA_DIR, exist_ok=True)
    
    downloaded_actions = set()
    
    # Process ALL entries in the JSON
    print(f"Found {len(data)} total glosses in dataset.")
    
    for entry in data:
        gloss = entry['gloss']
        
        # Skip if we already have data for this gloss (optional, for resuming)
        if os.path.exists(os.path.join(MODEL_DATA_DIR, gloss)) and len(os.listdir(os.path.join(MODEL_DATA_DIR, gloss))) > 0:
            downloaded_actions.add(gloss)
            continue
            
        print(f"Processing {gloss}...")
        
        instances = entry['instances']
        success_count = 0
        
        for i, inst in enumerate(instances):
            url = inst['url']
            
            # Filter non-direct links (YouTube links require yt-dlp which is external)
            # For this script, we focus on direct downloads to maximize success without deps
            # If user has yt-dlp installed, they can uncomment the yt-dlp logic
            if 'youtube' in url or 'youtu.be' in url:
                continue
                
            video_filename = f"{gloss}_{inst['video_id']}.mp4"
            video_path = os.path.join(RAW_VIDEO_DIR, video_filename)
            
            # Download
            if not os.path.exists(video_path):
                if not download_video(url, video_path):
                    continue
            
            # Process
            if process_video(video_path, gloss, success_count):
                success_count += 1
                downloaded_actions.add(gloss)
                
            if success_count >= MAX_VIDEOS_PER_CLASS:
                break
        
        if success_count == 0:
            pass # print(f"No valid videos found for {gloss}")
            
    # Update actions.txt
    if downloaded_actions:
        with open('actions.txt', 'w') as f:
            for action in sorted(list(downloaded_actions)):
                f.write(action + '\n')
        print(f"Updated actions.txt with {len(downloaded_actions)} words.")
    else:
        print("No actions downloaded.")

if __name__ == "__main__":
    main()
