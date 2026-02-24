# Technical Specification: Local Sign Language Translator (MVP)

## 1. System Overview
This project is a local, real-time Sign Language Translation application. It captures video via a webcam, extracts hand/pose landmarks using MediaPipe, and uses a lightweight LSTM/GRU neural network to classify sequences of landmarks into text (ASL glosses/words). The interface is a Flask-based web application.

## 2. Architecture

### Data Flow
1.  **Input**: Webcam video stream (30 FPS).
2.  **Preprocessing (MediaPipe)**: 
    -   Extract holistic landmarks (Left Hand, Right Hand, Pose).
    -   Flatten and normalize coordinates (x, y, z).
    -   Concatenate into a feature vector per frame.
3.  **Sequence Creation**: 
    -   Collect a rolling window of frames (e.g., 30 frames) to form a sequence.
4.  **Inference (Model)**: 
    -   Input: Sequence of shape `(30, N_FEATURES)`.
    -   Model: LSTM/GRU layers -> Dense Output Layer (Softmax).
    -   Output: Probability distribution over vocabulary (e.g., "Hello", "Thanks", "Yes").
5.  **Frontend (Flask)**: 
    -   Display video feed with landmark overlays.
    -   Display real-time translated text.

## 3. Tech Stack
-   **Language**: Python 3.x
-   **Computer Vision**: OpenCV (`cv2`), MediaPipe
-   **Machine Learning**: TensorFlow/Keras (for LSTM model)
-   **Web Framework**: Flask
-   **Frontend**: HTML, JavaScript (for polling/updating translation)

## 4. Data Structure
-   **Landmarks**: MediaPipe Holistic returns ~1662 values per frame (Pose: 33*4, Face: 468*3, Hands: 21*3*2).
-   *Optimization*: For an MVP, we will focus on **Pose (33 points)** and **Hands (21 points each)** to reduce dimensionality. 
    -   Pose: 33 * 4 (x,y,z,visibility) = 132
    -   Left Hand: 21 * 3 = 63
    -   Right Hand: 21 * 3 = 63
    -   Total Feature Vector Size: 258 (approx).
-   **Training Data**: Saved as NumPy arrays (`.npy`) in `model_data/`.
    -   Structure: `Sequence_Name/Sequence_Number.npy`

## 5. Feasibility & Constraints
-   **Latency**: MediaPipe is optimized for CPU. LSTM inference is lightweight. Real-time performance on a laptop is feasible.
-   **Accuracy**: Highly dependent on the quality and quantity of user-recorded training data.
-   **Scope**: Limited vocabulary (Start with 3-5 signs like "Hello", "Thank you", "I love you").
