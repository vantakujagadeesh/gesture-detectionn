# Comprehensive Sign Language Recognition System

A production-ready real-time sign language recognition system built with TensorFlow, OpenCV, and MediaPipe. Designed to scale to 2000+ vocabulary words with high accuracy and low latency.

## Features

-   **Real-time Recognition**: Processes webcam feed at 30+ FPS.
-   **Advanced Model**: Bidirectional LSTM / Transformer architecture.
-   **Multi-Language Support**: Configurable for ASL (default) and extensible for BSL, etc.
-   **ASL Grammar Engine**: Translates raw glosses into grammatically correct English (e.g., "Your name what" -> "What is your name?").
-   **Text-to-Speech (TTS)**: Voice output for translated sentences.
-   **Flexible Input**: Supports Webcam, Video Files, and IP Camera Streams.
-   **Large Vocabulary Support**: Scalable architecture ready for WLASL 2000+ classes.
-   **Confidence Visualization**: Real-time confidence bars.
-   **Modern UI**: Responsive, dark-themed chat interface with video preview.

## Architecture

1.  **Input**: Webcam Video Stream (configurable).
2.  **Feature Extraction**: MediaPipe Holistic (Pose + Hands) -> 258 keypoints per frame.
3.  **Sequence Processing**: Rolling window of 30 frames.
4.  **Classification**: Bidirectional LSTM / Transformer Neural Network.
5.  **Grammar Processing**: `asl_grammar.py` applies linguistic rules based on selected language.
6.  **Output**: Real-time text overlay and TTS audio.

## Configuration (`config.py`)
You can configure the system settings in `config.py`:
-   `LANGUAGE`: "ASL" or "BSL"
-   `INPUT_SOURCE`: `0` (Webcam), `"video.mp4"`, or `"http://..."` (IP Stream)
-   `MODEL_TYPE`: "lstm" or "transformer"
-   `OUTPUT_MODE`: "natural" (Grammar corrected) or "raw" (Gloss)

## Setup & Installation

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Data (2000+ Signs)**:
    The system is configured for the WLASL 2000 dataset.
    
    -   **Step 1: Extract Vocabulary**:
        ```bash
        python extract_wlasl_vocab.py
        ```
        This populates `actions.txt` with all 2000 glosses.
        
    -   **Step 2: Download Videos**:
        ```bash
        pip install yt-dlp
        python download_wlasl.py
        ```
        *Warning: This will attempt to download thousands of videos. Ensure you have ~50GB space and good internet.*

    -   **Step 3: Collect Custom Data (Optional)**:
        If you want to add specific signs manually:
        ```bash
        python collect_data.py <word_name>
        ```

3.  **Train Model**:
    The training script automatically handles sparse data (i.e., you can train the 2000-class model even if you only have data for 5 signs).
    
    ```bash
    # Train Standard LSTM
    python train_model.py lstm
    
    # OR Train Advanced Transformer (Recommended for large vocab)
    python train_model.py transformer
    ```

4.  **Run System**:
    ```bash
    python app.py
    ```
    Open `http://127.0.0.1:5000/` in your browser.

## Supported ASL Grammar
The system now detects specific ASL sentence structures:
-   **Topic-Comment**: "You from where" -> "Where are you from?"
-   **Questions**: "Your name what" -> "What is your name?"
-   **Compound Signs**: "See" + "Later" -> "See you later"
-   **Icebreakers**: "You" + "Like" + "What do" -> "What do you like to do?"

## Testing & Validation

To rigorously validate the system's performance (Accuracy & Latency):

```bash
python test_accuracy.py
```

## Project Structure

-   `app.py`: Main Flask application with inference logic and UI.
-   `config.py`: System configuration.
-   `asl_grammar.py`: Linguistic rules engine.
-   `model.py`: LSTM model definition.
-   `model_transformer.py`: Transformer model definition.
-   `train_model.py`: Training pipeline.
-   `camera.py`: Computer Vision pipeline (MediaPipe).
-   `download_wlasl.py`: Dataset downloader.
-   `actions.txt`: Dynamic vocabulary source of truth.
