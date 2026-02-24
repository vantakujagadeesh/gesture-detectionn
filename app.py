from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import os
import pickle
from camera import MediapipeHelper
from model import build_model
from model_transformer import build_transformer_model
from asl_grammar import GrammarProcessor
from spell_engine import SpellEngine
import tensorflow as tf
import pyttsx3
import threading
import sys
import config
import traceback
from collections import deque, Counter

app = Flask(__name__)

# --- Load Configuration ---
LANGUAGE = config.SYSTEM_CONFIG["LANGUAGE"]
INPUT_SOURCE = config.SYSTEM_CONFIG["INPUT_SOURCE"]
MODEL_TYPE = config.SYSTEM_CONFIG["MODEL_TYPE"]
OUTPUT_MODE = config.SYSTEM_CONFIG["OUTPUT_MODE"]
OPERATION_MODE = config.SYSTEM_CONFIG["OPERATION_MODE"]

# Initialize Processors
grammar_processor = GrammarProcessor(language=LANGUAGE)
spell_engine = SpellEngine(vocab_file='actions.txt')

# Initialize TTS Engine
engine = None
try:
    engine = pyttsx3.init()
except Exception as e:
    print(f"Warning: TTS Engine failed to initialize: {e}")

def speak_text(text):
    try:
        local_engine = pyttsx3.init()
        local_engine.say(text)
        local_engine.runAndWait()
    except Exception as e:
        print(f"FALLBACK TTS: {text}") 

# --- LOAD MODELS ---

# 1. Dynamic Sign Language Models
vocab_file = config.VOCAB_FILES.get(LANGUAGE, "actions.txt")
try:
    with open(vocab_file, 'r') as f:
        actions = np.array([line.strip() for line in f.readlines()])
except FileNotFoundError:
    actions = np.array([])

if MODEL_TYPE == 'transformer':
    sign_model = build_transformer_model((30, 258), len(actions))
    weights_path = config.MODEL_PATHS["transformer"]
else:
    sign_model = build_model((30, 258), len(actions))
    weights_path = config.MODEL_PATHS["lstm"]

try:
    sign_model.load_weights(weights_path)
    print(f"Sign Model ({MODEL_TYPE}) loaded.")
except:
    print("Sign Model weights not found.")

# 2. Static Chat Gesture Model
static_model = None
try:
    with open(config.MODEL_PATHS["static_gesture"], 'rb') as f:
        static_model = pickle.load(f)
    print("Static Chat Gesture Model loaded.")
except:
    print("Static Gesture Model not found.")

# 3. Alphabet Model (New)
alphabet_model = None
try:
    with open("alphabet_model.pkl", 'rb') as f:
        alphabet_model = pickle.load(f)
    print("Alphabet Model loaded.")
except:
    print("Alphabet Model not found.")


# Global variables
sequence = []
sentence = []
current_prediction = "Waiting..."
current_confidence = 0.0
threshold = 0.7
tts_enabled = config.SYSTEM_CONFIG["ENABLE_TTS"]
last_processed_sentence = ""
chat_messages = [] 

# Smoothing Buffer for Static Gestures
prediction_buffer = deque(maxlen=8)

# Track if hand is present to prevent phantom predictions
hand_present = False
# Cooldown for no-hand state
no_hand_frames = 0
NO_HAND_THRESHOLD = 5  # frames without hand before clearing prediction

def generate_frames():
    global sequence, sentence, current_prediction, current_confidence, last_processed_sentence, OUTPUT_MODE, OPERATION_MODE, chat_messages, prediction_buffer, hand_present, no_hand_frames
    
    source = INPUT_SOURCE
    if isinstance(source, str) and source.isdigit():
        source = int(source)
        
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return
        
    # Set higher resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    helper = MediapipeHelper()

    with helper.holistic as holistic:
        while True:
            try:
                success, frame = cap.read()
                if not success:
                    if isinstance(source, str) and os.path.exists(source):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break
                
                # 1. Detect using NEW Helper logic (Holistic + Hands Fallback)
                image, results_holistic, results_hands = helper.detect_landmarks(frame)
                
                # 2. Draw styled landmarks (handles both types)
                helper.draw_styled_landmarks(image, results_holistic, results_hands)
                
                # 3. Flip for user view
                image = cv2.flip(image, 1)
                
                # --- LOGIC BRANCHING ---
                
                if OPERATION_MODE == 'sign_language':
                    try:
                        # BUG FIX: Check if any hand is present before processing
                        has_any_hand = (
                            results_holistic.left_hand_landmarks is not None or
                            results_holistic.right_hand_landmarks is not None or
                            (results_hands and results_hands.multi_hand_landmarks)
                        )
                        
                        if not has_any_hand:
                            no_hand_frames += 1
                            if no_hand_frames > NO_HAND_THRESHOLD:
                                current_prediction = "No hand detected"
                                current_confidence = 0.0
                                sequence = []  # Clear stale sequence
                        else:
                            no_hand_frames = 0
                            facial_markers = helper.detect_facial_markers(results_holistic)
                            keypoints = helper.extract_keypoints(results_holistic)
                            sequence.append(keypoints)
                            sequence = sequence[-30:]
                            
                            if len(sequence) == 30:
                                res = sign_model.predict(np.expand_dims(np.array(sequence), axis=0), verbose=0)[0]
                                best_idx = np.argmax(res)
                                confidence = res[best_idx]
                                
                                current_prediction = actions[best_idx]
                                current_confidence = float(confidence)
                                
                                if confidence > threshold: 
                                    if len(sentence) > 0: 
                                        if actions[best_idx] != sentence[-1]:
                                            sentence.append(actions[best_idx])
                                    else:
                                        sentence.append(actions[best_idx])
                                else:
                                    current_prediction = "..."

                                if len(sentence) > 5: 
                                    sentence = sentence[-5:]
                                    
                                processed_text = grammar_processor.process_sequence(sentence, facial_markers=facial_markers)
                                if processed_text and processed_text != last_processed_sentence:
                                    last_processed_sentence = processed_text
                                    if tts_enabled:
                                        threading.Thread(target=speak_text, args=(processed_text,)).start()
                    except Exception:
                        pass
                    
                    display_text = last_processed_sentence if OUTPUT_MODE == 'natural' else " ".join(sentence)
                    cv2.putText(image, display_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                elif OPERATION_MODE == 'chat_gesture':
                    # Custom 20 Gestures
                    try:
                        hand_data, handedness, source_type = helper.extract_hand_normalized(results_holistic, results_hands)
                        
                        debug_text = f"Source: {source_type} | Hand: {handedness}"
                        cv2.putText(image, debug_text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                        # BUG FIX: Only predict when hand is actually present
                        if hand_data is not None and static_model:
                            no_hand_frames = 0
                            pred = static_model.predict([hand_data])[0]
                            probs = static_model.predict_proba([hand_data])[0]
                            confidence = np.max(probs)
                            
                            if confidence > 0.3:
                                prediction_buffer.append(pred)
                            else:
                                if confidence > 0.2:
                                    prediction_buffer.append(pred)
                                else:
                                    prediction_buffer.append(None)
                            
                            cv2.putText(image, f"Raw: {pred} ({confidence:.2f})", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                            
                            if len(prediction_buffer) >= 5:
                                valid_preds = [p for p in prediction_buffer if p is not None]
                                if valid_preds:
                                    most_common = Counter(valid_preds).most_common(1)[0]
                                    gesture_name, count = most_common
                                    
                                    if count >= 3:
                                        current_confidence = float(confidence)
                                        mapped_text = config.CHAT_GESTURE_MAP.get(gesture_name, gesture_name)
                                        current_prediction = mapped_text
                                        last_processed_sentence = mapped_text
                                    else:
                                        current_prediction = "..."
                                else:
                                    current_prediction = "..."
                        else:
                             # BUG FIX: No hand detected — clear predictions after threshold
                             no_hand_frames += 1
                             prediction_buffer.append(None)
                             if no_hand_frames > NO_HAND_THRESHOLD:
                                 current_prediction = "No hand detected"
                                 current_confidence = 0.0
                                 prediction_buffer.clear()

                    except Exception as e:
                        pass
                    
                    cv2.rectangle(image, (0,0), (640, 60), (30,30,30), -1)
                    cv2.putText(image, f"Meaning: {current_prediction}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                elif OPERATION_MODE == 'alphabet_mode':
                    # A-Z Fingerspelling
                    try:
                        hand_data, handedness, source_type = helper.extract_hand_normalized(results_holistic, results_hands)
                        
                        if hand_data is not None and alphabet_model:
                            probs = alphabet_model.predict_proba([hand_data])[0]
                            confidence = np.max(probs)
                            pred = alphabet_model.classes_[np.argmax(probs)]
                            
                            # Reduced threshold to 0.4 to be more permissive
                            if confidence > 0.4:
                                buffer_str, full_sentence = spell_engine.process_gesture(pred)
                                current_prediction = pred
                                current_confidence = float(confidence)
                                print(f"DEBUG: Pred={pred}, Conf={confidence:.2f}") # Log for developer
                                
                                # UI for Spelling
                                cv2.putText(image, f"Letter: {pred}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                                
                                # Show constructed word buffer
                                cv2.putText(image, f"Word: {buffer_str}", (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                
                                # Show full sentence
                                last_processed_sentence = full_sentence

                                # Add TTS for confirmed words
                                if tts_enabled and pred == 'SPACE':
                                     words = full_sentence.split()
                                     if words:
                                         threading.Thread(target=speak_text, args=(words[-1],)).start()
                            else:
                                current_prediction = "..."
                                current_confidence = float(confidence)
                        else:
                            current_prediction = "Waiting..."
                            current_confidence = 0.0

                    except Exception as e:
                        pass
                    
                    # Top Bar for Sentence
                    cv2.rectangle(image, (0,0), (640, 60), (30,30,30), -1)
                    cv2.putText(image, f"Sent: {last_processed_sentence}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


                bar_width = int(current_confidence * 100)
                color = (0, 255, 0) if current_confidence > threshold else (0, 0, 255)
                cv2.rectangle(image, (520, 10), (520 + bar_width, 30), color, -1)
                cv2.rectangle(image, (520, 10), (620, 30), (255, 255, 255), 1)
                
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            except Exception as e:
                continue

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    global last_processed_sentence, current_prediction
    return jsonify({
        'prediction': current_prediction,
        'confidence': f"{current_confidence:.2f}",
        'sentence': last_processed_sentence,
        'mode': OPERATION_MODE,
        'buffer': "".join(spell_engine.buffer) if OPERATION_MODE == 'alphabet_mode' else ""
    })

@app.route('/toggle_tts', methods=['POST'])
def toggle_tts():
    global tts_enabled
    tts_enabled = not tts_enabled
    return jsonify({'status': 'success', 'tts_enabled': tts_enabled})

@app.route('/update_settings', methods=['POST'])
def update_settings():
    global OUTPUT_MODE, LANGUAGE, grammar_processor, OPERATION_MODE
    data = request.json
    if 'output_mode' in data: OUTPUT_MODE = data['output_mode']
    if 'language' in data: LANGUAGE = data['language']
    if 'operation_mode' in data: OPERATION_MODE = data['operation_mode']
    return jsonify({'status': 'success'})

@app.route('/send_chat', methods=['POST'])
def send_chat():
    global chat_messages
    data = request.json
    msg = data.get('message', '').strip()
    if msg:
        chat_messages.append({'sender': 'User', 'text': msg})
        
        # Simple AI Response logic
        ai_reply = f"I see you sent: '{msg}'. How can I help you with Sign Language today?"
        if "hello" in msg.lower():
            ai_reply = "Hello! I am your SignAI assistant. You can use gestures or type to communicate."
        elif "thank" in msg.lower():
            ai_reply = "You're welcome! Happy to help."
            
        chat_messages.append({'sender': 'AI', 'text': ai_reply})
        
    return jsonify({'status': 'success', 'history': chat_messages})

@app.route('/get_chat_history')
def get_chat_history():
    return jsonify(chat_messages)

@app.route('/clear_text', methods=['POST'])
def clear_text():
    global last_processed_sentence, current_prediction, current_confidence, sentence, sequence
    spell_engine.buffer = []
    spell_engine.sentence = []
    spell_engine.last_char = None
    last_processed_sentence = ""
    current_prediction = "Waiting..."
    current_confidence = 0.0
    sentence = []
    sequence = []
    return jsonify({'status': 'success'})

@app.route('/speak_text_api', methods=['POST'])
def speak_text_api():
    """Trigger server-side TTS if browser TTS is unavailable"""
    data = request.json
    text = data.get('text', '')
    if text:
        threading.Thread(target=speak_text, args=(text,)).start()
    return jsonify({'status': 'success'})

if __name__ == "__main__":
    app.run(debug=True)
