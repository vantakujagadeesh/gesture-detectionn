import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from model import build_model
from model_transformer import build_transformer_model
import sys

# Path for exported data
DATA_PATH = os.path.join('model_data') 

# Load actions
try:
    with open('actions.txt', 'r') as f:
        all_actions = np.array([line.strip() for line in f.readlines()])
except FileNotFoundError:
    print("actions.txt not found.")
    exit()

no_sequences = 30
sequence_length = 30

def load_data():
    sequences, labels = [], []
    valid_actions = []
    
    print("Checking for collected data...")
    for action in all_actions:
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path):
            continue
            
        # Check if it has enough data
        available_sequences = os.listdir(action_path)
        if len(available_sequences) == 0:
            continue
            
        print(f"Loading data for: {action}")
        valid_actions.append(action)
        
        for sequence in range(no_sequences):
            # Check if sequence folder exists
            seq_path = os.path.join(action_path, str(sequence))
            if not os.path.exists(seq_path):
                continue
                
            window = []
            try:
                for frame_num in range(sequence_length):
                    res = np.load(os.path.join(seq_path, "{}.npy".format(frame_num)))
                    window.append(res)
                sequences.append(window)
                labels.append(all_actions.tolist().index(action))
            except Exception as e:
                pass
                
    return np.array(sequences), np.array(labels), valid_actions

def train(model_type='lstm'):
    X, y, valid_actions = load_data()
    
    if len(valid_actions) == 0:
        print("No data found! Please run collect_data.py or download_wlasl.py first.")
        return

    print(f"Training on {len(valid_actions)} actions.")
    print(f"Total sequences: {len(X)}")

    # One-hot encode using the FULL vocabulary size
    y = to_categorical(y, num_classes=len(all_actions)).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Save test data for validation script
    print("Saving test data for validation...")
    np.save('X_test.npy', X_test)
    np.save('y_test.npy', y_test)
    
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    es_callback = EarlyStopping(patience=30, monitor='categorical_accuracy', restore_best_weights=True)
    
    model_filename = 'action.h5' if model_type == 'lstm' else 'action_transformer.h5'
    mc_callback = ModelCheckpoint(model_filename, monitor='categorical_accuracy', save_best_only=True)
    
    if model_type == 'transformer':
        print("Building Transformer Model...")
        model = build_transformer_model((30, 258), len(all_actions))
    else:
        print("Building LSTM Model...")
        model = build_model((30, 258), len(all_actions))
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    model.fit(X_train, y_train, epochs=2000, batch_size=32, callbacks=[tb_callback, es_callback, mc_callback], validation_data=(X_test, y_test))
    
    model.summary()
    print(f"Model saved as {model_filename}")

if __name__ == "__main__":
    model_type = 'lstm'
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    
    print(f"Selected model type: {model_type}")
    train(model_type)
