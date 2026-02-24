from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization

def build_model(input_shape, output_shape):
    """
    Builds a robust LSTM architecture for sign language recognition.
    Designed to handle 2000+ classes with improved regularization.
    """
    model = Sequential()
    
    # Input Layer
    model.add(Input(shape=input_shape))
    
    # Bidirectional LSTM Layer 1
    # Bidirectional allows the model to learn from both past and future context in the sequence
    model.add(Bidirectional(LSTM(128, return_sequences=True, activation='relu')))
    model.add(BatchNormalization())
    model.add(Dropout(0.3)) # Prevent overfitting
    
    # LSTM Layer 2
    model.add(LSTM(256, return_sequences=True, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # LSTM Layer 3
    model.add(LSTM(128, return_sequences=False, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Dense Layers for Classification
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    
    # Output Layer
    model.add(Dense(output_shape, activation='softmax'))
    
    return model
