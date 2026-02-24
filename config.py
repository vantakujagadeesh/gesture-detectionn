# Configuration Settings for Sign Language Recognition System

# System Settings
SYSTEM_CONFIG = {
    "LANGUAGE": "ASL",
    "INPUT_SOURCE": 0,
    "MODEL_TYPE": "transformer",
    "ENABLE_TTS": False,
    "OUTPUT_MODE": "natural",
    # Available modes: 'sign_language' (Dynamic), 'chat_gesture' (Static 20), 'alphabet_mode' (A-Z Speller)
    "OPERATION_MODE": "alphabet_mode" 
}

# Vocabulary Paths
VOCAB_FILES = {
    "ASL": "actions.txt",
    "BSL": "actions_bsl.txt"
}

# Model Weights Paths
MODEL_PATHS = {
    "lstm": "action.h5",
    "transformer": "action_transformer.h5",
    "static_gesture": "static_gesture_model.pkl"
}

# Chat Gesture Mappings (Static Gestures -> Text)
CHAT_GESTURE_MAP = {
    'ONE_FINGER': "How are you doing today?",
    'TWO_FINGERS': "I am doing great, thank you!",
    'OPEN_PALM': "Hello, nice to meet you!",
    'CLOSED_FIST': "Please wait for a moment",
    'THUMBS_UP': "Yes, that sounds good",
    'THUMBS_DOWN': "No, I don't think so",
    'FIVE_FINGERS': "Thank you very much for your help!",
    'THREE_FINGERS': "What are you doing right now?",
    'FOUR_FINGERS': "I need some assistance, please",
    'POINT_FORWARD': "Please come over here",
    'POINT_DOWN': "Stop what you are doing",
    'POINT_SIDE': "Where is the nearest exit?",
    'OK_SIGN': "Everything is okay",
    'FINGERS_CROSSED': "I am hoping for the best",
    'ROCK_SIGN': "This is an emergency, help!",
    'THREE_FINGERS_THUMB': "I am thirsty, I need water",
    'FOUR_FINGERS_FOLD': "I am sorry, I don't understand",
    'TWO_FINGERS_FWD': "Let's go forward",
    'TAP_TWICE': "Could you please explain that again?",
    'WAVE': "Goodbye, see you later!"
}
