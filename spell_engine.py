import difflib
import time

class SpellEngine:
    def __init__(self, vocab_file='actions.txt'):
        self.buffer = []
        self.last_char = None
        self.last_time = 0
        self.sentence = []
        self.debounce_time = 0.5 # Seconds to hold a char to accept it (reduced from 1.0)
        
        # Load dictionary
        try:
            with open(vocab_file, 'r') as f:
                self.dictionary = [line.strip().lower() for line in f.readlines()]
        except FileNotFoundError:
            self.dictionary = ["hello", "world", "thank", "you", "please"] # Fallback

    def process_gesture(self, gesture):
        """
        Processes a continuous stream of gestures.
        Returns: (current_buffer_str, confirmed_sentence_str)
        """
        current_time = time.time()
        
        # Logic: 
        # If gesture is a character (A-Z) and held for 'debounce_time', append to buffer.
        # If gesture is SPACE or OPEN_PALM, commit buffer as word.
        # If gesture is FIST or DELETE, remove last char from buffer.
        
        if gesture == self.last_char:
            if current_time - self.last_time > self.debounce_time:
                # Character held long enough
                self._handle_input(gesture)
                self.last_time = current_time # Reset timer to prevent rapid fire
        else:
            self.last_char = gesture
            self.last_time = current_time
            
        return "".join(self.buffer), " ".join(self.sentence)

    def _handle_input(self, gesture):
        if gesture == 'SPACE':
            self._commit_word()
        elif gesture == 'DELETE':
            if self.buffer:
                self.buffer.pop()
            elif self.sentence:
                self.sentence.pop()
        elif len(gesture) == 1 and gesture.isalpha():
            self.buffer.append(gesture)
        
        elif gesture in ['OPEN_PALM', 'THUMBS_UP', 'OK']:
            self._commit_word()
            
        elif gesture in ['FIST', 'THUMBS_DOWN']: # Additional delete gestures
            if self.buffer:
                self.buffer.pop()
            elif self.sentence:
                self.sentence.pop()

    def _commit_word(self):
        if not self.buffer:
            return
            
        raw_word = "".join(self.buffer).lower()
        
        # Spell Check / Autocorrect
        matches = difflib.get_close_matches(raw_word, self.dictionary, n=1, cutoff=0.6)
        
        if matches:
            corrected_word = matches[0]
        else:
            corrected_word = raw_word
            
        # Capitalize first letter
        self.sentence.append(corrected_word.capitalize())
        self.buffer = [] # Clear buffer

    def get_suggestions(self):
        if not self.buffer:
            return []
        raw_word = "".join(self.buffer).lower()
        return difflib.get_close_matches(raw_word, self.dictionary, n=3)
