class GrammarProcessor:
    def __init__(self, language="ASL"):
        self.language = language
        
        # ASL Vocabulary Map (Advanced)
        self.asl_map = {
            "take_care": "Take care",
            "name": "Name",
            "nice_meet": "Nice to meet you",
            "bathroom": "Bathroom",
            "favorite": "Favorite",
            "please": "Please",
            "thank_you": "Thank you",
            "meaning": "What does this mean?",
            "excuse_me": "Excuse me",
            "fine": "I'm fine",
            "learn_sign": "I am learning sign language",
            "like": "Like"
        }
        
        self.bsl_map = {
            "thank_you": "Cheers/Thanks",
            "bathroom": "Toilet"
        }

    def process_sequence(self, sign_sequence, facial_markers=None):
        if self.language == "ASL":
            return self._process_asl(sign_sequence, facial_markers)
        elif self.language == "BSL":
            return self._process_bsl(sign_sequence)
        else:
            return " ".join(sign_sequence)

    def _process_asl(self, sign_sequence, facial_markers):
        if not sign_sequence:
            return ""
            
        markers = facial_markers if facial_markers else {}
        is_question_face = markers.get("eyebrows_furrowed", False) or markers.get("head_lean_forward", False)
        is_raised_brows = markers.get("eyebrows_raised", False)

        # 1. Complex Movement / Phrase Logic
        # "Like" + "What_Do" -> "What do you like to do?"
        if len(sign_sequence) >= 2:
            if sign_sequence[-2] == "like" and sign_sequence[-1] == "what_do":
                return "What do you like to do?"
            if sign_sequence[-2] == "see" and sign_sequence[-1] == "later":
                return "See you later"

        # 2. Topic-Comment / Question Structure (Reversal Rule)
        # Rule: [Object/Topic] + [QuestionWord] -> [QuestionWord] + [Object/Topic]
        question_words = ["what", "where", "who", "how", "why", "when"]
        last_word = sign_sequence[-1].lower()
        
        if last_word in question_words:
            # "Bathroom Where" -> "Where is the bathroom?"
            if len(sign_sequence) >= 2:
                topic = sign_sequence[-2]
                return f"{last_word.capitalize()} is the {topic}?"
                
            # "Your Name What" -> "What is your name?"
            if len(sign_sequence) >= 3 and sign_sequence[-2] == "name" and sign_sequence[-3] == "your":
                 return "What is your name?"
            
            # "You From Where" -> "Where are you from?"
            if len(sign_sequence) >= 3 and sign_sequence[-2] == "from" and sign_sequence[-3] == "you":
                return "Where are you from?"
                
            # "Car Sign How" -> "How do you sign 'car'?"
            if len(sign_sequence) >= 3 and sign_sequence[-2] == "sign":
                word = sign_sequence[-3]
                return f"How do you sign '{word}'?"

        # 3. Metalinguistic Functions
        # [Sign] + [Meaning] -> "What does that sign mean?"
        if len(sign_sequence) >= 2 and last_word == "meaning":
             return "What does that sign mean?"

        # 4. Contextual Facial Markers
        # If leaning forward/furrowed brows, force question mark
        sentence = " ".join(sign_sequence)
        
        # Direct Mapping Check first
        if len(sign_sequence) == 1:
            raw = sign_sequence[0]
            if raw in self.asl_map:
                sentence = self.asl_map[raw]
        
        if is_question_face and not sentence.endswith("?"):
            sentence += "?"
            
        return sentence

    def _process_bsl(self, sign_sequence):
        if not sign_sequence: return ""
        if len(sign_sequence) == 1 and sign_sequence[0] in self.bsl_map:
            return self.bsl_map[sign_sequence[0]]
        return " ".join(sign_sequence) + " (BSL)"
