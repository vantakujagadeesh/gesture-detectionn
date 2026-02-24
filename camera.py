import cv2
import numpy as np
import mediapipe as mp

class MediapipeHelper:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_hands = mp.solutions.hands 
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.3, 
            min_tracking_confidence=0.3
        )
        
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )

    def detect_landmarks(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        results_holistic = self.holistic.process(image_rgb)
        
        has_left = results_holistic.left_hand_landmarks is not None
        has_right = results_holistic.right_hand_landmarks is not None
        
        results_hands = None
        if not has_left and not has_right:
            results_hands = self.hands.process(image_rgb)
            
        image_rgb.flags.writeable = True
        return image, results_holistic, results_hands

    def draw_styled_landmarks(self, image, results_holistic, results_hands):
        # Custom Style for better visibility
        # Landmark: Red dots, large
        landmark_spec = self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=4)
        # Connection: White lines, thick
        connection_spec = self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=2)
        
        # Draw Holistic Hands
        if results_holistic.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, 
                results_holistic.left_hand_landmarks, 
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_spec, 
                connection_spec
            )
        if results_holistic.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                image, 
                results_holistic.right_hand_landmarks, 
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_spec, 
                connection_spec
            )
            
        # Draw Fallback Hands
        if results_hands and results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                 self.mp_drawing.draw_landmarks(
                    image, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_spec, 
                    connection_spec
                 )
        
        # Face Mesh (Subtle)
        if results_holistic.face_landmarks:
            self.mp_drawing.draw_landmarks(
                image, 
                results_holistic.face_landmarks, 
                self.mp_holistic.FACEMESH_CONTOURS, 
                self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
            )

    def extract_keypoints(self, results_holistic):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results_holistic.pose_landmarks.landmark]).flatten() if results_holistic.pose_landmarks else np.zeros(33*4)
        face = np.zeros(468*3) 
        if results_holistic.face_landmarks:
             face = np.array([[res.x, res.y, res.z] for res in results_holistic.face_landmarks.landmark]).flatten()
        
        lh = np.array([[res.x, res.y, res.z] for res in results_holistic.left_hand_landmarks.landmark]).flatten() if results_holistic.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results_holistic.right_hand_landmarks.landmark]).flatten() if results_holistic.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])
        
    def extract_hand_normalized(self, results_holistic, results_hands):
        hand_landmarks = None
        handedness = None
        source = "None"
        
        if results_holistic.right_hand_landmarks:
            hand_landmarks = results_holistic.right_hand_landmarks.landmark
            handedness = "Right"
            source = "Holistic"
        elif results_holistic.left_hand_landmarks:
            hand_landmarks = results_holistic.left_hand_landmarks.landmark
            handedness = "Left"
            source = "Holistic"
            
        if hand_landmarks is None and results_hands and results_hands.multi_hand_landmarks:
            hand_landmarks = results_hands.multi_hand_landmarks[0].landmark
            if results_hands.multi_handedness:
                label = results_hands.multi_handedness[0].classification[0].label
                handedness = label 
            else:
                handedness = "Right"
            source = "Hands_Fallback"
            
        if hand_landmarks:
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
            wrist = coords[0, :]
            coords = coords - wrist
            max_dist = np.max(np.linalg.norm(coords, axis=1))
            if max_dist > 0:
                coords = coords / max_dist
            return coords.flatten(), handedness, source
            
        return None, None, source

    def detect_facial_markers(self, results):
        markers = {
            "eyebrows_raised": False,
            "eyebrows_furrowed": False,
            "head_lean_forward": False
        }
        if not results.face_landmarks: return markers
        
        landmarks = results.face_landmarks.landmark
        r_eye_top = landmarks[159].y
        r_brow = landmarks[70].y
        r_dist = abs(r_eye_top - r_brow)
        l_eye_top = landmarks[386].y
        l_brow = landmarks[300].y
        l_dist = abs(l_eye_top - l_brow)
        avg_brow_dist = (r_dist + l_dist) / 2
        face_height = abs(landmarks[152].y - landmarks[10].y)
        
        if face_height > 0:
            ratio = avg_brow_dist / face_height
            if ratio > 0.065: markers["eyebrows_raised"] = True
            elif ratio < 0.035: markers["eyebrows_furrowed"] = True
        
        if results.pose_landmarks:
            nose_z = results.pose_landmarks.landmark[0].z
            l_shldr_z = results.pose_landmarks.landmark[11].z
            r_shldr_z = results.pose_landmarks.landmark[12].z
            avg_shldr_z = (l_shldr_z + r_shldr_z) / 2
            if nose_z < (avg_shldr_z - 0.1): markers["head_lean_forward"] = True
                 
        return markers
