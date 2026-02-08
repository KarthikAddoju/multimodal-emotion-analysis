import cv2
import numpy as np

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

class FaceEmotionModel:
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def predict(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return "neutral", 0.3, np.zeros(len(EMOTIONS))

        # Placeholder logic (replace with CNN inference)
        emotion_idx = np.random.randint(0, len(EMOTIONS))
        confidence = round(np.random.uniform(0.6, 0.9), 2)

        probs = np.zeros(len(EMOTIONS))
        probs[emotion_idx] = confidence

        return EMOTIONS[emotion_idx], confidence, probs
