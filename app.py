from models.audio_emotion import AudioEmotionModel
from models.face_emotion import FaceEmotionModel
from utils.audio_utils import record_audio
from utils.video_utils import get_video_frame
from utils.fusion import late_fusion

import numpy as np

audio_model = AudioEmotionModel()
face_model = FaceEmotionModel()

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def run_multimodal_inference():
    record_audio("temp.wav")

    audio_emotion, audio_conf, audio_probs = audio_model.predict("temp.wav")
    print(f"Audio Emotion: {audio_emotion} ({audio_conf:.2f})")

    frame = get_video_frame()
    if frame is not None:
        face_emotion, face_conf, face_probs = face_model.predict(frame)
        print(f"Face Emotion: {face_emotion} ({face_conf:.2f})")
    else:
        face_conf = 0.1
        face_probs = np.zeros(len(EMOTIONS))

    idx, fused_probs = late_fusion(audio_probs, face_probs, audio_conf, face_conf)
    print(f"\nFinal Emotion (Fused): {EMOTIONS[idx]}")

if __name__ == "__main__":
    run_multimodal_inference()
