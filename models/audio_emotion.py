import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

MODEL_NAME = "superb/wav2vec2-base-superb-er"

class AudioEmotionModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()
        self.labels = self.model.config.id2label

    def predict(self, audio_path):
        speech, sr = librosa.load(audio_path, sr=16000)
        inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        emotion = self.labels[np.argmax(probs)]
        confidence = np.max(probs)

        return emotion, confidence, probs
