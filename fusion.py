import numpy as np

def late_fusion(audio_pred, video_pred, audio_conf, video_conf):
    weights = np.array([audio_conf, video_conf])
    weights = weights / weights.sum()

    final_probs = weights[0] * audio_pred + weights[1] * video_pred
    final_emotion_idx = np.argmax(final_probs)

    return final_emotion_idx, final_probs
