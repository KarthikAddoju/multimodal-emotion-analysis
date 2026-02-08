import sounddevice as sd
import scipy.io.wavfile as wav

def record_audio(filename="temp.wav", duration=4, fs=16000):
    print("Recording audio...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wav.write(filename, fs, recording)
    print("Audio recorded")
