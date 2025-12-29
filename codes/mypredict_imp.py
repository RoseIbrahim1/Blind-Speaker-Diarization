import numpy as np
import soundfile as sf
import os
import librosa
import time
from tensorflow.keras.models import load_model

# ===== Audio processing parameters =====
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64
DURATION = 10  # in seconds
FRAME_LENGTH = SAMPLE_RATE * DURATION

def load_audio(path):
    """
    Load an audio file, convert to mono if needed, resample to SAMPLE_RATE,
    and ensure it has exactly FRAME_LENGTH samples (padding or truncating as necessary).
    """
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)  # convert to mono
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    if len(audio) < FRAME_LENGTH:
        audio = np.pad(audio, (0, FRAME_LENGTH - len(audio)), 'constant')
    else:
        audio = audio[:FRAME_LENGTH]
    return audio.astype('float32')

def extract_mel(audio):
    """
    Extract Mel-spectrogram features from the audio signal and convert to dB scale.
    Output shape: (time_steps, N_MELS)
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmax=8000
    )
    mel_db = librosa.power_to_db(mel_spec).T
    return mel_db.astype('float32')

def count(audio, model):
    """
    Predict the speaker count from an audio array using a loaded model.
    Returns an integer count (1-5).
    """
    mel = extract_mel(audio)
    X = mel[np.newaxis, ..., np.newaxis]  # add batch and channel dims
    preds = model.predict(X, verbose=0)
    count_pred = np.argmax(preds, axis=1)[0] + 1  # +1 because labels are 1–5
    return count_pred

def predict_speaker_count(audio_path, model_path='mymodel/speaker_model_fixed.h5'):
    """
    Load a model and predict the number of speakers in the given audio file.
    """
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found!")
    
    model = load_model(model_path)
    audio = load_audio(audio_path)
    return count(audio, model)


# ===== Run as standalone script =====
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Predict speaker count from audio')
    parser.add_argument('audio', help='Path to 16kHz audio file')
    parser.add_argument('--model', default='mymodel/speaker_model_fixed.h5', help='Path to model file (.h5)')
    args = parser.parse_args()

    estimate = predict_speaker_count(args.audio, args.model)
    print("Speaker Count Estimate:", estimate)
