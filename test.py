import onnxruntime as ort
import librosa
import numpy as np
import os

# Config
MODEL_PATH = "emotion_model.onnx"
AUDIO_DIR = "test_wav"     # Folder containing b1.wav, b2.wav, ..., b20.wav
SR = 22050
N_MELS = 64
HOP_LENGTH = 512
DURATION = 30                # Each audio except the last is 5 seconds
NUM_FILES = 24

# Load model
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"Loaded ONNX model: {MODEL_PATH}")
print(f"Input name: {input_name}, Output name: {output_name}\n")

# Process each audio file
for i in range(1, NUM_FILES + 1):
    filename = f"r{i}.wav"
    path = os.path.join(AUDIO_DIR, filename)

    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue

    # Load and preprocess audio
    y, sr = librosa.load(path, sr=SR)

    # Handle different durations
    max_len = SR * DURATION
    if len(y) > max_len:
        y = y[:max_len]
    elif len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))

    # Convert to mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db[np.newaxis, np.newaxis, :, :].astype(np.float32)  # [1, 1, n_mels, time]

    # Run inference
    outputs = session.run([output_name], {input_name: mel_db})
    emotion = outputs[0][0]  # [valence, arousal]

    # Print results
    print(f"Valence: {emotion[0]:.4f} | Arousal: {emotion[1]:.4f}")

