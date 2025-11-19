import onnxruntime as ort
import librosa
import numpy as np
import os

# Config - following train_dynamic.py rules
MODEL_PATH = "emotion_model_dynamic.onnx"
AUDIO_DIR = "test_wav"     # Folder containing r1.wav, r2.wav, ..., r24.wav
SR = 22050
N_MELS = 64
HOP_LENGTH = 512
SEGMENT_DURATION = 5  # 5-second audio segments (matches train_dynamic.py SEGMENT_DURATION)
NUM_FILES = 24

# Load model
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"Loaded ONNX model: {MODEL_PATH}")
print(f"Input name: {input_name}, Output name: {output_name}")
print(f"Segment duration: {SEGMENT_DURATION} seconds\n")

# Process each audio file
for i in range(1, NUM_FILES + 1):
    filename = f"r{i}.wav"
    path = os.path.join(AUDIO_DIR, filename)

    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue

    # Load audio
    y, sr = librosa.load(path, sr=SR)
    
    # Extract 5-second segment (following train_dynamic.py preprocessing)
    target_length = int(SEGMENT_DURATION * SR)
    
    if len(y) > target_length:
        # Take first 5 seconds
        y = y[:target_length]
    elif len(y) < target_length:
        # Pad if segment is too short (following train_dynamic.py line 167-169)
        y = np.pad(y, (0, target_length - len(y)))
    
    # Generate mel-spectrogram (following train_dynamic.py line 174-175)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Convert to tensor format [1, n_mels, time] then add batch and channel dims
    mel_db = mel_db[np.newaxis, :, :].astype(np.float32)  # [1, n_mels, time]
    
    # Normalize per-sample (following train_dynamic.py line 179)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    
    # Add channel dimension for CNN input: [1, 1, n_mels, time]
    mel_db = mel_db[np.newaxis, :, :, :].astype(np.float32)
    
    # Run inference
    outputs = session.run([output_name], {input_name: mel_db})
    emotion = outputs[0][0]  # [valence, arousal]
    
    valence = emotion[0]
    arousal = emotion[1]

    normalized_valence = (valence - 4.01) / 2.46
    normalized_arousal = (arousal - 3.58) / 3.30
    # Print results
    print(f"{filename}: Valence: {normalized_valence:.4f} | Arousal: {normalized_arousal:.4f}")

