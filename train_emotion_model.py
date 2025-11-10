import os
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

# CONFIG
AUDIO_DIR = "audio_wav"
CSV_PATH = "annotations/static_annotations_averaged_songs_1_2000.csv"
OUTPUT_MODEL = "emotion_model_improved.onnx"
SR = 22050
N_MELS = 64
HOP_LENGTH = 512
DURATION = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
BATCH_SIZE = 16
LR = 3e-4
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 8


# Optional SpecAugment
def spec_augment(mel, time_mask_param=40, freq_mask_param=15):
    mel = mel.clone()
    t = mel.size(2)
    f = mel.size(1)

    # time mask
    if t > time_mask_param:
        t0 = np.random.randint(0, t - time_mask_param)
        t_len = np.random.randint(1, time_mask_param)
        mel[:, :, t0:t0+t_len] = 0

    # freq mask
    if f > freq_mask_param:
        f0 = np.random.randint(0, f - freq_mask_param)
        f_len = np.random.randint(1, freq_mask_param)
        mel[:, f0:f0+f_len, :] = 0

    return mel


# Dataset
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, audio_dir, augment=False):
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip()
        self.audio_dir = audio_dir
        self.augment = augment

        # normalize labels to 0–1
        self.val_min, self.val_max = self.df["valence_mean"].min(), self.df["valence_mean"].max()
        self.aro_min, self.aro_max = self.df["arousal_mean"].min(), self.df["arousal_mean"].max()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        song_id = str(int(row["song_id"]))
        file_path = os.path.join(self.audio_dir, f"{song_id}.wav")

        # load & preprocess audio
        y, sr = librosa.load(file_path, sr=SR)
        y = y[:SR * DURATION]
        if len(y) < SR * DURATION:
            y = np.pad(y, (0, SR * DURATION - len(y)))

        # mel-spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = torch.tensor(mel_db).float().unsqueeze(0)  # [1, n_mels, time]

        # normalize per-sample
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        # apply SpecAugment during training
        if self.augment:
            if np.random.rand() < 0.5:
                mel_db = spec_augment(mel_db)

        # normalize labels 0–1
        val = (row["valence_mean"] - self.val_min) / (self.val_max - self.val_min)
        aro = (row["arousal_mean"] - self.aro_min) / (self.aro_max - self.aro_min)
        label = torch.tensor([val, aro]).float()

        return mel_db, label


# Determine mel length dynamically
print("Pre-calculating spectrogram length...")
y_dummy = np.zeros(SR * DURATION)
mel_dummy = librosa.feature.melspectrogram(y=y_dummy, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
mel_db_length = mel_dummy.shape[1]
print(f"Mel spectrogram shape: {mel_dummy.shape}")


# Model
class ImprovedEmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


# Prepare data
dataset = EmotionDataset(CSV_PATH, AUDIO_DIR, augment=True)
train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.1, random_state=42)
train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE)

# Initialize model
model = ImprovedEmotionCNN().to(DEVICE)
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)

# Training Loop
train_losses, val_losses = [], []
train_maes, val_maes = [], []
best_val_loss = float("inf")
epochs_no_improve = 0

for epoch in range(EPOCHS):
    # Training
    model.train()
    total_train_loss, total_train_mae = 0, 0
    for mel, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        mel, label = mel.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        pred = model(mel)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        total_train_mae += torch.abs(pred - label).mean().item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_train_mae = total_train_mae / len(train_loader)

    # Validation
    model.eval()
    total_val_loss, total_val_mae = 0, 0
    with torch.no_grad():
        for mel, label in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            mel, label = mel.to(DEVICE), label.to(DEVICE)
            pred = model(mel)
            loss = criterion(pred, label)
            total_val_loss += loss.item()
            total_val_mae += torch.abs(pred - label).mean().item()

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_mae = total_val_mae / len(val_loader)

    scheduler.step(avg_val_loss)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_maes.append(avg_train_mae)
    val_maes.append(avg_val_mae)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
          f"Train MAE: {avg_train_mae:.4f}, Val MAE: {avg_val_mae:.4f}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_emotion_model.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered.")
            break


# Export to ONNX
model.eval()
dummy = torch.zeros(1, 1, N_MELS, mel_db_length)
torch.onnx.export(model.cpu(), dummy, OUTPUT_MODEL,
                  input_names=["mel"], output_names=["emotion"],
                  opset_version=13, dynamic_axes={"mel": {0: "batch"}})
print(f"Model exported to {OUTPUT_MODEL}")


# Plot Results
epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss", marker="o")
plt.plot(epochs, val_losses, label="Val Loss", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs, train_maes, label="Train MAE", marker="o")
plt.plot(epochs, val_maes, label="Val MAE", marker="o")
plt.xlabel("Epoch")
plt.ylabel("Mean Absolute Error")
plt.title("Training and Validation MAE")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
