import os
import warnings

import librosa
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for safe training runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
AUDIO_DIR = "audio_wav"
CSV_PATH = "annotations/static_annotations_averaged_songs_1_2000.csv"
OUTPUT_MODEL = "emotion_model_improved.onnx"
BEST_MODEL_PATH = "best_emotion_model.pt"
SR = 22050
N_MELS = 64
HOP_LENGTH = 512
DURATION = 30  # seconds
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 25
BATCH_SIZE = 16
LR = 5e-4
WEIGHT_DECAY = 1e-5
ACC_THRESHOLD = 1.0  # accuracy threshold in 1–9 label space

print(f"Using device: {DEVICE}")


# -----------------------------------------------------------------------------
# SpecAugment
# -----------------------------------------------------------------------------
def spec_augment(mel, time_mask_param=40, freq_mask_param=15):
    mel = mel.clone()
    t = mel.size(2)
    f = mel.size(1)

    if t > time_mask_param:
        t0 = np.random.randint(0, t - time_mask_param)
        t_len = np.random.randint(1, time_mask_param)
        mel[:, :, t0:t0 + t_len] = 0

    if f > freq_mask_param:
        f0 = np.random.randint(0, f - freq_mask_param)
        f_len = np.random.randint(1, freq_mask_param)
        mel[:, f0:f0 + f_len, :] = 0

    return mel


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class EmotionDataset(Dataset):
    def __init__(self, csv_path, audio_dir, augment=False):
        self.df = pd.read_csv(csv_path)
        self.df.columns = self.df.columns.str.strip()
        self.audio_dir = audio_dir
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        song_id = str(int(row["song_id"]))
        file_path = os.path.join(self.audio_dir, f"{song_id}.wav")

        y, sr = librosa.load(file_path, sr=SR)
        y = y[:SR * DURATION]
        if len(y) < SR * DURATION:
            y = np.pad(y, (0, SR * DURATION - len(y)))

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = torch.tensor(mel_db).float().unsqueeze(0)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        if self.augment and np.random.rand() < 0.5:
            mel_db = spec_augment(mel_db)

        # Use original 1–9 label scale directly
        val = float(row["valence_mean"])
        aro = float(row["arousal_mean"])
        label = torch.tensor([val, aro]).float()

        return mel_db, label


# -----------------------------------------------------------------------------
# Determine mel length dynamically
# -----------------------------------------------------------------------------
print("Pre-calculating spectrogram length...")
y_dummy = np.zeros(SR * DURATION)
mel_dummy = librosa.feature.melspectrogram(y=y_dummy, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
mel_db_length = mel_dummy.shape[1]
print(f"Mel spectrogram shape: {mel_dummy.shape}")


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class ImprovedEmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


# -----------------------------------------------------------------------------
# Data preparation
# -----------------------------------------------------------------------------
dataset = EmotionDataset(CSV_PATH, AUDIO_DIR, augment=True)
train_idx, val_idx = train_test_split(
    list(range(len(dataset))),
    test_size=0.1,
    random_state=42,
)
train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE)

print(f"Train samples: {len(train_idx)} | Val samples: {len(val_idx)}")


# -----------------------------------------------------------------------------
# Training setup
# -----------------------------------------------------------------------------
model = ImprovedEmotionCNN().to(DEVICE)
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=4, factor=0.5)

train_losses, val_losses = [], []
train_maes, val_maes = [], []
train_r2s, val_r2s = [], []
train_accs, val_accs = [], []

best_val_loss = float("inf")
print("\nStarting training...")


def compute_metrics(preds, labels):
    preds_np = np.vstack(preds)
    labels_np = np.vstack(labels)
    r2 = r2_score(labels_np, preds_np)
    mae = np.mean(np.abs(preds_np - labels_np))
    acc = np.mean((np.abs(preds_np - labels_np) < ACC_THRESHOLD).all(axis=1))
    return r2, mae, acc


for epoch in range(EPOCHS):
    # Training ---------------------------------------------------------------
    model.train()
    total_train_loss = 0.0
    train_preds, train_labels = [], []

    for mel, label in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]"):
        mel, label = mel.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        pred = model(mel)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        train_preds.append(pred.detach().cpu().numpy())
        train_labels.append(label.detach().cpu().numpy())

    avg_train_loss = total_train_loss / len(train_loader)
    train_r2, train_mae, train_acc = compute_metrics(train_preds, train_labels)

    # Validation -------------------------------------------------------------
    model.eval()
    total_val_loss = 0.0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for mel, label in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]"):
            mel, label = mel.to(DEVICE), label.to(DEVICE)
            pred = model(mel)
            loss = criterion(pred, label)
            total_val_loss += loss.item()

            val_preds.append(pred.cpu().numpy())
            val_labels.append(label.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    val_r2, val_mae, val_acc = compute_metrics(val_preds, val_labels)

    scheduler.step(avg_val_loss)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_maes.append(train_mae)
    val_maes.append(val_mae)
    train_r2s.append(train_r2)
    val_r2s.append(val_r2)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
        f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | "
        f"Train R^2: {train_r2:.4f} | Val R^2: {val_r2:.4f} | "
        f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
    )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"  -> Saved new best checkpoint (val_loss={best_val_loss:.4f})")


# -----------------------------------------------------------------------------
# Final metrics summary
# -----------------------------------------------------------------------------
final_train_loss = train_losses[-1]
final_val_loss = val_losses[-1]
final_train_mae = train_maes[-1]
final_val_mae = val_maes[-1]
final_train_acc = train_accs[-1]
final_val_acc = val_accs[-1]
final_train_r2 = train_r2s[-1]
final_val_r2 = val_r2s[-1]

print("\nTraining completed (25 epochs).")
print(
    f"Final Train Metrics -> Loss: {final_train_loss:.4f} | "
    f"MAE: {final_train_mae:.4f} | Acc: {final_train_acc:.4f} | R^2: {final_train_r2:.4f}"
)
print(
    f"Final Val Metrics   -> Loss: {final_val_loss:.4f} | "
    f"MAE: {final_val_mae:.4f} | Acc: {final_val_acc:.4f} | R^2: {final_val_r2:.4f}"
)


# -----------------------------------------------------------------------------
# Export best model to ONNX
# -----------------------------------------------------------------------------
print("\nLoading best checkpoint and exporting to ONNX...")
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
model.eval()
dummy = torch.zeros(1, 1, N_MELS, mel_db_length)
torch.onnx.export(
    model.cpu(),
    dummy,
    OUTPUT_MODEL,
    input_names=["mel"],
    output_names=["emotion"],
    opset_version=13,
    dynamic_axes={"mel": {0: "batch"}},
)
print(f"Model exported to {OUTPUT_MODEL}")


# -----------------------------------------------------------------------------
# Plot results (saved to disk)
# -----------------------------------------------------------------------------
epochs_range = range(1, len(train_losses) + 1)
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label="Train Loss", marker="o")
plt.plot(epochs_range, val_losses, label="Val Loss", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_maes, label="Train MAE", marker="o")
plt.plot(epochs_range, val_maes, label="Val MAE", marker="s")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.title("Training vs Validation MAE")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_results_static.png", dpi=300, bbox_inches="tight")
print("Saved training curves to training_results_static.png")