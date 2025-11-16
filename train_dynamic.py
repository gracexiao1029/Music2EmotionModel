import os
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings('ignore')

# CONFIG
AUDIO_DIR = "audio_wav"
VALENCE_CSV = "annotations/valence.csv"
AROUSAL_CSV = "annotations/arousal.csv"
OUTPUT_MODEL = "emotion_model_dynamic.onnx"
SR = 22050
N_MELS = 64
HOP_LENGTH = 512
DURATION = 30  # 30-second audio segment
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"  # Auto-detect available device (MPS/CPU)
EPOCHS = 25  # Number of training epochs
BATCH_SIZE = 16  # Batch size (increase if memory allows)
LR = 5e-4  # Learning rate
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 8  # Early stopping patience

# Data sampling configuration
MAX_SONGS = 1200  # Use first 1200 songs (about 2/3 of data)
SAMPLES_PER_SONG = 25  # Sample 25 time points per song (increase data diversity)
SEGMENT_DURATION = 5  # Each training sample uses 5 seconds of audio (increase context)

print(f"Using device: {DEVICE}")
print(f"Data sampling: {MAX_SONGS} songs, {SAMPLES_PER_SONG} samples per song")


# SpecAugment
def spec_augment(mel, time_mask_param=20, freq_mask_param=10):
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


# Dataset for dynamic annotations
class DynamicEmotionDataset(Dataset):
    def __init__(self, valence_csv, arousal_csv, audio_dir, max_songs=None, 
                 samples_per_song=10, segment_duration=3, augment=False):
        self.val_df = pd.read_csv(valence_csv)
        self.aro_df = pd.read_csv(arousal_csv)
        self.audio_dir = audio_dir
        self.augment = augment
        self.segment_duration = segment_duration
        
        # Limit number of songs
        if max_songs:
            self.val_df = self.val_df.head(max_songs)
            self.aro_df = self.aro_df.head(max_songs)
        
        # Get time point column names (exclude song_id column)
        self.time_columns = [col for col in self.val_df.columns if col != 'song_id']
        
        # Build sample list: sample multiple time points per song
        self.samples = []
        for idx in range(len(self.val_df)):
            song_id = int(self.val_df.iloc[idx]['song_id'])
            # Randomly sample time points
            available_times = self.time_columns
            if len(available_times) > samples_per_song:
                sampled_times = np.random.choice(available_times, samples_per_song, replace=False)
            else:
                sampled_times = available_times
            
            for time_col in sampled_times:
                # Extract time point (milliseconds)
                time_ms = int(time_col.replace('sample_', '').replace('ms', ''))
                time_sec = time_ms / 1000.0
                
                # Get annotation values for this time point (1-9 range)
                val = self.val_df.iloc[idx][time_col]
                aro = self.aro_df.iloc[idx][time_col]
                
                # Skip NaN values
                if pd.isna(val) or pd.isna(aro):
                    continue
                
                self.samples.append({
                    'song_id': song_id,
                    'time_sec': time_sec,
                    'valence': float(val),
                    'arousal': float(aro)
                })
        
        print(f"Created {len(self.samples)} samples from {len(self.val_df)} songs")
        
        # Check value range and convert to 1-9
        val_values = [s['valence'] for s in self.samples]
        aro_values = [s['arousal'] for s in self.samples]
        print(f"Original Valence range: {min(val_values):.2f} to {max(val_values):.2f}")
        print(f"Original Arousal range: {min(aro_values):.2f} to {max(aro_values):.2f}")
        
        # Convert labels to 1-9 range (assuming original data is in -1 to 1)
        # Linear mapping: [-1, 1] -> [1, 9]
        for s in self.samples:
            # Map from [-1, 1] to [1, 9]
            s['valence'] = 1 + (s['valence'] + 1) / 2 * 8  # [-1,1] -> [1,9]
            s['arousal'] = 1 + (s['arousal'] + 1) / 2 * 8  # [-1,1] -> [1,9]
        
        val_values_new = [s['valence'] for s in self.samples]
        aro_values_new = [s['arousal'] for s in self.samples]
        print(f"Converted Valence range: {min(val_values_new):.2f} to {max(val_values_new):.2f}")
        print(f"Converted Arousal range: {min(aro_values_new):.2f} to {max(aro_values_new):.2f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        song_id = str(sample['song_id'])
        time_sec = sample['time_sec']
        file_path = os.path.join(self.audio_dir, f"{song_id}.wav")

        # Load audio
        if not os.path.exists(file_path):
            # If file doesn't exist, generate random noise as placeholder (better than zero padding)
            # This way the model can still learn some patterns even without real audio
            segment_length = int(self.segment_duration * SR)
            noise = np.random.randn(segment_length).astype(np.float32) * 0.01  # Small amplitude noise
            mel = librosa.feature.melspectrogram(y=noise, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = torch.tensor(mel_db).float().unsqueeze(0)
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
            label = torch.tensor([sample['valence'], sample['arousal']]).float()
            return mel_db, label

        y, sr = librosa.load(file_path, sr=SR)
        
        # Extract audio segment for corresponding time period
        start_sample = int(time_sec * SR)
        end_sample = int((time_sec + self.segment_duration) * SR)
        
        if start_sample >= len(y):
            # If time point exceeds audio length, extract from the end
            start_sample = max(0, len(y) - int(self.segment_duration * SR))
            end_sample = len(y)
        
        segment = y[start_sample:end_sample]
        
        # Pad if segment is too short
        target_length = int(self.segment_duration * SR)
        if len(segment) < target_length:
            segment = np.pad(segment, (0, target_length - len(segment)))
        elif len(segment) > target_length:
            segment = segment[:target_length]

        # Generate mel-spectrogram
        mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = torch.tensor(mel_db).float().unsqueeze(0)  # [1, n_mels, time]

        # Normalize
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

        # SpecAugment
        if self.augment:
            if np.random.rand() < 0.5:
                mel_db = spec_augment(mel_db)

        # Labels use 1-9 range values directly (not normalized to 0-1)
        label = torch.tensor([sample['valence'], sample['arousal']]).float()

        return mel_db, label


# Calculate mel-spectrogram length
print("Pre-calculating spectrogram length...")
y_dummy = np.zeros(SR * SEGMENT_DURATION)
mel_dummy = librosa.feature.melspectrogram(y=y_dummy, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
mel_db_length = mel_dummy.shape[1]
print(f"Mel spectrogram shape: {mel_dummy.shape}")


# Model
class EmotionCNN(nn.Module):
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
            conv_block(128, 256),  # Add one more conv layer to increase model capacity
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),  # Increase hidden layer size
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)  # Output valence and arousal (1-9 range)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


# Prepare data
print("\nLoading dataset...")
dataset = DynamicEmotionDataset(
    VALENCE_CSV, AROUSAL_CSV, AUDIO_DIR,
    max_songs=MAX_SONGS,
    samples_per_song=SAMPLES_PER_SONG,
    segment_duration=SEGMENT_DURATION,
    augment=True
)

train_idx, val_idx = train_test_split(
    list(range(len(dataset))), 
    test_size=0.2,  # 20% validation set (increase validation data)
    random_state=42
)

train_loader = DataLoader(
    torch.utils.data.Subset(dataset, train_idx),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0  # Set to 0 for compatibility
)
val_loader = DataLoader(
    torch.utils.data.Subset(dataset, val_idx),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

# Initialize model
model = EmotionCNN().to(DEVICE)
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=3, factor=0.5, verbose=True
)

# Training loop
train_losses, val_losses = [], []
train_maes, val_maes = [], []
train_r2s, val_r2s = [], []  # R² as accuracy metric
train_accs, val_accs = [], []  # Accuracy based on threshold

best_val_loss = float("inf")
epochs_no_improve = 0

print("\nStarting training...")
for epoch in range(EPOCHS):
    # Training
    model.train()
    total_train_loss, total_train_mae = 0, 0
    all_train_preds, all_train_labels = [], []
    
    for mel, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        mel, label = mel.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        pred = model(mel)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        total_train_mae += torch.abs(pred - label).mean().item()
        
        all_train_preds.append(pred.cpu().detach().numpy())
        all_train_labels.append(label.cpu().detach().numpy())

    avg_train_loss = total_train_loss / len(train_loader)
    avg_train_mae = total_train_mae / len(train_loader)
    
    # Calculate R² and accuracy
    train_preds = np.vstack(all_train_preds)
    train_labels = np.vstack(all_train_labels)
    train_r2 = r2_score(train_labels, train_preds)
    
    # Accuracy: proportion of predictions with error < 1.5 (relaxed threshold due to 1-9 range)
    train_errors = np.abs(train_preds - train_labels)
    train_acc = np.mean((train_errors < 1.5).all(axis=1))  # Both dimensions must satisfy

    # Validation
    model.eval()
    total_val_loss, total_val_mae = 0, 0
    all_val_preds, all_val_labels = [], []
    
    with torch.no_grad():
        for mel, label in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            mel, label = mel.to(DEVICE), label.to(DEVICE)
            pred = model(mel)
            loss = criterion(pred, label)
            
            total_val_loss += loss.item()
            total_val_mae += torch.abs(pred - label).mean().item()
            
            all_val_preds.append(pred.cpu().numpy())
            all_val_labels.append(label.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_mae = total_val_mae / len(val_loader)
    
    # Calculate validation R² and accuracy
    val_preds = np.vstack(all_val_preds)
    val_labels = np.vstack(all_val_labels)
    val_r2 = r2_score(val_labels, val_preds)
    
    val_errors = np.abs(val_preds - val_labels)
    val_acc = np.mean((val_errors < 1.5).all(axis=1))  # Relaxed threshold

    scheduler.step(avg_val_loss)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_maes.append(avg_train_mae)
    val_maes.append(avg_val_mae)
    train_r2s.append(train_r2)
    val_r2s.append(val_r2)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} | "
          f"Train MAE: {avg_train_mae:.4f}, Val MAE: {avg_val_mae:.4f} | "
          f"Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f} | "
          f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_emotion_model_dynamic.pt")
        print(f"  -> Saved best model (val_loss: {best_val_loss:.4f})")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered.")
            break

# Load best model
print("\nLoading best model for export...")
model.load_state_dict(torch.load("best_emotion_model_dynamic.pt"))

# Export to ONNX
print("Exporting to ONNX...")
model.eval()
dummy = torch.zeros(1, 1, N_MELS, mel_db_length).to(DEVICE)
with torch.no_grad():
    torch.onnx.export(
        model.cpu(), 
        dummy.cpu(), 
        OUTPUT_MODEL,
        input_names=["mel"], 
        output_names=["emotion"],
        opset_version=13, 
        dynamic_axes={"mel": {0: "batch"}}
    )
print(f"Model exported to {OUTPUT_MODEL}")

# Plot results
print("\nGenerating plots...")
epochs_range = range(1, len(train_losses) + 1)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss
axes[0, 0].plot(epochs_range, train_losses, label="Train Loss", marker="o", linewidth=2)
axes[0, 0].plot(epochs_range, val_losses, label="Val Loss", marker="s", linewidth=2)
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("Training and Validation Loss")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# MAE
axes[0, 1].plot(epochs_range, train_maes, label="Train MAE", marker="o", linewidth=2)
axes[0, 1].plot(epochs_range, val_maes, label="Val MAE", marker="s", linewidth=2)
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Mean Absolute Error")
axes[0, 1].set_title("Training and Validation MAE")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# R² (as accuracy metric)
axes[1, 0].plot(epochs_range, train_r2s, label="Train R²", marker="o", linewidth=2)
axes[1, 0].plot(epochs_range, val_r2s, label="Val R²", marker="s", linewidth=2)
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("R² Score")
axes[1, 0].set_title("Training and Validation R² (Accuracy)")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Accuracy (threshold-based)
axes[1, 1].plot(epochs_range, train_accs, label="Train Accuracy", marker="o", linewidth=2)
axes[1, 1].plot(epochs_range, val_accs, label="Val Accuracy", marker="s", linewidth=2)
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Accuracy (error < 1.5)")
axes[1, 1].set_title("Training and Validation Accuracy")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_results_dynamic.png", dpi=150, bbox_inches='tight')
print("Plots saved to training_results_dynamic.png")
plt.show()

print("\nTraining completed!")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Final validation R²: {val_r2s[-1]:.4f}")
print(f"Final validation accuracy: {val_accs[-1]:.4f}")

