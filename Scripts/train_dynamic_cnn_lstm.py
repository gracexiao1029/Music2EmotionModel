"""
CNN+LSTM architecture for dynamic emotion prediction (Unity Barracuda compatible)
Architecture:
1. CNN extracts local spatial features from mel-spectrogram
2. LSTM processes time sequences to capture temporal evolution of emotions
3. Fully connected layers output valence and arousal

Advantages:
- CNN excels at extracting local features (frequency-time patterns)
- LSTM excels at modeling temporal dependencies (Unity Barracuda supports LSTM)
- Combining both can better predict dynamic emotional changes
- Compatible with Unity Barracuda for Quest 2 deployment
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid plt.show() blocking
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
OUTPUT_MODEL = "emotion_model_dynamic_cnn_lstm.onnx"
SR = 22050
N_MELS = 64
HOP_LENGTH = 512
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
EPOCHS = 25
BATCH_SIZE = 8  # CNN+LSTM requires more memory, reduce batch size
LR = 5e-4
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 8

# Data sampling configuration
MAX_SONGS = 1200
SAMPLES_PER_SONG = 25
SEGMENT_DURATION = 5  # Audio length per time step (seconds)
TIME_STEPS = 5  # Time sequence length (5 time steps, 25 seconds total)

print(f"Using device: {DEVICE}")
print(f"Data sampling: {MAX_SONGS} songs, {SAMPLES_PER_SONG} samples per song")
print(f"Time sequence: {TIME_STEPS} steps, {SEGMENT_DURATION}s per step")
print("Unity Barracuda compatible: Using LSTM instead of GRU")


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


# Dataset for dynamic annotations with time sequences
class DynamicEmotionDatasetSequence(Dataset):
    def __init__(self, valence_csv, arousal_csv, audio_dir, max_songs=None, 
                 samples_per_song=10, segment_duration=5, time_steps=5, augment=False):
        self.val_df = pd.read_csv(valence_csv)
        self.aro_df = pd.read_csv(arousal_csv)
        self.audio_dir = audio_dir
        self.augment = augment
        self.segment_duration = segment_duration
        self.time_steps = time_steps
        
        # Limit number of songs
        if max_songs:
            self.val_df = self.val_df.head(max_songs)
            self.aro_df = self.aro_df.head(max_songs)
        
        # Get time point column names (exclude song_id column)
        self.time_columns = [col for col in self.val_df.columns if col != 'song_id']
        
        # Build sample list: sample multiple time points per song
        # Each sample contains a time sequence (multiple consecutive time steps)
        self.samples = []
        for idx in range(len(self.val_df)):
            song_id = int(self.val_df.iloc[idx]['song_id'])
            # Randomly sample starting time points
            available_times = self.time_columns
            if len(available_times) > samples_per_song:
                # Sample starting points, ensure enough time steps
                max_start_idx = len(available_times) - time_steps
                if max_start_idx > 0:
                    start_indices = np.random.choice(range(max_start_idx), 
                                                     min(samples_per_song, max_start_idx), 
                                                     replace=False)
                else:
                    start_indices = [0] if len(available_times) > 0 else []
            else:
                start_indices = [0] if len(available_times) >= time_steps else []
            
            for start_idx in start_indices:
                # Extract consecutive time steps
                time_sequence = []
                label_sequence = []
                
                for step in range(time_steps):
                    if start_idx + step >= len(available_times):
                        break
                    
                    time_col = available_times[start_idx + step]
                    # Extract time point (milliseconds)
                    time_ms = int(time_col.replace('sample_', '').replace('ms', ''))
                    time_sec = time_ms / 1000.0
                    
                    # Get annotation values for this time point
                    val = self.val_df.iloc[idx][time_col]
                    aro = self.aro_df.iloc[idx][time_col]
                    
                    # Skip NaN values
                    if pd.isna(val) or pd.isna(aro):
                        break
                    
                    time_sequence.append({
                        'time_sec': time_sec,
                        'valence': float(val),
                        'arousal': float(aro)
                    })
                    label_sequence.append([float(val), float(aro)])
                
                # If time sequence is complete, add to sample list
                if len(time_sequence) == time_steps:
                    self.samples.append({
                        'song_id': song_id,
                        'time_sequence': time_sequence,
                        'label': label_sequence[-1]  # Use last time step's label as target
                    })
        
        print(f"Created {len(self.samples)} time sequence samples from {len(self.val_df)} songs")
        
        # Check value range and convert to 1-9
        all_val = []
        all_aro = []
        for s in self.samples:
            for ts in s['time_sequence']:
                all_val.append(ts['valence'])
                all_aro.append(ts['arousal'])
        
        if all_val:
            print(f"Original Valence range: {min(all_val):.2f} to {max(all_val):.2f}")
            print(f"Original Arousal range: {min(all_aro):.2f} to {max(all_aro):.2f}")
            
            # Convert labels to 1-9 range
            for s in self.samples:
                for ts in s['time_sequence']:
                    ts['valence'] = 1 + (ts['valence'] + 1) / 2 * 8  # [-1,1] -> [1,9]
                    ts['arousal'] = 1 + (ts['arousal'] + 1) / 2 * 8  # [-1,1] -> [1,9]
                # Convert target label
                s['label'] = [1 + (s['label'][0] + 1) / 2 * 8, 
                             1 + (s['label'][1] + 1) / 2 * 8]
            
            all_val_new = [ts['valence'] for s in self.samples for ts in s['time_sequence']]
            all_aro_new = [ts['arousal'] for s in self.samples for ts in s['time_sequence']]
            print(f"Converted Valence range: {min(all_val_new):.2f} to {max(all_val_new):.2f}")
            print(f"Converted Arousal range: {min(all_aro_new):.2f} to {max(all_aro_new):.2f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        song_id = str(sample['song_id'])
        time_sequence = sample['time_sequence']
        file_path = os.path.join(self.audio_dir, f"{song_id}.wav")

        # Store mel-spectrogram for each time step
        mel_sequence = []
        
        # Load audio (load only once)
        if os.path.exists(file_path):
            y, sr = librosa.load(file_path, sr=SR)
        else:
            # If file doesn't exist, generate random noise
            y = np.random.randn(int(self.segment_duration * TIME_STEPS * SR)).astype(np.float32) * 0.01
            sr = SR

        # Extract mel-spectrogram for each time step
        for ts in time_sequence:
            time_sec = ts['time_sec']
            
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

            # SpecAugment (only applied to training set)
            if self.augment:
                if np.random.rand() < 0.5:
                    mel_db = spec_augment(mel_db)

            mel_sequence.append(mel_db)

        # Stack as time sequence: [time_steps, 1, n_mels, time_frames]
        mel_sequence = torch.stack(mel_sequence, dim=0)
        
        # Label (use last time step's label)
        label = torch.tensor(sample['label']).float()

        return mel_sequence, label


# Calculate mel-spectrogram length
print("Pre-calculating spectrogram length...")
y_dummy = np.zeros(SR * SEGMENT_DURATION)
mel_dummy = librosa.feature.melspectrogram(y=y_dummy, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
mel_db_length = mel_dummy.shape[1]
print(f"Mel spectrogram shape: {mel_dummy.shape}")


# CNN+LSTM Model (Unity Barracuda compatible)
class EmotionCNNLSTM(nn.Module):
    def __init__(self, n_mels=64, time_frames=None):
        super().__init__()
        
        # CNN feature extractor (extract spatial features for each time step)
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

        self.cnn_features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),  # Add one more conv layer to increase model capacity (consistent with train_dynamic.py)
        )
        
        # Calculate CNN output dimension
        # Input: [1, n_mels, time_frames] = [1, 64, time_frames]
        # After 4 MaxPool2d(2): [1, 64, time_frames] -> [1, 4, time_frames/16]
        # Output channels: 256
        # Use AdaptiveAvgPool2d to pool spatial dimensions to 1x1
        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # CNN output feature dimension
        cnn_output_dim = 256
        
        # LSTM processes time sequence (Unity Barracuda compatible)
        # Using batch_first=True for better Unity compatibility
        # Single layer LSTM for better Unity support (multi-layer may have issues)
        self.lstm_hidden_dim = 128
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=self.lstm_hidden_dim,
            num_layers=1,  # Single layer for Unity compatibility
            batch_first=True,  # [batch, seq_len, features] format for Unity
            dropout=0.0,  # No dropout for single layer
            bidirectional=False  # Unidirectional for Unity compatibility
        )
        
        # Output layer
        self.head = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)  # Output valence and arousal
        )

    def forward(self, x):
        """
        Args:
            x: [batch, time_steps, 1, n_mels, time_frames]
        Returns:
            output: [batch, 2] (valence, arousal)
        """
        batch_size, time_steps, channels, n_mels, time_frames = x.size()
        
        # Reshape to [batch * time_steps, 1, n_mels, time_frames]
        x = x.view(batch_size * time_steps, channels, n_mels, time_frames)
        
        # CNN feature extraction: [batch * time_steps, 256]
        x = self.cnn_features(x)  # [batch * time_steps, 256, h, w]
        x = self.cnn_pool(x)  # [batch * time_steps, 256, 1, 1]
        x = x.view(batch_size * time_steps, -1)  # [batch * time_steps, 256]
        
        # Reshape back to time sequence: [batch, time_steps, 256]
        x = x.view(batch_size, time_steps, -1)  # [batch, time_steps, 256]
        
        # LSTM processes time sequence (batch_first=True)
        lstm_out, (hidden, cell) = self.lstm(x)  # lstm_out: [batch, time_steps, hidden_dim]
        
        # Use last time step's output
        last_output = lstm_out[:, -1, :]  # [batch, hidden_dim]
        
        # Output layer
        output = self.head(last_output)  # [batch, 2]
        
        return output


# Prepare data
print("\nLoading dataset...")
dataset = DynamicEmotionDatasetSequence(
    VALENCE_CSV, AROUSAL_CSV, AUDIO_DIR,
    max_songs=MAX_SONGS,
    samples_per_song=SAMPLES_PER_SONG,
    segment_duration=SEGMENT_DURATION,
    time_steps=TIME_STEPS,
    augment=True
)

train_idx, val_idx = train_test_split(
    list(range(len(dataset))), 
    test_size=0.2,
    random_state=42
)

train_loader = DataLoader(
    torch.utils.data.Subset(dataset, train_idx),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)
val_loader = DataLoader(
    torch.utils.data.Subset(dataset, val_idx),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

# Initialize model
model = EmotionCNNLSTM(n_mels=N_MELS, time_frames=mel_db_length).to(DEVICE)
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=3, factor=0.5, verbose=True
)

# Training loop
train_losses, val_losses = [], []
train_maes, val_maes = [], []
train_r2s, val_r2s = [], []
train_accs, val_accs = [], []

best_val_loss = float("inf")
epochs_no_improve = 0

print("\nStarting training...")
for epoch in range(EPOCHS):
    # Training
    model.train()
    total_train_loss, total_train_mae = 0, 0
    all_train_preds, all_train_labels = [], []
    
    for mel_seq, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        mel_seq, label = mel_seq.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        pred = model(mel_seq)
        loss = criterion(pred, label)
        loss.backward()
        
        # Gradient clipping (prevent gradient explosion)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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
    
    train_errors = np.abs(train_preds - train_labels)
    train_acc = np.mean((train_errors < 1.5).all(axis=1))

    # Validation
    model.eval()
    total_val_loss, total_val_mae = 0, 0
    all_val_preds, all_val_labels = [], []
    
    with torch.no_grad():
        for mel_seq, label in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            mel_seq, label = mel_seq.to(DEVICE), label.to(DEVICE)
            pred = model(mel_seq)
            loss = criterion(pred, label)
            
            total_val_loss += loss.item()
            total_val_mae += torch.abs(pred - label).mean().item()
            
            all_val_preds.append(pred.cpu().numpy())
            all_val_labels.append(label.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_mae = total_val_mae / len(val_loader)
    
    val_preds = np.vstack(all_val_preds)
    val_labels = np.vstack(all_val_labels)
    val_r2 = r2_score(val_labels, val_preds)
    
    val_errors = np.abs(val_preds - val_labels)
    val_acc = np.mean((val_errors < 1.5).all(axis=1))

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
        torch.save(model.state_dict(), "best_emotion_model_dynamic_cnn_lstm.pt")
        print(f"  -> Saved best model (val_loss: {best_val_loss:.4f})")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered.")
            break

# Load best model
print("\nLoading best model for export...")
model.load_state_dict(torch.load("best_emotion_model_dynamic_cnn_lstm.pt"))

# Export to ONNX (Unity Barracuda compatible)
print("Exporting to ONNX for Unity Barracuda...")
model.eval()
dummy = torch.zeros(1, TIME_STEPS, 1, N_MELS, mel_db_length).to(DEVICE)
with torch.no_grad():
    torch.onnx.export(
        model.cpu(), 
        dummy.cpu(), 
        OUTPUT_MODEL,
        input_names=["mel_sequence"], 
        output_names=["emotion"],
        opset_version=12,  # Use opset 12 for Unity Barracuda compatibility
        dynamic_axes={"mel_sequence": {0: "batch"}},
        do_constant_folding=True  # Optimize for inference
    )
print(f"Model exported to {OUTPUT_MODEL}")
print("Note: This model uses LSTM which is supported by Unity Barracuda (with limitations)")

# Plot results
print("\nGenerating plots...")
epochs_range = range(1, len(train_losses) + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(epochs_range, train_losses, label="Train Loss", marker="o", linewidth=2)
axes[0].plot(epochs_range, val_losses, label="Val Loss", marker="s", linewidth=2)
axes[0].set_xlabel("Epoch", fontsize=12)
axes[0].set_ylabel("Loss", fontsize=12)
axes[0].set_title("Training and Validation Loss", fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(epochs_range, train_accs, label="Train Accuracy", marker="o", linewidth=2)
axes[1].plot(epochs_range, val_accs, label="Val Accuracy", marker="s", linewidth=2)
axes[1].set_xlabel("Epoch", fontsize=12)
axes[1].set_ylabel("Accuracy (error < 1.5)", fontsize=12)
axes[1].set_title("Training and Validation Accuracy", fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
# Save as PNG, SVG, and PDF for high resolution and selectable text
plt.savefig("training_results_dynamic_cnn_lstm.png", dpi=300, bbox_inches='tight')
plt.savefig("training_results_dynamic_cnn_lstm.svg", format='svg', bbox_inches='tight')
plt.savefig("training_results_dynamic_cnn_lstm.pdf", format='pdf', bbox_inches='tight')
print("Plots saved to:")
print("  - training_results_dynamic_cnn_lstm.png (300 DPI)")
print("  - training_results_dynamic_cnn_lstm.svg (vector, selectable text)")
print("  - training_results_dynamic_cnn_lstm.pdf (vector, selectable text)")
# Comment out plt.show() to avoid blocking
# plt.show()

print("\nTraining completed!")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Final validation R²: {val_r2s[-1]:.4f}")
print(f"Final validation accuracy: {val_accs[-1]:.4f}")
print("\nModel is ready for Unity Barracuda deployment on Quest 2!")

