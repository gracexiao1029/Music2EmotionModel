"""
Training script using pre-extracted features files
Main differences from train_dynamic.py:
1. Load pre-extracted features from features/ directory instead of extracting mel-spectrogram from audio
2. Model input changed to feature vectors instead of images
3. Model architecture changed to MLP (Multi-Layer Perceptron) instead of CNN
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings('ignore')

# CONFIG
FEATURES_DIR = "features"  # Features file directory
VALENCE_CSV = "annotations/valence.csv"
AROUSAL_CSV = "annotations/arousal.csv"
OUTPUT_MODEL = "emotion_model_dynamic_features.onnx"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 25
BATCH_SIZE = 32  # Can increase batch size with features (feature vectors are smaller than images)
LR = 5e-4
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 8

# Data sampling configuration
MAX_SONGS = 1200
SAMPLES_PER_SONG = 25
FEATURE_WINDOW_SIZE = 5  # Use features from 5 frames before and after (11 frames total) to increase context

print(f"Using device: {DEVICE}")
print(f"Data sampling: {MAX_SONGS} songs, {SAMPLES_PER_SONG} samples per song")
print(f"Feature window size: {FEATURE_WINDOW_SIZE} frames")


# Dataset for dynamic annotations using pre-extracted features
class DynamicEmotionDatasetFeatures(Dataset):
    def __init__(self, valence_csv, arousal_csv, features_dir, max_songs=None, 
                 samples_per_song=10, feature_window_size=5, augment=False):
        self.val_df = pd.read_csv(valence_csv)
        self.aro_df = pd.read_csv(arousal_csv)
        self.features_dir = features_dir
        self.augment = augment
        self.feature_window_size = feature_window_size
        
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
                
                # Get annotation values for this time point
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
        
        # Convert labels to 1-9 range
        for s in self.samples:
            s['valence'] = 1 + (s['valence'] + 1) / 2 * 8  # [-1,1] -> [1,9]
            s['arousal'] = 1 + (s['arousal'] + 1) / 2 * 8  # [-1,1] -> [1,9]
        
        val_values_new = [s['valence'] for s in self.samples]
        aro_values_new = [s['arousal'] for s in self.samples]
        print(f"Converted Valence range: {min(val_values_new):.2f} to {max(val_values_new):.2f}")
        print(f"Converted Arousal range: {min(aro_values_new):.2f} to {max(aro_values_new):.2f}")
        
        # Preload all features files (optional, if memory is sufficient)
        # Here we use on-demand loading
        
        # Calculate feature dimension (from first file)
        self.feature_dim = None
        self._init_feature_dim()

    def _init_feature_dim(self):
        """Initialize feature dimension"""
        if self.feature_dim is None:
            # Try loading first sample's features file to determine dimension
            if len(self.samples) > 0:
                song_id = str(self.samples[0]['song_id'])
                feature_file = os.path.join(self.features_dir, f"{song_id}.csv")
                if os.path.exists(feature_file):
                    try:
                        df = pd.read_csv(feature_file, sep=';')
                        # Exclude frameTime column, only use feature columns
                        feature_cols = [col for col in df.columns if col != 'frameTime']
                        self.feature_dim = len(feature_cols)
                        print(f"Feature dimension: {self.feature_dim}")
                    except:
                        pass
            if self.feature_dim is None:
                # Default value (based on previously seen features files)
                self.feature_dim = 260
                print(f"Using default feature dimension: {self.feature_dim}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        song_id = str(sample['song_id'])
        time_sec = sample['time_sec']
        feature_file = os.path.join(self.features_dir, f"{song_id}.csv")

        # Load features file
        if not os.path.exists(feature_file):
            # If file doesn't exist, generate random features as placeholder
            feature_dim = self.feature_dim or 260
            features = np.random.randn(feature_dim).astype(np.float32) * 0.01
            features = torch.tensor(features).float()
            label = torch.tensor([sample['valence'], sample['arousal']]).float()
            return features, label

        try:
            # Read features CSV file
            df = pd.read_csv(feature_file, sep=';')
            
            # Find frame closest to time_sec
            frame_times = df['frameTime'].values
            target_frame_idx = np.argmin(np.abs(frame_times - time_sec))
            
            # Extract window features (feature_window_size frames before and after)
            start_idx = max(0, target_frame_idx - self.feature_window_size)
            end_idx = min(len(df), target_frame_idx + self.feature_window_size + 1)
            
            # Get feature columns (exclude frameTime)
            feature_cols = [col for col in df.columns if col != 'frameTime']
            window_features = df.iloc[start_idx:end_idx][feature_cols].values
            
            # Pad if window is insufficient
            window_size = 2 * self.feature_window_size + 1
            if len(window_features) < window_size:
                # Pad with first or last frame
                if len(window_features) == 0:
                    window_features = np.zeros((window_size, len(feature_cols)))
                else:
                    padding = window_size - len(window_features)
                    if start_idx == 0:
                        # Pad at the beginning
                        first_frame = window_features[0:1]
                        window_features = np.vstack([np.tile(first_frame, (padding, 1)), window_features])
                    else:
                        # Pad at the end
                        last_frame = window_features[-1:]
                        window_features = np.vstack([window_features, np.tile(last_frame, (padding, 1))])
            
            # Flatten to vector (or can keep as sequence, use RNN/Transformer)
            # Here we use flattening
            features = window_features.flatten().astype(np.float32)
            
            # Handle NaN and Inf
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Convert to tensor
            features = torch.tensor(features).float()
            
            # Normalize (optional, can skip if features are already normalized)
            # features = (features - features.mean()) / (features.std() + 1e-6)
            
            # Data augmentation: add small noise
            if self.augment and np.random.rand() < 0.5:
                noise = torch.randn_like(features) * 0.01
                features = features + noise
            
        except Exception as e:
            # If loading fails, use random features
            print(f"Error loading features for song {song_id}: {e}")
            feature_dim = self.feature_dim or 260
            window_size = 2 * self.feature_window_size + 1
            features = np.random.randn(feature_dim * window_size).astype(np.float32) * 0.01
            features = torch.tensor(features).float()

        # Label
        label = torch.tensor([sample['valence'], sample['arousal']]).float()

        return features, label


# Model - Changed to MLP (Multi-Layer Perceptron) because input is feature vector instead of image
class EmotionMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        # Dynamically adjust based on input dimension
        hidden_dim1 = min(512, input_dim * 2)
        hidden_dim2 = min(256, input_dim)
        hidden_dim3 = 128
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.BatchNorm1d(hidden_dim3),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim3, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 2)  # Output valence and arousal
        )

    def forward(self, x):
        return self.head(x)


# Prepare data
print("\nLoading dataset...")
dataset = DynamicEmotionDatasetFeatures(
    VALENCE_CSV, AROUSAL_CSV, FEATURES_DIR,
    max_songs=MAX_SONGS,
    samples_per_song=SAMPLES_PER_SONG,
    feature_window_size=FEATURE_WINDOW_SIZE,
    augment=True
)

# Get feature dimension
feature_dim = dataset.feature_dim or 260
window_size = 2 * FEATURE_WINDOW_SIZE + 1
input_dim = feature_dim * window_size
print(f"Model input dimension: {input_dim}")

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
model = EmotionMLP(input_dim).to(DEVICE)
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
    
    for features, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        features, label = features.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        pred = model(features)
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
    
    train_errors = np.abs(train_preds - train_labels)
    train_acc = np.mean((train_errors < 1.5).all(axis=1))

    # Validation
    model.eval()
    total_val_loss, total_val_mae = 0, 0
    all_val_preds, all_val_labels = [], []
    
    with torch.no_grad():
        for features, label in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            features, label = features.to(DEVICE), label.to(DEVICE)
            pred = model(features)
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
        torch.save(model.state_dict(), "best_emotion_model_dynamic_features.pt")
        print(f"  -> Saved best model (val_loss: {best_val_loss:.4f})")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered.")
            break

# Load best model
print("\nLoading best model for export...")
model.load_state_dict(torch.load("best_emotion_model_dynamic_features.pt"))

# Export to ONNX
print("Exporting to ONNX...")
model.eval()
dummy = torch.zeros(1, input_dim).to(DEVICE)
with torch.no_grad():
    torch.onnx.export(
        model.cpu(), 
        dummy.cpu(), 
        OUTPUT_MODEL,
        input_names=["features"], 
        output_names=["emotion"],
        opset_version=13, 
        dynamic_axes={"features": {0: "batch"}}
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

# R²
axes[1, 0].plot(epochs_range, train_r2s, label="Train R²", marker="o", linewidth=2)
axes[1, 0].plot(epochs_range, val_r2s, label="Val R²", marker="s", linewidth=2)
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("R² Score")
axes[1, 0].set_title("Training and Validation R²")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Accuracy
axes[1, 1].plot(epochs_range, train_accs, label="Train Accuracy", marker="o", linewidth=2)
axes[1, 1].plot(epochs_range, val_accs, label="Val Accuracy", marker="s", linewidth=2)
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Accuracy (error < 1.5)")
axes[1, 1].set_title("Training and Validation Accuracy")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_results_dynamic_features.png", dpi=150, bbox_inches='tight')
print("Plots saved to training_results_dynamic_features.png")
plt.show()

print("\nTraining completed!")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Final validation R²: {val_r2s[-1]:.4f}")
print(f"Final validation accuracy: {val_accs[-1]:.4f}")

