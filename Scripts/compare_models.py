"""
Compare two model variants: Dynamic CNN and CNN+LSTM.
Produce a comparison table and representative plots.

Notes:
- Dynamic CNN (from train_dynamic.py) is a single-frame model. During evaluation we only use the last
  frame of each temporal sequence.
- CNN+LSTM consumes entire temporal sequences.
- All models are evaluated on the same temporal dataset to ensure fairness.
"""
import os
import torch
import torch.nn as nn
import librosa
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend so the script doesn't hang on plt.show()
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

# CONFIG
AUDIO_DIR = "audio_wav"
VALENCE_CSV = "annotations/valence.csv"
AROUSAL_CSV = "annotations/arousal.csv"
SR = 22050
N_MELS = 64
HOP_LENGTH = 512
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 8
MAX_SONGS = 1200
SAMPLES_PER_SONG = 25
SEGMENT_DURATION = 5
TIME_STEPS = 5

# Font config (fallbacks so matplotlib can render Unicode if needed)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print(f"Using device: {DEVICE}")

# ==================== MODEL DEFINITIONS ====================
# These definitions must match the training scripts exactly so we can load checkpoints.
# - DynamicCNN   -> EmotionCNN in train_dynamic.py (single-frame model, 4 conv blocks)
# - CNNLSTM      -> EmotionCNNLSTM in train_dynamic_cnn_lstm.py (temporal model)
#
# Why redefine?
# 1. Class names differ between training scripts and this evaluation script.
# 2. Loading checkpoints requires identical module structures.
# 3. For evaluation we only need model definitions plus inference utilities.

# Dynamic CNN (single-frame model)
# Uses only the last frame of each temporal sequence for evaluation.
class DynamicCNN(nn.Module):
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
            nn.Linear(64, 2)
        )
    def forward(self, x):
        # x: [batch, 1, n_mels, time_frames]
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.head(x)

# CNN + LSTM (temporal, Unity Barracuda friendly)
class CNNLSTM(nn.Module):
    def __init__(self, n_mels=64, time_frames=None):
        super().__init__()
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
            conv_block(128, 256),  # Add one more conv layer (consistent with train_dynamic_cnn_lstm.py)
        )
        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm_hidden_dim = 128
        self.lstm = nn.LSTM(
            input_size=256,  # Updated to match 4-layer CNN output
            hidden_size=self.lstm_hidden_dim,
            num_layers=1,  # Single layer for Unity compatibility
            batch_first=True,  # [batch, seq_len, features] format for Unity
            dropout=0.0,  # No dropout for single layer
            bidirectional=False  # Unidirectional for Unity compatibility
        )
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

# ==================== DATASET ====================

# Temporal dataset (sequence sampler)
class DynamicEmotionDatasetSequence:
    def __init__(self, valence_csv, arousal_csv, audio_dir, max_songs=None, 
                 samples_per_song=10, segment_duration=5, time_steps=5):
        self.val_df = pd.read_csv(valence_csv)
        self.aro_df = pd.read_csv(arousal_csv)
        self.audio_dir = audio_dir
        self.segment_duration = segment_duration
        self.time_steps = time_steps
        if max_songs:
            self.val_df = self.val_df.head(max_songs)
            self.aro_df = self.aro_df.head(max_songs)
        self.time_columns = [col for col in self.val_df.columns if col != 'song_id']
        self.samples = []
        for idx in range(len(self.val_df)):
            song_id = int(self.val_df.iloc[idx]['song_id'])
            available_times = self.time_columns
            if len(available_times) > samples_per_song:
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
                time_sequence = []
                label_sequence = []
                for step in range(time_steps):
                    if start_idx + step >= len(available_times):
                        break
                    time_col = available_times[start_idx + step]
                    time_ms = int(time_col.replace('sample_', '').replace('ms', ''))
                    time_sec = time_ms / 1000.0
                    val = self.val_df.iloc[idx][time_col]
                    aro = self.aro_df.iloc[idx][time_col]
                    if pd.isna(val) or pd.isna(aro):
                        break
                    time_sequence.append({
                        'time_sec': time_sec,
                        'valence': float(val),
                        'arousal': float(aro)
                    })
                    label_sequence.append([float(val), float(aro)])
                if len(time_sequence) == time_steps:
                    for ts in time_sequence:
                        ts['valence'] = 1 + (ts['valence'] + 1) / 2 * 8
                        ts['arousal'] = 1 + (ts['arousal'] + 1) / 2 * 8
                    self.samples.append({
                        'song_id': song_id,
                        'time_sequence': time_sequence,
                        'label': [1 + (label_sequence[-1][0] + 1) / 2 * 8, 
                                 1 + (label_sequence[-1][1] + 1) / 2 * 8]
                    })
        print(f"Dynamic dataset: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        song_id = str(sample['song_id'])
        time_sequence = sample['time_sequence']
        file_path = os.path.join(self.audio_dir, f"{song_id}.wav")
        mel_sequence = []
        if os.path.exists(file_path):
            y, sr = librosa.load(file_path, sr=SR)
        else:
            y = np.random.randn(int(self.segment_duration * TIME_STEPS * SR)).astype(np.float32) * 0.01
            sr = SR
        for ts in time_sequence:
            time_sec = ts['time_sec']
            start_sample = int(time_sec * SR)
            end_sample = int((time_sec + self.segment_duration) * SR)
            if start_sample >= len(y):
                start_sample = max(0, len(y) - int(self.segment_duration * SR))
                end_sample = len(y)
            segment = y[start_sample:end_sample]
            target_length = int(self.segment_duration * SR)
            if len(segment) < target_length:
                segment = np.pad(segment, (0, target_length - len(segment)))
            elif len(segment) > target_length:
                segment = segment[:target_length]
            mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = torch.tensor(mel_db).float().unsqueeze(0)
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
            mel_sequence.append(mel_db)
        mel_sequence = torch.stack(mel_sequence, dim=0)
        label = torch.tensor(sample['label']).float()
        return mel_sequence, label

# ==================== EVALUATION UTILITIES ====================

def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, label in loader:
            data, label = data.to(device), label.to(device)
            pred = model(data)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(label.cpu().numpy())
    preds = np.vstack(all_preds)
    labels = np.vstack(all_labels)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)
    return mae, r2, preds, labels

# ==================== MAIN ====================

print("\n=== Model Comparison Evaluation ===\n")

# Compute mel length
y_dummy = np.zeros(SR * SEGMENT_DURATION)
mel_dummy = librosa.feature.melspectrogram(y=y_dummy, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
mel_db_length = mel_dummy.shape[1]

# Prepare dynamic dataset
print("Loading dynamic dataset...")
dynamic_dataset = DynamicEmotionDatasetSequence(VALENCE_CSV, AROUSAL_CSV, AUDIO_DIR,
                                                max_songs=MAX_SONGS, samples_per_song=SAMPLES_PER_SONG,
                                                segment_duration=SEGMENT_DURATION, time_steps=TIME_STEPS)
dynamic_train_idx, dynamic_val_idx = train_test_split(list(range(len(dynamic_dataset))),
                                                       test_size=0.2, random_state=42)
dynamic_val_loader = DataLoader(torch.utils.data.Subset(dynamic_dataset, dynamic_val_idx),
                                batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Evaluate Dynamic CNN
print("\nEvaluating Dynamic CNN...")
dynamic_model = DynamicCNN().to(DEVICE)
if os.path.exists("best_emotion_model_dynamic.pt"):
    dynamic_model.load_state_dict(torch.load("best_emotion_model_dynamic.pt", map_location=DEVICE))
    # Dynamic CNN consumes a single frame, so we use the last frame of each sequence
    dynamic_preds_list, dynamic_labels_list = [], []
    dynamic_model.eval()
    with torch.no_grad():
        for mel_seq, label in dynamic_val_loader:
            mel_seq, label = mel_seq.to(DEVICE), label.to(DEVICE)
            # last temporal slice
            last_frame = mel_seq[:, -1, :, :, :]  # [batch, 1, n_mels, time_frames]
            pred = dynamic_model(last_frame)
            dynamic_preds_list.append(pred.cpu().numpy())
            dynamic_labels_list.append(label.cpu().numpy())
    dynamic_preds = np.vstack(dynamic_preds_list)
    dynamic_labels = np.vstack(dynamic_labels_list)
    dynamic_mae = mean_absolute_error(dynamic_labels, dynamic_preds)
    dynamic_r2 = r2_score(dynamic_labels, dynamic_preds)
    print(f"Dynamic CNN - MAE: {dynamic_mae:.4f}, R²: {dynamic_r2:.4f}")
else:
    print("Dynamic CNN model not found, skipping...")
    dynamic_mae, dynamic_r2 = None, None

# Evaluate CNN + LSTM
print("\nEvaluating CNN+LSTM...")
lstm_model = CNNLSTM(n_mels=N_MELS, time_frames=mel_db_length).to(DEVICE)
if os.path.exists("best_emotion_model_dynamic_cnn_lstm.pt"):
    lstm_model.load_state_dict(torch.load("best_emotion_model_dynamic_cnn_lstm.pt", map_location=DEVICE))
    lstm_mae, lstm_r2, lstm_preds, lstm_labels = evaluate_model(lstm_model, dynamic_val_loader, DEVICE)
    print(f"CNN+LSTM - MAE: {lstm_mae:.4f}, R²: {lstm_r2:.4f}")
else:
    print("CNN+LSTM model not found, skipping...")
    lstm_mae, lstm_r2 = None, None

# ==================== VISUALIZATION ====================

print("\n=== Building comparison plots ===\n")

# Build comparison table
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Table
ax_table = fig.add_subplot(gs[0, :])
ax_table.axis('tight')
ax_table.axis('off')

# Prepare rows
models = ['Dynamic CNN', 'CNN+LSTM']
mae_values = []
r2_values = []

if dynamic_mae is not None:
    mae_values.append(f"{dynamic_mae:.4f}")
    r2_values.append(f"{dynamic_r2:.4f}")
else:
    mae_values.append("N/A")
    r2_values.append("N/A")

if lstm_mae is not None:
    mae_values.append(f"{lstm_mae:.4f}")
    r2_values.append(f"{lstm_r2:.4f}")
else:
    mae_values.append("N/A")
    r2_values.append("N/A")

table_data = [
    ['Model', 'MAE (Mean Absolute Error)', 'R² Score'],
    ['Dynamic CNN', mae_values[0], r2_values[0]],
    ['CNN+LSTM', mae_values[1], r2_values[1]]
]

table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                       colWidths=[0.3, 0.35, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1, 2.5)

# Header style
for i in range(3):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Row colors
colors = ['#C8E6C9', '#A5D6A7']
for i in range(1, 3):
    for j in range(3):
        table[(i, j)].set_facecolor(colors[i-1])

ax_table.set_title('Model Performance Comparison', fontsize=18, fontweight='bold', pad=20)

# MAE bar chart
ax_mae = fig.add_subplot(gs[1, 0])
mae_numeric = []
mae_labels = []
for i, (m, v) in enumerate(zip(models, mae_values)):
    if v != "N/A":
        mae_numeric.append(float(v))
        mae_labels.append(m)
    else:
        mae_numeric.append(0)
        mae_labels.append(m)

bars = ax_mae.bar(mae_labels, mae_numeric, color=['#4ECDC4', '#45B7D1'], alpha=0.8)
ax_mae.set_ylabel('MAE', fontsize=12, fontweight='bold')
ax_mae.set_title('Mean Absolute Error Comparison', fontsize=14, fontweight='bold')
ax_mae.grid(True, alpha=0.3, axis='y')
for i, (bar, val) in enumerate(zip(bars, mae_values)):
    if val != "N/A":
        ax_mae.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   val, ha='center', va='bottom', fontweight='bold', fontsize=11)

# R² bar chart
ax_r2 = fig.add_subplot(gs[1, 1])
r2_numeric = []
r2_labels = []
for i, (m, v) in enumerate(zip(models, r2_values)):
    if v != "N/A":
        r2_numeric.append(float(v))
        r2_labels.append(m)
    else:
        r2_numeric.append(0)
        r2_labels.append(m)

bars = ax_r2.bar(r2_labels, r2_numeric, color=['#4ECDC4', '#45B7D1'], alpha=0.8)
ax_r2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax_r2.set_title('R² Score Comparison', fontsize=14, fontweight='bold')
ax_r2.grid(True, alpha=0.3, axis='y')
ax_r2.set_ylim([0, 1])
for i, (bar, val) in enumerate(zip(bars, r2_values)):
    if val != "N/A":
        ax_r2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   val, ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.suptitle('Model Performance Comparison: Dynamic CNN vs CNN+LSTM', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("Comparison plots saved to model_comparison.png")

# Console table
print("\n" + "="*60)
print("Model Performance Summary (Dynamic CNN vs CNN+LSTM)")
print("="*60)
print(f"{'Model':<20} {'MAE':<20} {'R²':<20}")
print("-"*60)
for i, model in enumerate(models):
    print(f"{model:<20} {mae_values[i]:<20} {r2_values[i]:<20}")
print("="*60)

# plt.show() intentionally omitted so the script can run headless.

print("\nEvaluation finished!")
