## Music2EmotionModel

A deep learning system that predicts emotional dimensions (valence and arousal) from music audio using mel-spectrogram features and a CNN architecture.

### Environment Setup

This project requires **Python 3.7+**, recommended **Python 3.8-3.10**.

#### Method 1: Using conda (Recommended)

**Option A: One-click creation using environment.yml (Simplest)**

```bash
# Execute in project directory, will automatically create environment named music2emotion
conda env create -f environment.yml

# Activate environment
conda activate music2emotion
```

**Option B: Manual environment creation**

```bash
# Create new conda environment
conda create -n music2emotion python=3.9
conda activate music2emotion

# Install PyTorch (choose according to your system, CPU version as example here)
# If you have GPU, visit https://pytorch.org/ for appropriate installation command
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install other dependencies
pip install -r requirements.txt
```

**Update environment** (if environment.yml is modified):
```bash
conda env update -f environment.yml --prune
```

**Remove environment** (if need to recreate):
```bash
conda deactivate  # Exit environment first
conda env remove -n music2emotion
```

#### Method 2: Using venv (Built-in Python)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Method 3: Direct pip (Not recommended, may pollute system environment)

```bash
pip install -r requirements.txt
```

#### Verify Installation

```bash
python -c "import torch; import librosa; import onnxruntime; print('All dependencies installed successfully!')"
```

#### System Requirements

- **Operating System**: macOS, Linux, or Windows
- **Python**: 3.7 or higher
- **Memory**: Recommended at least 8GB RAM (during training)
- **Storage**: At least 2GB free space (for models and dependencies)
- **GPU**: Optional, but can accelerate training (requires CUDA version of PyTorch)

### Dataset

https://cvml.unige.ch/databases/DEAM/

### Features

- **CNN Architecture**: 3-layer convolutional network with batch normalization, max pooling, and adaptive average pooling
- **Data Augmentation**: SpecAugment with time and frequency masking applied during training
- **Training Optimizations**: AdamW optimizer with weight decay, ReduceLROnPlateau scheduler, and early stopping
- **ONNX Export**: Model exported to ONNX format for cross-platform deployment 

### Input

**Audio Format**:
- WAV files at 22,050 Hz sample rate
- Fixed 30-second duration (truncated or zero-padded)

**Feature Representation**:
- Mel-spectrograms with 64 mel bands and 512 hop length
- Converted to dB scale and normalized per-sample (mean=0, std=1)
- Input tensor shape: `[batch, 1, 64, 1292]`

**Training Labels**:
- Valence and arousal annotations from CSV file
- Normalized to [0, 1] range

### Output

**Predictions**:
- Two-dimensional emotion vector: `[valence, arousal]`
- Valence: pleasure/displeasure axis
- Arousal: energy/activation level
- Output tensor shape: `[batch, 2]`

**Model Artifacts**:
- PyTorch checkpoint: `best_emotion_model.pt` (for continued training)
- ONNX model: `emotion_model_improved.onnx` (for inference)

### Notes

The training uses SmoothL1Loss and tracks both loss and MAE metrics during training and validation. The system includes visualization of training curves and supports early stopping after 8 epochs without improvement. For inference, the exported ONNX model can be used with `test.py` to process audio files in batch.
