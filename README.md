## Music2EmotionModel

A deep learning system that predicts emotional dimensions (valence and arousal) from music audio using mel-spectrogram features and a CNN architecture.

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
