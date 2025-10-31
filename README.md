# Real-Time Keyword Spotting: Hello Detector

A lightweight PyTorch-based keyword spotting system that detects the word "hello" in real-time audio streams. Optimized for edge deployment with models under 2MB, millisecond latency, and support for ESP32 modules.

![GUI Interface](gui.png)

## 🎯 Features

- **Ultra-lightweight models**: 2D CNN architectures optimized for edge devices (<2MB)
- **Real-time inference**: Sub-10ms latency with streaming audio processing
- **Robust to noise**: Aggressive data augmentation (noise, pitch shift, time stretch, filtering, masking)
- **Binary classification**: Simple "hello" vs "unknown" detection
- **Interactive GUI**: Real-time visualization with confidence scores and statistics
- **Edge deployment ready**: Export to ONNX, TorchScript, ExecuTorch for embedded systems
- **Comprehensive metrics**: Precision, recall, F1-score, confusion matrix, ROC curves
- **Flexible data sources**: Support for custom datasets and Google Speech Commands dataset

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Training](#training)
- [Inference](#inference)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Data Augmentation](#data-augmentation)
- [Model Export](#model-export)
- [Performance Metrics](#performance-metrics)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional Dependencies (for advanced optimization)

```bash
# For ONNX optimization
pip install onnxoptimizer

# For ExecuTorch export
pip install executorch
```

## 🎮 Quick Start

### 1. Prepare Training Data

Organize your audio files into directories:

```
data/
├── hello/          # Audio files with "hello" keyword (.wav format)
│   ├── hello_001.wav
│   ├── hello_002.wav
│   └── ...
└── other/          # Background audio or other words (.wav format)
    ├── other_001.wav
    ├── other_002.wav
    └── ...
```

**Audio Requirements:**
- Format: WAV (16kHz, mono recommended)
- Duration: ~1 second per sample
- Minimum: 20+ samples per class (more is better)

**Record Training Data:**
```bash
python create_sample_data.py
```

This will help you record audio samples directly for training.

### 2. Train the Model

**Using only your custom data:**
```bash
python train_enhanced.py --hello_dir data/hello --other_dir data/other --epochs 50
```

**Combining custom data with Speech Commands dataset:**
```bash
python train_enhanced.py --combine --hello_dir data/hello --data_root data --epochs 50
```

### 3. Run Real-Time Inference

**GUI (Recommended):**
```bash
python gui_inference.py
```

**Command-line streaming:**
```bash
python inference_enhanced.py --model models/hello_keyword.pt
```

## 📁 Project Structure

```
hello2/
├── config.py                 # Centralized configuration (dataclasses)
├── data_loader.py            # Dataset loading with augmentation support
├── train_enhanced.py         # Main training script
├── inference_enhanced.py     # CLI inference with sliding window
├── gui_inference.py          # GUI application for real-time detection
├── gui.png                   # GUI screenshot
├── model_2d.py               # 2D CNN model architectures
├── model.py                  # 1D CNN models (legacy)
├── audio_utils.py            # Audio preprocessing (MFCC extraction)
├── temporal_smoother.py      # Temporal smoothing for false positive reduction
├── metrics.py                # Evaluation metrics (precision, recall, F1)
├── evaluate_model.py         # Standalone evaluation script
├── optimized_export.py       # Model export (ONNX, TorchScript, ExecuTorch)
├── create_sample_data.py     # Utility to record training samples
│
├── data/
│   ├── hello/                # Training: "hello" samples
│   ├── other/                # Training: "unknown" samples
│   └── SpeechCommands/       # Google Speech Commands dataset (auto-downloaded)
│
├── models/                    # Trained models
│   ├── hello_keyword.pt      # PyTorch checkpoint
│   ├── hello_detector.ts     # TorchScript model
│   └── hello_detector.pte    # ExecuTorch model
│
├── plots/                     # Evaluation plots (confusion matrix, ROC, etc.)
│
└── Documentation:
    ├── OPTIMIZATION_GUIDE.md       # Model optimization techniques
    ├── CLASS_IMBALANCE_GUIDE.md    # Handling class imbalance
    └── TROUBLESHOOTING.md           # Common issues and solutions
```

## 🎓 Training

### Basic Training

Train with your custom dataset:

```bash
python train_enhanced.py --hello_dir data/hello --other_dir data/other
```

### Training Options

```bash
python train_enhanced.py \
    --hello_dir data/hello \
    --other_dir data/other \
    --batch_size 64 \
    --epochs 50 \
    --lr 0.001 \
    --model_type tiny \
    --output models/hello_model.pth
```

**Arguments:**
- `--hello_dir`: Directory containing "hello" audio files
- `--other_dir`: Directory containing "unknown" audio files
- `--combine`: Combine custom hello with Speech Commands dataset
- `--use_speech_commands`: Use only Speech Commands dataset
- `--batch_size`: Batch size (default: 64)
- `--epochs`: Number of epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--model_type`: `tiny` or `minimal` (default: `tiny`)
- `--output`: Output model path

### Training Process

The training script:
1. **Loads data** with automatic augmentation
2. **Splits dataset** into train/val/test (65%/20%/15%)
3. **Applies class weighting** to handle imbalance
4. **Uses Focal Loss** for better hard negative handling
5. **Implements early stopping** based on validation loss
6. **Exports models** to TorchScript and ExecuTorch formats
7. **Generates metrics** and plots after training

### Training Output

```
============================================================
Keyword Spotting Training
============================================================
Model type: tiny
Use 2D CNN: True
Num classes: 2 (('unknown', 'hello'))
Batch size: 64
Epochs: 50
Learning rate: 0.001

Loaded 20 hello files → 160 samples (8x)
Loaded 50 unknown files → 100 samples (2x)
Total samples: 260
Data augmentation: ENABLED (aggressive: noise, shift, pitch, stretch, volume, filter, mask)

Dataset splits:
  Train: 169 (65.0%)
  Validation: 52 (20.0%)
  Test: 39 (15.0%)

Class distribution:
  unknown: 34 (68.0%) - weight: 1.00
  hello: 16 (32.0%) - weight: 1.46

Using Weighted Focal Loss

Epoch 1/50 | train_loss=0.0469 acc=0.407 | val_loss=0.0443 acc=0.444
  → Saved best model (val_loss=0.0443, val_acc=0.444)
...
```

## 🎤 Inference

### GUI Application (Recommended)

Launch the interactive GUI:

```bash
python gui_inference.py
```

![GUI Interface](gui.png)

**Features:**
- Real-time audio streaming from microphone
- Visual detection indicators
- Adjustable confidence threshold slider
- Live statistics (detections, latency, accuracy)
- Model loading through file dialog
- Start/Stop controls

**Usage:**
1. Click "Load Model" → select `models/hello_keyword.pt`
2. Click "Start Detection"
3. Say "hello" into your microphone
4. Watch real-time detections with confidence scores

### Command-Line Inference

**Real-time streaming from microphone:**
```bash
python inference_enhanced.py --model models/hello_keyword.pt
```

**Process audio file:**
```bash
python inference_enhanced.py --model models/hello_keyword.pt --file test_audio.wav
```

**Options:**
- `--model`: Path to trained model (`.pt`, `.ts`, or `.pte` format)
- `--threshold`: Detection threshold (0-1, default: 0.85)
- `--file`: Process WAV file instead of microphone
- `--chunk_ms`: Audio chunk duration in ms (default: 1000)

### Inference Features

- **Sliding window averaging**: Reduces false positives
- **Suppression window**: Prevents duplicate detections
- **Temporal smoothing**: Debouncing for stable output
- **Multi-format support**: PyTorch, TorchScript, ExecuTorch

## 🏗️ Model Architecture

The project uses 2D convolutional neural networks that treat MFCC features as 2D images:

**Available Models:**
- **Tiny**: ~40K-50K parameters, ~1.5MB (FP32), better accuracy
- **Minimal**: ~10K-20K parameters, ~500KB (FP32), ultra-lightweight

**Input:** MFCC features `(1, 20, 98)` - 20 MFCC coefficients × 98 time frames (~1 second)

**Output:** Binary classification - "unknown" (0) or "hello" (1)

## ⚙️ Configuration

All hyperparameters are centralized in `config.py`:

### Audio Configuration

```python
AudioConfig:
  sample_rate: 16000 Hz
  clip_duration_ms: 1000 ms
  n_mfcc: 20 coefficients
  window_size_ms: 25.0 ms
  window_stride_ms: 10.0 ms
```

### Training Configuration

```python
TrainingConfig:
  batch_size: 64
  epochs: 50
  learning_rate: 0.001
  validation_split: 0.20
  label_smoothing: 0.2
  use_focal_loss: True
  weight_decay: 5e-4
```

### Model Configuration

```python
ModelConfig:
  model_type: "tiny" or "minimal"
  num_classes: 2 (binary classification)
  dropout: 0.3
  use_2d_conv: True
```

### Inference Configuration

```python
InferenceConfig:
  detection_threshold: 0.85
  averaging_window_ms: 1000
  suppression_ms: 750
  debounce_ms: 800
```

## 🔄 Data Augmentation

The training pipeline applies **aggressive data augmentation** to increase effective dataset size:

### Augmentation Techniques

1. **Noise Addition** (90% chance): Random Gaussian noise (8-25% of signal std)
2. **Time Shift** (80% chance): Shift audio left/right by ±15% of duration
3. **Pitch Shift** (70% chance): Change pitch by ±3 semitones
4. **Time Stretching** (60% chance): Speed up/slow down by 85-115%
5. **Volume Variation** (85% chance): Scale volume by 50-150%
6. **Spectral Filtering** (50% chance): Simulate different room acoustics
7. **Time Masking** (40% chance): Randomly zero small segments (10-50 samples)

### Effective Dataset Size

With augmentation enabled:
- Each file appears **10+ times per epoch** with different augmentations
- 20 hello files → 160 samples (8x multiplier) → ~1,600 variants
- 50 unknown files → 100 samples (2x multiplier) → ~1,000 variants

### Automatic Oversampling

- **Hello samples**: Automatically oversampled up to 8x to balance with unknown samples
- **Unknown samples**: Multiplied by 2x for more training data
- **Class weighting**: Automatically applied in loss function

## 📦 Model Export

Export models for edge deployment:

```bash
python optimized_export.py --model models/hello_keyword.pt
```

### Supported Formats

1. **PyTorch (.pt)**: Standard PyTorch checkpoint
2. **TorchScript (.ts)**: Optimized for inference, no Python dependency
3. **ONNX (.onnx)**: For ONNX Runtime (CPU/GPU/mobile)
4. **ExecuTorch (.pte)**: For ExecuTorch runtime (embedded systems)
5. **Quantized INT8**: Reduced size with minimal accuracy loss

### Export Options

```bash
python optimized_export.py \
    --model models/hello_keyword.pt \
    --format onnx \
    --quantize \
    --benchmark
```

## 📊 Performance Metrics

### Evaluation Script

Generate comprehensive metrics and plots:

```bash
python evaluate_model.py --model models/hello_keyword.pt
```

### Metrics Generated

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual classification breakdown
- **ROC Curve**: True positive rate vs false positive rate
- **Per-class metrics**: Detailed breakdown by class

### Output Location

Metrics and plots are saved to `plots/`:
- `evaluation_metrics.json`: Numerical metrics
- `evaluation_confusion_matrix.png`: Confusion matrix visualization
- `evaluation_roc_curve.png`: ROC curve
- `evaluation_metrics_bar.png`: Bar chart comparison

## 🔧 Troubleshooting

### Common Issues

**1. Validation accuracy stuck at same value:**
- **Cause**: Very small validation set
- **Solution**: Increase dataset size or reduce validation_split ratio
- **Check**: Look at debug output showing validation sample count

**2. Poor detection accuracy:**
- **Cause**: Insufficient or poor-quality training data
- **Solution**: 
  - Record more diverse samples (different speakers, environments)
  - Use `--combine` flag to leverage Speech Commands dataset
  - Increase training epochs
  - Adjust detection threshold

**3. Model overfitting (high train acc, low val acc):**
- **Cause**: Too few samples, insufficient regularization
- **Solution**: 
  - Increase dropout (already set to 0.3)
  - Use more aggressive augmentation
  - Increase weight_decay
  - Early stopping is enabled automatically

**4. No audio device found:**
- **Solution**: Install `sounddevice` and system audio drivers
  ```bash
  pip install sounddevice
  # On Linux: sudo apt-get install portaudio19-dev
  ```

**5. torch.load error with ProjectConfig:**
- **Cause**: PyTorch security changes
- **Solution**: Already fixed in code with `weights_only=False`
- **Note**: All torch.load calls are updated

**6. Low validation accuracy initially:**
- **Cause**: Normal during early training
- **Solution**: Wait for training to progress, early stopping will prevent overfitting

### Getting Help

1. Check `TROUBLESHOOTING.md` for detailed solutions
2. Review training debug output for batch analysis
3. Check validation debug output for prediction breakdown
4. Verify dataset splits are balanced

## 📈 Model Performance

### Typical Results

With ~20 hello samples and ~50 unknown samples:

- **Training Accuracy**: 85-95%
- **Validation Accuracy**: 70-85%
- **Test Accuracy**: 70-85%
- **Model Size**: 
  - Tiny: ~1.5MB (FP32), ~400KB (INT8)
  - Minimal: ~500KB (FP32), ~150KB (INT8)
- **Inference Latency**: ~5-10ms per chunk (CPU)

### Performance Tips

1. **More training data** = Better accuracy
2. **Diverse augmentation** = Better generalization
3. **Balanced classes** = More stable training
4. **Early stopping** = Prevents overfitting
5. **Class weighting** = Handles imbalance automatically

## 🚢 Deployment

### ESP32 Integration

1. **Export model:**
   ```bash
   python optimized_export.py --model models/hello_keyword.pt --format onnx --quantize
   ```

2. **Use ONNX Runtime Micro:**
   - Convert ONNX to ORT format
   - Integrate ONNX Runtime Micro library
   - Implement audio capture (I2S microphone)
   - Implement MFCC feature extraction in C
   - Run inference on audio chunks

3. **Use ExecuTorch:**
   - Export to `.pte` format
   - Use ExecuTorch runtime on ESP32
   - Lower memory footprint than ONNX

### Edge Device Considerations

- **Memory**: Quantized models reduce memory by 4x
- **Preprocessing**: MFCC extraction must be implemented in C/C++
- **Audio**: Use I2S interface for microphone input on ESP32

## 📚 References

- Architecture inspired by MobileNet depthwise separable convolutions
- MFCC feature extraction using librosa
- Data augmentation techniques from audio processing literature
- Inspired by Google Speech Commands and Hello repository patterns
- Optimized for edge AI inference on resource-constrained devices

## 📄 License

This project is provided as-is for educational and development purposes.

---

**Happy Training! 🎉**

For questions or issues, please open a GitHub issue.
