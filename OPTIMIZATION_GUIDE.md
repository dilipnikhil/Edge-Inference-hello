# Optimization Guide: Reducing False Positives & Edge Deployment

This guide covers strategies to reduce false positives and optimize models for edge inference (ESP32, etc.).

## Reducing False Positives

### 1. Temporal Smoothing (Implemented)

**What it does:** Requires multiple consecutive detections before confirming "hello", filtering out single-frame false positives.

**Usage in GUI:**
- Enable "Temporal Smoothing" checkbox (enabled by default)
- Adjust "Smoothing Window" (1-10 frames)
  - Higher = fewer false positives but more latency
  - Recommended: 3-5 frames

**How it works:**
- Maintains a sliding window of recent detections
- Requires majority of positive detections in window
- Includes debouncing (minimum time between detections)
- Adaptive threshold adjustment based on noise level

### 2. Training Improvements

**Focal Loss** (already implemented in `train.py`):
- Focuses on hard negative examples
- Reduces false positives by penalizing confident wrong predictions
- Better than standard cross-entropy for imbalanced data

**Training Tips:**
1. **More negative samples:** Collect diverse "other" audio (background noise, different words, silence)
2. **Hard negative mining:** Add confusing samples that sound similar to "hello"
3. **Data augmentation:**
   - Noise injection (already implemented)
   - Speed variation (time stretching)
   - Pitch shifting
   - Room reverb simulation

**Enhanced Training Command:**
```bash
python train.py --hello_dir data/hello --other_dir data/other \
    --epochs 100 --batch_size 32 --lr 0.001 \
    --model_type tiny
```

### 3. Threshold Tuning

**In GUI:**
- Start with threshold 0.75-0.85 (higher = fewer false positives)
- Monitor "Filtered FP" statistic to see how many false positives are caught
- Balance between false positives and missed detections

**Rule of thumb:**
- If too many false positives → Increase threshold
- If missing real detections → Decrease threshold

### 4. Post-Processing Strategies

**Additional filters you can add:**

1. **Energy-based filtering:**
   - Reject detections in very quiet audio
   - Reject if audio energy is too low

2. **Duration filtering:**
   - "Hello" should take ~0.5-1.5 seconds
   - Reject very short or very long detections

3. **Pattern matching:**
   - Check if detection pattern matches expected "hello" pattern
   - Look for specific MFCC coefficient sequences

## Model Optimization for Edge Inference

### Export Formats Comparison

| Format | Size | Speed | ESP32 Support |
|--------|------|-------|---------------|
| PyTorch (.pth) | ~1.5MB | Medium | No |
| ONNX (.onnx) | ~1.2MB | Fast | Via ONNX Runtime Micro |
| ONNX Optimized | ~1.0MB | Very Fast | Via ONNX Runtime Micro |
| TorchScript | ~1.5MB | Fast | Limited |
| INT8 Dynamic | ~400KB | Very Fast | Yes (with custom engine) |
| INT8 Static | ~400KB | Very Fast | Yes (with custom engine) |
| ExecuTorch | ~500KB | Very Fast | Yes (officially supported) |
| Pruned (30%) | ~1.0MB | Fast | Yes |

### Optimization Commands

**1. Export Optimized ONNX:**
```bash
python optimized_export.py --model models/hello_model.pth \
    --formats onnx --output_dir exports
```

**2. Export INT8 Quantized (Smallest):**
```bash
python optimized_export.py --model models/hello_model.pth \
    --formats int8_dynamic --output_dir exports
```

**3. Export All Formats:**
```bash
python optimized_export.py --model models/hello_model.pth \
    --formats all --output_dir exports
```

**4. Export with Pruning (Reduce size by 30%):**
```bash
python optimized_export.py --model models/hello_model.pth \
    --formats pruned --pruning_ratio 0.3 --output_dir exports
```

**5. Benchmark Model:**
```bash
python optimized_export.py --model models/hello_model.pth \
    --benchmark
```

### ESP32 Deployment Options

#### Option 1: ONNX Runtime Micro (Recommended)

1. Export to ONNX:
```bash
python optimized_export.py --model models/hello_model.pth --formats onnx
```

2. Convert to ORT format for micro:
```bash
python -m onnxruntime.tools.convert_onnx_models_to_ort exports/model_optimized.onnx
```

3. Use ONNX Runtime Micro C++ library on ESP32
4. Implement MFCC preprocessing in C/C++

**Pros:** Official support, optimized runtime
**Cons:** Requires ONNX Runtime Micro setup

#### Option 2: ExecuTorch (Best for ESP32)

1. Install ExecuTorch:
```bash
pip install executorch
```

2. Export:
```bash
python optimized_export.py --model models/hello_model.pth --formats executorch
```

3. Use ExecuTorch runtime on ESP32

**Pros:** Officially supports ESP32, efficient
**Cons:** Newer framework

#### Option 3: INT8 Quantized with Custom Engine

1. Export quantized model:
```bash
python optimized_export.py --model models/hello_model.pth --formats int8_dynamic
```

2. Extract weights and implement custom INT8 inference in C/C++
3. Use fixed-point arithmetic

**Pros:** Maximum efficiency, full control
**Cons:** Requires custom implementation

### Performance Benchmarks

**Model Sizes:**
- TinyKWS FP32: 1.5 MB
- TinyKWS INT8: ~400 KB
- MinimalKWS FP32: 500 KB
- MinimalKWS INT8: ~150 KB

**Expected Latency on ESP32 (INT8 quantized):**
- TinyKWS: 5-10ms inference
- MinimalKWS: 2-5ms inference
- MFCC preprocessing: 5-10ms
- **Total: 10-20ms** (well under real-time requirement)

### Recommendations

**For Maximum Accuracy:**
1. Use Temporal Smoothing (window=5)
2. Retrain with Focal Loss (already enabled)
3. Collect more diverse negative samples
4. Use TinyKWS model
5. Threshold: 0.75-0.80

**For Minimum False Positives:**
1. Temporal Smoothing (window=7)
2. Higher threshold (0.85+)
3. Energy-based filtering
4. Longer debounce (1000ms)

**For Edge Deployment (ESP32):**
1. Export to ExecuTorch or INT8 ONNX
2. Use MinimalKWS if size is critical
3. Implement efficient MFCC in C/C++
4. Use fixed-point arithmetic for preprocessing
5. Target: <2MB total (model + runtime)

### Troubleshooting False Positives

**If you still have false positives:**

1. **Check your training data:**
   - Do you have enough "other" samples?
   - Are there similar-sounding words in negatives?
   - Is background noise diverse?

2. **Increase smoothing:**
   - Raise window size to 7-10
   - Increase debounce time to 1000ms

3. **Raise threshold:**
   - Try 0.85-0.90
   - Monitor missed detections vs false positives

4. **Retrain with better data:**
   - Add false positive samples to "other" class
   - Use hard negative mining
   - Increase training epochs

5. **Model architecture:**
   - Try MinimalKWS (sometimes simpler = better generalization)
   - Or train TinyKWS longer with regularization

## Quick Start Checklist

- [ ] Enable Temporal Smoothing in GUI (window=3-5)
- [ ] Set threshold to 0.75-0.80
- [ ] Monitor "Filtered FP" statistic
- [ ] Retrain with Focal Loss if needed
- [ ] Export to INT8 for edge deployment
- [ ] Test with diverse audio samples

