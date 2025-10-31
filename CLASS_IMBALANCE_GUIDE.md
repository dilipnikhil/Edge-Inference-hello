# Class Imbalance Handling Guide

## The Problem

When you have only **20 "hello" samples** but **thousands of "unknown" samples**, you create a severe class imbalance that can hurt model performance.

### Issues:
1. **Model bias**: Model learns to always predict "unknown" (it's right 99% of the time!)
2. **Poor recall**: Model misses real "hello" detections
3. **Overfitting**: Model memorizes the 20 hello samples instead of learning patterns
4. **Unstable training**: Loss fluctuates wildly

## Solutions Implemented

### 1. **Automatic Oversampling** ✅
The `CombinedDataset` automatically oversamples your "hello" samples by 5x:
- Your 20 hello files → 100 effective samples
- Uses data augmentation (noise, time variation) to create variety
- Cycles through your files multiple times with different augmentations

### 2. **Class Weighting in Loss** ✅
Automatically calculates and applies class weights:
- "hello" class gets higher weight (rarer class)
- "unknown" class gets lower weight (common class)
- Balances the loss contribution from each class

### 3. **Dataset Balancing** ✅
Limits "unknown" samples to max 10x "hello" samples:
- Prevents extreme imbalance
- Keeps training efficient
- Maintains diversity

## Recommendations

### For 20 Hello Samples:

**Good Setup:**
```bash
# This gives you:
# - 100 effective hello samples (20 × 5 augmentation)
# - 1000 unknown samples (balanced ratio)
python train_enhanced.py --hello_dir data/hello --combine --epochs 50
```

**Results:**
- ~10:1 ratio (unknown:hello) - manageable
- Weighted loss handles imbalance
- Augmentation prevents overfitting

### If You Can Add More Hello Samples:

**Best Practice:** Try to get at least **50-100 hello samples**

You can:
1. Record more samples using `create_sample_data.py`
2. Use data augmentation to expand your 20 samples
3. Collect samples in different environments

### Monitoring Class Balance:

Check the training output:
```
Class distribution:
  silence: 2004 samples (weight: 1.00)
  unknown: 10000 samples (weight: 0.10)
  hello: 100 samples (weight: 5.00)  ← Higher weight!
```

## Expected Performance

With 20 hello samples + augmentation:

**Realistic Expectations:**
- Accuracy: 85-95% (depends on diversity of hello samples)
- Precision (hello): 80-90% (some false positives)
- Recall (hello): 70-85% (may miss some detections)
- F1-Score: 75-87%

**To Improve:**
1. Record more hello samples (aim for 50-100)
2. Record in different environments
3. Use different speakers if possible
4. Increase augmentation strength

## Configuration

You can adjust in `config.py`:

```python
# Increase augmentation for hello samples
# (already handled automatically in CombinedDataset)

# Adjust class weights manually if needed
# (already calculated automatically)
```

## Best Practices

1. **Start with what you have** - 20 samples is workable with augmentation
2. **Monitor validation metrics** - Watch for overfitting
3. **Collect more data gradually** - Each new sample helps
4. **Test in real environment** - Validate on your actual use case
5. **Use early stopping** - Prevent overfitting on small dataset

## Summary

**20 hello samples is workable with:**
- ✅ 5x oversampling through augmentation
- ✅ Automatic class weighting
- ✅ Balanced dataset (10:1 ratio)
- ✅ Weighted loss function

**You'll get decent results**, but more hello samples (50-100) would definitely improve performance!

