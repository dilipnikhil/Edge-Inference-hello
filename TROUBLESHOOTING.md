# Troubleshooting Guide

## Speech Commands Dataset Indexing Takes Time

### Issue: "Stuck" after download completes

**What's happening:**
After the download completes (100%), the script needs to index all ~105,000 audio samples in the Speech Commands dataset to find which ones are "hello" and which are "unknown". This is a **one-time operation** that can take 10-15 minutes.

**You'll see:**
```
Using Speech Commands dataset...
100%|████████████| 2.26G/2.26G [00:37<00:00, 65.2MB/s]
Indexing dataset (this may take a few minutes for large datasets)...
  Progress: 5000/105829 (4.7%) - Found 231 'hello' samples, 4567 'unknown' samples
```

**This is normal!** Just wait - it's processing each sample. The progress will update.

### Solutions:

**Option 1: Wait it out (Recommended)**
- First-time indexing takes 10-15 minutes
- Subsequent runs will be faster (cached indices)
- Progress indicator shows what's happening

**Option 2: Use Custom Dataset Instead**
```bash
python train_enhanced.py \
    --hello_dir data/hello \
    --other_dir data/other
```
This skips Speech Commands entirely and uses your local files.

**Option 3: Pre-index in Background**
Run indexing separately:
```bash
python cache_dataset_index.py
```
Then training will use cached indices (much faster).

### Why it's slow:
- Speech Commands has ~105,829 training samples
- Each sample needs to be loaded to read its label
- File I/O for 105k+ files takes time
- This is a limitation of how torchaudio.datasets.SPEECHCOMMANDS works

### Future runs:
After first indexing, subsequent runs should be faster as the dataset structure is cached. The indexing only happens once per dataset version.

## Other Common Issues

### Out of Memory
- Reduce batch size: `--batch_size 32` or `--batch_size 16`
- Use minimal model: `--model_type minimal`
- Close other applications

### Slow Training
- Use smaller model: `--model_type minimal`
- Reduce epochs for testing: `--epochs 10`
- Use smaller dataset (don't use Speech Commands, use custom)

### Audio Loading Errors
- Check audio files are valid WAV format
- Ensure files are 16kHz (or let librosa resample)
- Check file permissions

## Performance Tips

1. **First Time Setup:**
   - Download and index Speech Commands (one-time, 10-15 min)
   - Or use custom dataset (instant)

2. **Training:**
   - Start with few epochs: `--epochs 10` to test
   - Use `--model_type minimal` for faster training
   - Use custom dataset if you have enough samples

3. **Development:**
   - Use small custom dataset for quick iteration
   - Use Speech Commands for final training

