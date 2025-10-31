"""
Enhanced data loading with Speech Commands dataset support
Inspired by Hello repository - auto-downloads and creates 2-class dataset (unknown, hello)
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
from pathlib import Path
from typing import Callable, Optional, Tuple
import torchaudio

from config import ProjectConfig
from audio_utils import AudioPreprocessor

TARGET_LABEL = "hello"


class HelloSpeechCommands(Dataset):
    """
    Dataset using Google Speech Commands with auto-download
    Creates 2-class dataset: unknown, hello
    """
    
    def __init__(
        self,
        root: str,
        subset: str = "training",
        project_cfg: ProjectConfig = None,
        include_unknown: bool = True,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            root: Root directory for dataset
            subset: "training", "validation", or "testing"
            project_cfg: Project configuration
            include_unknown: Whether to include unknown words
            transform: Optional transform function
        """
        super().__init__()
        
        if project_cfg is None:
            project_cfg = ProjectConfig()
        
        self.project_cfg = project_cfg
        self.transform = transform
        self.include_unknown = include_unknown
        
        # Download Speech Commands dataset
        try:
            self.dataset = torchaudio.datasets.SPEECHCOMMANDS(
                root=root, 
                subset=subset, 
                download=True
            )
        except Exception as e:
            print(f"Warning: Could not load Speech Commands dataset: {e}")
            print("Falling back to custom dataset loader")
            self.dataset = None
        
        if self.dataset is not None:
            # Find hello and other samples
            # This can be slow for large datasets, so we add progress indication
            self.hello_indices: list[int] = []
            self.unknown_indices: list[int] = []
            
            print(f"Indexing dataset (this may take a few minutes for large datasets)...")
            total_samples = len(self.dataset)
            processed = 0
            
            # Process in chunks to show progress
            chunk_size = max(1000, total_samples // 100)  # Show progress every ~1%
            
            for idx in range(total_samples):
                try:
                    _, _, label, *_ = self.dataset[idx]
                    if label == TARGET_LABEL:
                        self.hello_indices.append(idx)
                    elif include_unknown:
                        self.unknown_indices.append(idx)
                except Exception as e:
                    # Skip samples that fail to load
                    continue
                
                processed += 1
                
                # Show progress every chunk
                if processed % chunk_size == 0 or processed == total_samples:
                    percent = (processed / total_samples) * 100
                    print(f"  Progress: {processed}/{total_samples} ({percent:.1f}%) - "
                          f"Found {len(self.hello_indices)} 'hello' samples, "
                          f"{len(self.unknown_indices)} 'unknown' samples", end='\r')
            
            print()  # New line after progress
            
            # Check if we found any hello samples
            if len(self.hello_indices) == 0:
                print("\n" + "="*60)
                print("WARNING: No 'hello' samples found in Speech Commands dataset!")
                print("="*60)
                print("Speech Commands v0.02 does not include 'hello' as a keyword.")
                print("\nAvailable keywords: yes, no, up, down, left, right, on, off,")
                print("stop, go, zero, one, two, three, four, five, six, seven, eight, nine,")
                print("backward, forward, bed, bird, cat, dog, happy, house, marvin,")
                print("sheila, tree, wow, visual, follow, learn")
                print("\nSolutions:")
                print("1. Use your custom dataset: --hello_dir data/hello --other_dir data/other")
                print("2. Use a different keyword from Speech Commands (modify TARGET_LABEL)")
                print("3. Download Speech Commands v1 or v2 which may have 'hello'")
                print("="*60)
                raise ValueError(
                    "No 'hello' samples found. Use custom dataset with --hello_dir and --other_dir"
                )
            
            # Balance unknown samples (limit to 2x hello samples)
            generator = torch.Generator().manual_seed(project_cfg.training.seed)
            if self.include_unknown and self.unknown_indices:
                unknown_tensor = torch.tensor(self.unknown_indices)
                perm = torch.randperm(len(unknown_tensor), generator=generator)
                cap = min(len(unknown_tensor), len(self.hello_indices) * 2)
                sampled_unknown = unknown_tensor[perm[:cap]].tolist()
            else:
                sampled_unknown = []
            
            self.indices: list[Tuple[int, str]] = []
            self.indices.extend((idx, TARGET_LABEL) for idx in self.hello_indices)
            self.indices.extend((idx, "unknown") for idx in sampled_unknown)
            self.total_len = len(self.indices)
            
            print(f"\nDataset summary:")
            print(f"  'hello' samples: {len(self.hello_indices)}")
            print(f"  'unknown' samples: {len(sampled_unknown)} (sampled from {len(self.unknown_indices)})")
            print(f"  Total samples: {self.total_len}")
        else:
            # Fallback: empty dataset (will be populated by custom loader)
            self.indices = []
            self.total_len = 0
    
    def __len__(self) -> int:
        return self.total_len
    
    def __getitem__(self, index: int):
        if self.dataset is None or index >= len(self.indices):
            # Return dummy unknown data if dataset not available or index out of bounds
            noise = torch.randn(self.project_cfg.audio.target_num_samples) * 0.01
            waveform = noise
            label_name = "unknown"
        else:
            # Get sample from Speech Commands
            sample_idx, semantic_label = self.indices[index]
            try:
                waveform, sample_rate, _, *_ = self.dataset[sample_idx]
                waveform = self._fix_length(waveform, sample_rate)
                waveform = waveform.squeeze(0)
                label_name = semantic_label
            except:
                # Fallback on error - use unknown
                noise = torch.randn(self.project_cfg.audio.target_num_samples) * 0.01
                waveform = noise
                label_name = "unknown"
        
        # Apply transform if provided
        if self.transform:
            features = self.transform(waveform).to(torch.float32)
        else:
            features = waveform.to(torch.float32)
        
        # Convert label to index
        label_idx = self.project_cfg.labels.index(label_name)
        return features, torch.tensor(label_idx, dtype=torch.long)
    
    def _fix_length(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Fix waveform length to target samples"""
        if sample_rate != self.project_cfg.audio.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, 
                sample_rate, 
                self.project_cfg.audio.sample_rate
            )
        
        num_samples = waveform.shape[-1]
        target_samples = self.project_cfg.audio.target_num_samples
        
        if num_samples < target_samples:
            pad_amount = target_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif num_samples > target_samples:
            waveform = waveform[..., :target_samples]
        
        return waveform


class CustomKeywordDataset(Dataset):
    """
    Custom dataset loader from local directories
    Falls back to this if Speech Commands not available
    Creates 2-class dataset from local files (unknown, hello)
    """
    
    def __init__(
        self,
        hello_dir: str,
        other_dir: str,
        project_cfg: ProjectConfig = None,
        preprocessor: Optional[AudioPreprocessor] = None,
        augment: bool = True,
    ):
        """
        Args:
            hello_dir: Directory with "hello" audio files
            other_dir: Directory with other words/background (unknown)
            project_cfg: Project configuration
            preprocessor: Audio preprocessor
            augment: Whether to apply augmentation
        """
        super().__init__()
        
        if project_cfg is None:
            project_cfg = ProjectConfig()
        
        self.project_cfg = project_cfg
        self.preprocessor = preprocessor
        self.augment = augment
        
        # Load hello samples
        self.hello_files = []
        if hello_dir and os.path.exists(hello_dir):
            self.hello_files = list(Path(hello_dir).glob("*.wav"))
        
        # Load other/unknown samples
        self.other_files = []
        if other_dir and os.path.exists(other_dir):
            self.other_files = list(Path(other_dir).glob("*.wav"))
        
        # Create samples list - oversample hello if needed to balance
        self.samples = []
        
        # Add hello samples (with cycling for augmentation) - more aggressive oversampling
        hello_multiplier = max(1, len(self.other_files) // max(len(self.hello_files), 1))
        hello_multiplier = min(hello_multiplier, 8)  # Increased cap from 5x to 8x for more diversity
        
        for i in range(len(self.hello_files) * hello_multiplier):
            file_idx = i % len(self.hello_files)
            self.samples.append((str(self.hello_files[file_idx]), "hello"))
        
        # Also augment other samples to increase dataset size
        # Use 2x multiplier for other samples to get more training data
        other_multiplier = 2
        for _ in range(other_multiplier):
            for f in self.other_files:
                self.samples.append((str(f), "unknown"))
        
        self.total_len = len(self.samples)
        
        effective_hello = len(self.hello_files) * hello_multiplier
        effective_unknown = len(self.other_files) * other_multiplier
        
        print(f"Loaded {len(self.hello_files)} hello files → {effective_hello} samples ({hello_multiplier}x)")
        print(f"Loaded {len(self.other_files)} unknown files → {effective_unknown} samples ({other_multiplier}x)")
        print(f"Total samples: {self.total_len}")
        if self.augment:
            print(f"Data augmentation: ENABLED (aggressive: noise, shift, pitch, stretch, volume, filter, mask)")
            print(f"Each sample gets randomized augmentation every epoch → ~{self.total_len * 10} effective variants")
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        if idx < len(self.samples):
            file_path, label_name = self.samples[idx]
            
            # Load audio
            if self.preprocessor:
                audio = self.preprocessor.load_wav(file_path)
            else:
                import librosa
                audio, _ = librosa.load(file_path, sr=self.project_cfg.audio.sample_rate, mono=True)
            
            if audio is None:
                audio = np.zeros(self.project_cfg.audio.target_num_samples)
            
            # Apply augmentation if enabled
            if self.augment:
                audio = self._augment_audio(audio)
            
            # Preprocess
            if self.preprocessor:
                # Don't add noise in extract_mfcc since we handle augmentation separately
                mfcc = self.preprocessor.extract_mfcc(audio, add_noise=False)
                
                if mfcc is None:
                    features = torch.zeros(1, self.project_cfg.audio.n_mfcc, 98)
                else:
                    # Pad/truncate
                    mfcc = self.preprocessor.pad_or_truncate(
                        mfcc, 
                        target_length=98
                    )
                    # Convert to tensor: (n_mfcc, time_frames)
                    features = torch.FloatTensor(mfcc.T)
                    if features.dim() == 1:
                        features = features.unsqueeze(0)
            else:
                # Fallback: use raw waveform
                if len(audio) < self.project_cfg.audio.target_num_samples:
                    audio = np.pad(
                        audio, 
                        (0, self.project_cfg.audio.target_num_samples - len(audio))
                    )
                elif len(audio) > self.project_cfg.audio.target_num_samples:
                    audio = audio[:self.project_cfg.audio.target_num_samples]
                features = torch.FloatTensor(audio)
        else:
            # Should not reach here, but handle edge case
            raise IndexError(f"Index {idx} out of range [0, {self.total_len})")
        
        # Convert to proper shape for 2D model
        if self.project_cfg.model.use_2d_conv:
            if features.dim() == 2:
                # (n_mfcc, time_frames) -> (1, n_mfcc, time_frames)
                features = features.unsqueeze(0)
            elif features.dim() == 1:
                # Expand to 2D
                features = features.unsqueeze(0).unsqueeze(0)
        
        # Get label index
        label_idx = self.project_cfg.labels.index(label_name)
        return features, torch.tensor(label_idx, dtype=torch.long)
    
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply aggressive data augmentation to audio
        Includes: noise, time shift, pitch shift, time stretching, volume variation, spectral masking
        """
        if len(audio) == 0:
            return audio
        
        # Apply multiple augmentations - more aggressive approach
        augmentations_applied = []
        
        # 1. Add noise (90% chance, more aggressive)
        if np.random.random() < 0.9:
            noise_factor = np.random.uniform(0.08, 0.25)  # Increased from 0.05-0.15
            noise = np.random.normal(0, noise_factor * np.std(audio), len(audio))
            audio = audio + noise
            augmentations_applied.append("noise")
        
        # 2. Time shift (80% chance, larger shifts)
        if np.random.random() < 0.8:
            shift_amount = np.random.randint(
                -int(self.project_cfg.audio.sample_rate * 0.15), 
                int(self.project_cfg.audio.sample_rate * 0.15)
            )  # Increased from 0.1 to 0.15
            if shift_amount > 0:
                audio = np.pad(audio, (shift_amount, 0), mode='constant')[:len(audio)]
            elif shift_amount < 0:
                audio = np.pad(audio, (0, -shift_amount), mode='constant')[-len(audio):]
            augmentations_applied.append("shift")
        
        # 3. Pitch shift (70% chance, wider range)
        if np.random.random() < 0.7:
            pitch_shift = np.random.uniform(-3, 3)  # Increased from -2 to 2
            if abs(pitch_shift) > 0.1:
                try:
                    import librosa
                    audio = librosa.effects.pitch_shift(
                        audio, 
                        sr=self.project_cfg.audio.sample_rate, 
                        n_steps=pitch_shift
                    )
                    augmentations_applied.append("pitch")
                except:
                    pass  # Skip if librosa effects not available
        
        # 4. Time stretching (60% chance, wider range)
        if np.random.random() < 0.6:
            stretch_factor = np.random.uniform(0.85, 1.15)  # Increased from 0.9-1.1
            if abs(stretch_factor - 1.0) > 0.01:
                try:
                    import librosa
                    audio = librosa.effects.time_stretch(audio, rate=stretch_factor)
                    # Pad or truncate to maintain length
                    target_len = self.project_cfg.audio.target_num_samples
                    if len(audio) > target_len:
                        audio = audio[:target_len]
                    elif len(audio) < target_len:
                        audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
                    augmentations_applied.append("stretch")
                except:
                    pass
        
        # 5. Volume variation (85% chance, wider range)
        if np.random.random() < 0.85:
            volume_factor = np.random.uniform(0.5, 1.5)  # Increased from 0.7-1.3
            audio = audio * volume_factor
            augmentations_applied.append("volume")
        
        # 6. Add background noise / room simulation (50% chance)
        if np.random.random() < 0.5:
            # Simulate room reverb with simple filtering
            try:
                from scipy import signal
                # Add slight filtering to simulate different acoustic environments
                b, a = signal.butter(3, np.random.uniform(0.3, 0.7), 'low')
                audio = signal.filtfilt(b, a, audio)
                augmentations_applied.append("filter")
            except:
                pass
        
        # 7. Time masking / dropout (40% chance) - randomly zero out small segments
        if np.random.random() < 0.4:
            mask_length = np.random.randint(10, 50)  # 10-50 samples
            mask_start = np.random.randint(0, max(1, len(audio) - mask_length))
            audio[mask_start:mask_start + mask_length] = 0
            augmentations_applied.append("mask")
        
        # Normalize after augmentation
        if np.max(np.abs(audio)) > 0:
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        return audio


class CombinedDataset(Dataset):
    """
    Combines custom hello samples with Speech Commands dataset
    Uses custom 'hello' files + Speech Commands as 'unknown'
    """
    
    def __init__(
        self,
        hello_dir: str,
        project_cfg: ProjectConfig,
        speech_commands_root: str = "./data",
        transform: Optional[Callable] = None,
        preprocessor: Optional[AudioPreprocessor] = None,
    ):
        super().__init__()
        self.project_cfg = project_cfg
        self.transform = transform
        self.preprocessor = preprocessor
        
        # Load custom hello samples
        self.hello_files = []
        if hello_dir and os.path.exists(hello_dir):
            self.hello_files = list(Path(hello_dir).glob("*.wav"))
        
        print(f"Loaded {len(self.hello_files)} custom 'hello' samples from {hello_dir}")
        
        # Load Speech Commands for unknown samples
        self.speech_commands_indices = []
        try:
            sc_dataset = torchaudio.datasets.SPEECHCOMMANDS(
                root=speech_commands_root,
                subset="training",
                download=False  # Assume already downloaded
            )
            
            print(f"Indexing Speech Commands dataset for 'unknown' samples...")
            total_samples = len(sc_dataset)
            chunk_size = max(1000, total_samples // 100)
            processed = 0
            
            # Get all non-hello samples from Speech Commands as "unknown"
            for idx in range(total_samples):
                try:
                    _, _, label, *_ = sc_dataset[idx]
                    # Use all Speech Commands keywords as "unknown"
                    self.speech_commands_indices.append(idx)
                except:
                    continue
                
                processed += 1
                if processed % chunk_size == 0 or processed == total_samples:
                    percent = (processed / total_samples) * 100
                    print(f"  Progress: {processed}/{total_samples} ({percent:.1f}%) - "
                          f"Found {len(self.speech_commands_indices)} samples", end='\r')
            
            print()  # New line
            self.sc_dataset = sc_dataset
            
            # Limit Speech Commands to reasonable size (e.g., 10k samples)
            import torch
            generator = torch.Generator().manual_seed(project_cfg.training.seed)
            if len(self.speech_commands_indices) > 10000:
                indices_tensor = torch.tensor(self.speech_commands_indices)
                perm = torch.randperm(len(indices_tensor), generator=generator)
                self.speech_commands_indices = indices_tensor[perm[:10000]].tolist()
                print(f"Sampled {len(self.speech_commands_indices)} samples from Speech Commands")
            
        except Exception as e:
            print(f"Warning: Could not load Speech Commands: {e}")
            self.sc_dataset = None
            self.speech_commands_indices = []
        
        # Calculate total length
        # Oversample hello samples to balance dataset better
        # Multiply hello samples by 5x through augmentation cycling
        hello_multiplier = 5 if len(self.hello_files) < 100 else 1
        self.total_hello = len(self.hello_files) * hello_multiplier
        
        self.total_unknown = len(self.speech_commands_indices)
        
        # Balance: limit unknown to reasonable ratio (max 10x hello)
        max_unknown_ratio = 10
        max_unknown = self.total_hello * max_unknown_ratio
        if self.total_unknown > max_unknown:
            import torch
            generator = torch.Generator().manual_seed(project_cfg.training.seed)
            indices_tensor = torch.tensor(self.speech_commands_indices)
            perm = torch.randperm(len(indices_tensor), generator=generator)
            self.speech_commands_indices = indices_tensor[perm[:max_unknown]].tolist()
            self.total_unknown = len(self.speech_commands_indices)
            print(f"Limited 'unknown' samples to {self.total_unknown} (max {max_unknown_ratio}x 'hello')")
        
        self.total_len = self.total_hello + self.total_unknown
        
        if hello_multiplier > 1:
            print(f"Oversampling 'hello' by {hello_multiplier}x through augmentation")
        
        print(f"\nDataset summary:")
        print(f"  Custom 'hello' samples: {self.total_hello}")
        print(f"  Speech Commands 'unknown' samples: {self.total_unknown}")
        print(f"  Total samples: {self.total_len}")
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, index: int):
        if index < self.total_hello:
            # Custom hello sample (with augmentation to increase effective samples)
            # Use modulo to cycle through hello files multiple times
            file_idx = index % len(self.hello_files)
            file_path = self.hello_files[file_idx]
            
            # Load audio
            import librosa
            audio, _ = librosa.load(str(file_path), sr=self.project_cfg.audio.sample_rate, mono=True)
            if len(audio) < self.project_cfg.audio.target_num_samples:
                audio = np.pad(audio, (0, self.project_cfg.audio.target_num_samples - len(audio)))
            elif len(audio) > self.project_cfg.audio.target_num_samples:
                audio = audio[:self.project_cfg.audio.target_num_samples]
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float()
            
            # Apply augmentation to hello samples (noise variation)
            if self.preprocessor and index % 5 != 0:  # Augment most samples
                noise_factor = 0.05 + (index % 10) * 0.01
                noise = torch.randn_like(audio_tensor) * noise_factor * audio_tensor.std()
                audio_tensor = audio_tensor + noise
            
            # Extract MFCC features
            if self.transform:
                # torchaudio transform: expects (batch, time) -> returns (batch, n_mfcc, time_frames)
                # Add batch dimension
                audio_tensor = audio_tensor.unsqueeze(0)  # (1, time)
                features = self.transform(audio_tensor).to(torch.float32)
                # Remove batch dim: (1, n_mfcc, time_frames) -> (n_mfcc, time_frames)
                if features.dim() == 3:
                    features = features.squeeze(0)
            elif self.preprocessor:
                # Use preprocessor
                audio_np = audio_tensor.numpy()
                mfcc = self.preprocessor.extract_mfcc(audio_np, add_noise=False)
                if mfcc is None:
                    features = torch.zeros(self.project_cfg.audio.n_mfcc, 98)
                else:
                    mfcc = self.preprocessor.pad_or_truncate(mfcc, target_length=98)
                    # mfcc is (time_frames, n_mfcc), transpose to (n_mfcc, time_frames)
                    features = torch.FloatTensor(mfcc.T)
            else:
                # Fallback: raw waveform (won't work with 2D model)
                features = audio_tensor
            
            label_name = "hello"
        
        elif index < self.total_hello + self.total_unknown:
            # Speech Commands sample
            sc_idx = index - self.total_hello
            sc_sample_idx = self.speech_commands_indices[sc_idx]
            
            try:
                waveform, sample_rate, _, *_ = self.sc_dataset[sc_sample_idx]
                
                # Fix length and sample rate
                if sample_rate != self.project_cfg.audio.sample_rate:
                    waveform = torchaudio.functional.resample(
                        waveform, sample_rate, self.project_cfg.audio.sample_rate
                    )
                
                num_samples = waveform.shape[-1]
                target_samples = self.project_cfg.audio.target_num_samples
                
                if num_samples < target_samples:
                    waveform = torch.nn.functional.pad(waveform, (0, target_samples - num_samples))
                elif num_samples > target_samples:
                    waveform = waveform[..., :target_samples]
                
                waveform = waveform.squeeze(0)
                
                if self.transform:
                    # torchaudio MFCC transform returns: (1, n_mfcc, time_frames)
                    features = self.transform(waveform).to(torch.float32)
                    # Squeeze batch dim if present: (1, n_mfcc, time_frames) -> (n_mfcc, time_frames)
                    if features.dim() == 3:
                        if features.shape[0] == 1:
                            features = features.squeeze(0)  # (n_mfcc, time_frames)
                        else:
                            # Might be (n_mfcc, time_frames, 1) - transpose
                            if features.shape[2] == 1:
                                features = features.squeeze(2)  # (n_mfcc, time_frames)
                else:
                    # No transform - this shouldn't happen with 2D model
                    # But if it does, need to convert waveform to MFCC or use different model
                    print("Warning: No transform provided but 2D model expects MFCC features")
                    # For now, return zeros - this will fail training and show the issue
                    features = torch.zeros(self.project_cfg.audio.n_mfcc, 98)
            except:
                # Fallback on error
                noise = torch.randn(self.project_cfg.audio.target_num_samples) * 0.01
                features = noise.to(torch.float32)
            
            label_name = "unknown"
        
        else:
            # Should not reach here, but handle edge case
            raise IndexError(f"Index {index} out of range [0, {self.total_len})")
        
        # Ensure features are in correct format for return
        # Dataset should return (n_mfcc, time_frames) for 2D model
        # DataLoader will batch it to (batch, n_mfcc, time_frames)
        # Then training code adds channel: (batch, 1, n_mfcc, time_frames)
        
        if self.project_cfg.model.use_2d_conv:
            # Must be 2D: (n_mfcc, time_frames)
            if features.dim() == 1:
                # Wrong - create proper 2D features
                print(f"Warning: Got 1D features for 2D model, creating zeros: {features.shape}")
                features = torch.zeros(self.project_cfg.audio.n_mfcc, 98)
            elif features.dim() == 2:
                # Correct: (n_mfcc, time_frames) - do nothing
                pass
            elif features.dim() == 3:
                # Has extra dimension - squeeze it
                if features.shape[0] == 1:
                    features = features.squeeze(0)
                elif features.shape[2] == 1:
                    features = features.squeeze(2)
        else:
            # 1D model: should be (time,)
            if features.dim() == 2:
                # (n_mfcc, time) - take first row or average
                features = features[0, :]  # Take first MFCC
            elif features.dim() == 1:
                # Already (time,) - correct
                pass
        
        label_idx = self.project_cfg.labels.index(label_name)
        return features, torch.tensor(label_idx, dtype=torch.long)


def create_dataloaders(
    project_cfg: ProjectConfig,
    batch_size: int = None,
    num_workers: int = 4,
    root: str = "./data",
    hello_dir: str = None,
    other_dir: str = None,
    use_speech_commands: bool = None,
    combine_custom_with_sc: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders
    
    Args:
        project_cfg: Project configuration
        batch_size: Batch size (uses config if None)
        num_workers: Number of worker processes
        root: Root directory for Speech Commands dataset
        hello_dir: Directory with hello samples (for custom dataset)
        other_dir: Directory with other samples (for custom dataset)
        use_speech_commands: Whether to use Speech Commands (auto-detect if None)
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if batch_size is None:
        batch_size = project_cfg.training.batch_size
    
    if use_speech_commands is None:
        use_speech_commands = project_cfg.training.use_speech_commands
    
    # Create MFCC transform
    try:
        import torchaudio
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=project_cfg.audio.sample_rate,
            n_mfcc=project_cfg.audio.n_mfcc,
            melkwargs={
                "n_mels": project_cfg.audio.n_mels,
                "n_fft": int(project_cfg.audio.sample_rate * project_cfg.audio.window_size_ms / 1000),
                "hop_length": int(project_cfg.audio.sample_rate * project_cfg.audio.window_stride_ms / 1000),
                "f_min": project_cfg.audio.f_min,
                "f_max": project_cfg.audio.f_max,
            },
        )
    except ImportError:
        mfcc_transform = None
    
    # Create dataset
    if combine_custom_with_sc and hello_dir:
        # Combined: Custom hello + Speech Commands unknown
        print("="*60)
        print("Using COMBINED dataset:")
        print("  - Custom 'hello' samples from local directory")
        print("  - Speech Commands keywords as 'unknown' samples")
        print("="*60)
        
        # Use transform if available, otherwise use preprocessor
        # For 2D model, we need MFCC features
        if mfcc_transform:
            train_dataset = CombinedDataset(
                hello_dir=hello_dir,
                project_cfg=project_cfg,
                speech_commands_root=root,
                transform=mfcc_transform,
                preprocessor=None  # Use transform instead
            )
        else:
            train_dataset = CombinedDataset(
                hello_dir=hello_dir,
                project_cfg=project_cfg,
                speech_commands_root=root,
                transform=None,
                preprocessor=AudioPreprocessor(
                    sample_rate=project_cfg.audio.sample_rate,
                    n_mfcc=project_cfg.audio.n_mfcc
                )
            )
        
        if len(train_dataset) == 0:
            raise ValueError("Combined dataset is empty!")
            
    elif use_speech_commands:
        try:
            print("="*60)
            print("Using Speech Commands dataset...")
            print("="*60)
            print("Note: First-time indexing may take 5-10 minutes for large datasets")
            print("This is a one-time operation. Subsequent runs will be faster.\n")
            
            train_dataset = HelloSpeechCommands(
                root=root,
                subset="training",
                project_cfg=project_cfg,
                transform=mfcc_transform
            )
            
            if len(train_dataset) == 0:
                print("\nSpeech Commands dataset empty, falling back to custom dataset")
                use_speech_commands = False
            else:
                print("\n" + "="*60)
                print("Dataset loaded successfully!")
                print("="*60)
        except KeyboardInterrupt:
            print("\n\nIndexing interrupted by user.")
            print("You can:")
            print("  1. Wait for it to complete (it's processing the dataset)")
            print("  2. Use custom dataset instead: --hello_dir data/hello --other_dir data/other")
            print("  3. Press Ctrl+C again to exit")
            raise
        except Exception as e:
            print(f"\nCould not load Speech Commands dataset: {e}")
            print("Falling back to custom dataset")
            import traceback
            traceback.print_exc()
            use_speech_commands = False
    
    if not use_speech_commands:
        # Use custom dataset
        print("Using custom dataset from local directories...")
        preprocessor = AudioPreprocessor(
            sample_rate=project_cfg.audio.sample_rate,
            n_mfcc=project_cfg.audio.n_mfcc
        )
        
        train_dataset = CustomKeywordDataset(
            hello_dir=hello_dir or "data/hello",
            other_dir=other_dir or "data/other",
            project_cfg=project_cfg,
            preprocessor=preprocessor,
            augment=True
        )
    
    if len(train_dataset) == 0:
        raise ValueError(
            "No training data found! Provide audio files in:\n"
            f"  - {hello_dir or 'data/hello'} (for 'hello' keyword)\n"
            f"  - {other_dir or 'data/other'} (for other words/background)\n"
            "\nOr enable Speech Commands dataset with use_speech_commands=True"
        )
    
    # Split dataset with stratification to ensure balanced splits
    total_len = len(train_dataset)
    val_len = int(total_len * project_cfg.training.validation_split)
    test_len = int(total_len * project_cfg.training.test_split)
    train_len = total_len - val_len - test_len
    
    # Ensure minimum sizes
    if val_len < 2:
        val_len = max(2, total_len // 10)
        train_len = total_len - val_len - test_len
    if test_len < 2:
        test_len = max(2, total_len // 10)
        train_len = total_len - val_len - test_len
    
    print(f"\nDataset splits:")
    print(f"  Train: {train_len} ({train_len/total_len*100:.1f}%)")
    print(f"  Validation: {val_len} ({val_len/total_len*100:.1f}%)")
    print(f"  Test: {test_len} ({test_len/total_len*100:.1f}%)")
    
    generator = torch.Generator().manual_seed(project_cfg.training.seed)
    train_set, val_set, test_set = random_split(
        train_dataset, 
        [train_len, val_len, test_len], 
        generator=generator
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

