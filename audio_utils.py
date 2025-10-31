"""
Audio preprocessing utilities for real-time keyword spotting
Optimized for edge devices with efficient feature extraction
Supports both librosa and torchaudio for flexibility
"""

import numpy as np
import torch
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
from scipy import signal
import librosa


class AudioPreprocessor:
    """
    Efficient audio preprocessing for keyword spotting
    Uses MFCC features optimized for edge devices
    """
    
    def __init__(self, 
                 sample_rate=16000,
                 n_mfcc=13,
                 n_fft=512,
                 hop_length=256,
                 n_mels=40,
                 frame_length_ms=25,
                 frame_shift_ms=10):
        """
        Args:
            sample_rate: Audio sample rate (16kHz typical for speech)
            n_mfcc: Number of MFCC coefficients (13 is standard)
            n_fft: FFT window size
            hop_length: Hop length between frames
            frame_length_ms: Frame length in milliseconds
            frame_shift_ms: Frame shift in milliseconds
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.frame_length = int(sample_rate * frame_length_ms / 1000)
        self.frame_shift = int(sample_rate * frame_shift_ms / 1000)
        
    def load_wav(self, file_path):
        """Load WAV file and return audio array"""
        try:
            # Use librosa for flexible audio loading
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            return audio
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None
    
    def extract_mfcc(self, audio, add_noise=False, noise_factor=0.1, use_torchaudio=False):
        """
        Extract MFCC features from audio
        
        Args:
            audio: Audio signal (numpy array)
            add_noise: Whether to add noise for robustness
            noise_factor: Noise scaling factor
            use_torchaudio: Use torchaudio instead of librosa (faster, PyTorch-native)
        
        Returns:
            MFCC features (time_steps, n_mfcc)
        """
        if len(audio) == 0:
            return None
            
        # Add noise if requested (for data augmentation during training)
        if add_noise:
            noise = np.random.normal(0, noise_factor * np.std(audio), len(audio))
            audio = audio + noise
        
        # Normalize audio
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # Use torchaudio if available and requested (better for PyTorch pipeline)
        if use_torchaudio and TORCHAUDIO_AVAILABLE:
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Create MFCC transform
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=self.n_mfcc,
                melkwargs={
                    "n_mels": self.n_mels,
                    "n_fft": self.n_fft,
                    "hop_length": self.hop_length,
                    "f_min": 20.0,
                    "f_max": 4000.0,
                }
            )
            
            # Extract MFCC
            mfcc_tensor = mfcc_transform(audio_tensor)
            # Convert to numpy: (1, n_mfcc, time_frames) -> (time_frames, n_mfcc)
            mfcc = mfcc_tensor.squeeze(0).T.numpy()
        else:
            # Use librosa (fallback)
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            
            # Transpose to (time_steps, n_mfcc)
            mfcc = mfcc.T
        
        return mfcc
    
    def pad_or_truncate(self, features, target_length=98):
        """
        Pad or truncate features to target length
        
        Args:
            features: Feature array (time_steps, n_features)
            target_length: Target number of time steps
        
        Returns:
            Padded/truncated features (target_length, n_features)
        """
        if features is None:
            return np.zeros((target_length, self.n_mfcc))
            
        current_length = features.shape[0]
        
        if current_length > target_length:
            # Truncate
            features = features[:target_length, :]
        elif current_length < target_length:
            # Pad with zeros
            padding = np.zeros((target_length - current_length, features.shape[1]))
            features = np.vstack([features, padding])
        
        return features
    
    def preprocess_for_model(self, audio, target_length=98, add_noise=False):
        """
        Complete preprocessing pipeline for model input
        
        Returns:
            Tensor of shape (1, 1, target_length) ready for model
        """
        # Extract MFCC
        mfcc = self.extract_mfcc(audio, add_noise=add_noise)
        
        # Pad/truncate
        mfcc = self.pad_or_truncate(mfcc, target_length)
        
        # Use delta features (first derivative) for better accuracy
        # This is optional but improves performance
        delta = librosa.feature.delta(mfcc.T)
        mfcc_delta = np.vstack([mfcc.T, delta]).T
        
        # Take only the first MFCC coefficient across time (simple feature)
        # For minimal model, we use just the first coefficient
        feature_vector = mfcc[:, 0]  # Shape: (target_length,)
        
        # Normalize
        feature_vector = (feature_vector - np.mean(feature_vector)) / (np.std(feature_vector) + 1e-8)
        
        # Convert to tensor and add batch/channel dimensions
        feature_tensor = torch.FloatTensor(feature_vector).unsqueeze(0).unsqueeze(0)
        # Shape: (1, 1, target_length)
        
        return feature_tensor
    
    def stream_preprocess(self, audio_chunk):
        """
        Preprocess audio chunk for real-time streaming
        
        Args:
            audio_chunk: Audio array (numpy)
        
        Returns:
            Preprocessed tensor ready for inference
        """
        return self.preprocess_for_model(audio_chunk, add_noise=False)


class AudioStreamer:
    """
    Real-time audio streaming interface
    Supports microphone input and file playback
    """
    
    def __init__(self, sample_rate=16000, chunk_size_ms=1000):
        """
        Args:
            sample_rate: Audio sample rate
            chunk_size_ms: Chunk size in milliseconds for processing
        """
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_size_ms / 1000)
        
    def record_chunk(self, duration_ms=1000):
        """
        Record audio chunk from microphone
        
        Args:
            duration_ms: Duration in milliseconds
        
        Returns:
            Audio array
        """
        try:
            import sounddevice as sd
            duration = duration_ms / 1000.0
            audio = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            return audio.flatten()
        except ImportError:
            print("sounddevice not installed. Install with: pip install sounddevice")
            return None
        except Exception as e:
            print(f"Error recording audio: {e}")
            return None
    
    def stream_from_mic(self, callback, chunk_duration_ms=1000):
        """
        Stream audio from microphone in real-time
        
        Args:
            callback: Function to call with each audio chunk
            chunk_duration_ms: Duration of each chunk in milliseconds
        """
        try:
            import sounddevice as sd
            
            def audio_callback(indata, frames, time, status):
                if status:
                    print(f"Audio callback status: {status}")
                audio_chunk = indata[:, 0].flatten()
                callback(audio_chunk)
            
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=audio_callback,
                blocksize=self.chunk_size,
                dtype='float32'
            ):
                print("Streaming audio... Press Ctrl+C to stop")
                import time
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopped streaming")
        except ImportError:
            print("sounddevice not installed. Install with: pip install sounddevice")
        except Exception as e:
            print(f"Error in audio streaming: {e}")

