"""
Configuration management using dataclasses
Inspired by the Hello repository structure
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 16000
    clip_duration_ms: int = 1000
    window_size_ms: float = 25.0
    window_stride_ms: float = 10.0
    n_mfcc: int = 20  # Increased from 13 for better features
    n_mels: int = 32
    f_min: float = 20.0
    f_max: float = 4000.0
    
    @property
    def target_num_samples(self) -> int:
        return int(self.sample_rate * self.clip_duration_ms / 1000)


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 0.001
    validation_split: float = 0.20  # Increased for better validation estimate
    test_split: float = 0.15
    seed: int = 42
    label_smoothing: float = 0.2  # Increased to reduce overfitting
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    weight_decay: float = 5e-4  # Increased from 1e-4 for stronger regularization
    use_speech_commands: bool = False  # Auto-download dataset


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    model_type: str = "tiny"  # "tiny" or "minimal"
    num_classes: int = 2  # unknown, hello (binary classification)
    dropout: float = 0.3  # Increased from 0.2 to reduce overfitting
    alpha: float = 0.5  # Width multiplier
    use_2d_conv: bool = True  # Treat MFCC as 2D image
    exported_model_path: str = "models/hello_detector.pte"
    torchscript_model_path: str = "models/hello_detector.ts"
    onnx_model_path: str = "models/hello_detector.onnx"


@dataclass
class InferenceConfig:
    """Real-time inference configuration"""
    detection_threshold: float = 0.85  # Higher default to reduce false positives
    averaging_window_ms: int = 1000  # Sliding window for averaging scores
    suppression_ms: int = 750  # Suppress detections after trigger
    window_ms: int = 200  # Audio chunk size for streaming
    enable_temporal_smoothing: bool = True
    smoothing_window_size: int = 5
    debounce_ms: int = 800


@dataclass
class ProjectConfig:
    """Main project configuration"""
    labels: tuple = ("unknown", "hello")  # Binary classification: not hello vs hello
    audio: AudioConfig = None
    training: TrainingConfig = None
    model: ModelConfig = None
    inference: InferenceConfig = None
    
    def __post_init__(self):
        if self.audio is None:
            self.audio = AudioConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.inference is None:
            self.inference = InferenceConfig()

