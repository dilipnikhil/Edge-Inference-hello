"""
2D CNN model that treats MFCC features as images
Inspired by the Hello repository - often better for audio classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


class DepthwiseSeparable2D(nn.Module):
    """Depthwise separable 2D convolution block"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.block(x)


class TinyKWS2D(nn.Module):
    """
    2D CNN for keyword spotting
    Treats MFCC features as (n_mfcc, time_frames) images
    Often performs better than 1D convolutions
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        base_channels = max(int(16 * config.alpha), 8)
        
        # Feature extraction - process MFCC as 2D
        self.features = nn.Sequential(
            # Initial conv
            nn.Conv2d(1, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # Depthwise separable blocks
            DepthwiseSeparable2D(base_channels, base_channels, stride=1),
            DepthwiseSeparable2D(base_channels, base_channels * 2, stride=2),
            DepthwiseSeparable2D(base_channels * 2, base_channels * 2, stride=1),
            DepthwiseSeparable2D(base_channels * 2, base_channels * 4, stride=2),
            
            # Final projection
            nn.Conv2d(base_channels * 4, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.num_classes),
        )
    
    def forward(self, x):
        # x shape: (batch, 1, n_mfcc, time_frames)
        x = self.features(x)
        x = self.classifier(x)
        return x


class MinimalKWS2D(nn.Module):
    """Minimal 2D CNN for ultra-low resource devices"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        base_channels = max(int(8 * config.alpha), 4)
        
        self.features = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            DepthwiseSeparable2D(base_channels, base_channels * 2, stride=2),
            DepthwiseSeparable2D(base_channels * 2, base_channels * 4, stride=2),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(config.dropout),
            nn.Linear(base_channels * 4, config.num_classes),
        )
    
    def forward(self, x):
        return self.features(x)


def build_model(config: ModelConfig):
    """Build model based on configuration"""
    if config.use_2d_conv:
        if config.model_type == "tiny":
            return TinyKWS2D(config)
        else:
            return MinimalKWS2D(config)
    else:
        # Fallback to 1D models from model.py
        from model import TinyKWS, MinimalKWS
        if config.model_type == "tiny":
            return TinyKWS(num_classes=config.num_classes, num_filters=64)
        else:
            return MinimalKWS(num_classes=config.num_classes, num_filters=32)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """Calculate model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


if __name__ == "__main__":
    from config import ProjectConfig
    
    cfg = ProjectConfig()
    cfg.model.model_type = "tiny"
    cfg.model.use_2d_conv = True
    
    model = build_model(cfg.model)
    
    # Test with typical MFCC input: (batch, 1, n_mfcc=20, time_frames=~98)
    dummy_input = torch.randn(1, 1, cfg.audio.n_mfcc, 98)
    output = model(dummy_input)
    
    print(f"Model: TinyKWS2D")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Model size: {get_model_size_mb(model):.2f} MB")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

