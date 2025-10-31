"""
Lightweight Keyword Spotting Model for "hello" detection
Optimized for edge inference (<2MB, millisecond latency)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyKWS(nn.Module):
    """
    Ultra-lightweight keyword spotting model
    Architecture inspired by MobileNet and optimized for keyword detection
    Target size: <2MB, latency: <10ms on edge devices
    """
    
    def __init__(self, num_classes=2, input_size=49, num_filters=64):
        """
        Args:
            num_classes: Number of output classes (2 for hello/other)
            input_size: Number of MFCC coefficients or frequency bins
            num_filters: Base number of filters (controls model size)
        """
        super(TinyKWS, self).__init__()
        
        # First conv block - process audio features
        self.conv1 = nn.Conv1d(1, num_filters, kernel_size=10, stride=2, padding=4)
        self.bn1 = nn.BatchNorm1d(num_filters)
        
        # Depthwise separable convolution blocks (MobileNet style)
        self.conv2_dw = nn.Conv1d(num_filters, num_filters, kernel_size=3, 
                                  stride=1, padding=1, groups=num_filters)
        self.conv2_pw = nn.Conv1d(num_filters, num_filters * 2, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(num_filters * 2)
        
        self.conv3_dw = nn.Conv1d(num_filters * 2, num_filters * 2, kernel_size=3,
                                  stride=2, padding=1, groups=num_filters * 2)
        self.conv3_pw = nn.Conv1d(num_filters * 2, num_filters * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(num_filters * 4)
        
        # Global average pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final classifier
        self.fc = nn.Linear(num_filters * 4, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x shape: (batch, 1, time_steps)
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Depthwise separable convolution 1
        x = F.relu(self.bn2(self.conv2_pw(self.conv2_dw(x))))
        
        # Depthwise separable convolution 2
        x = F.relu(self.bn3(self.conv3_pw(self.conv3_dw(x))))
        
        # Global pooling
        x = self.adaptive_pool(x)
        x = x.squeeze(-1)  # Remove time dimension
        
        # Classification
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class MinimalKWS(nn.Module):
    """
    Even more minimal model for ultra-low resource devices
    Target size: <500KB
    """
    
    def __init__(self, num_classes=2, input_size=49, num_filters=32):
        super(MinimalKWS, self).__init__()
        
        self.conv1 = nn.Conv1d(1, num_filters, kernel_size=10, stride=2, padding=4)
        self.bn1 = nn.BatchNorm1d(num_filters)
        
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(num_filters * 2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters * 2, num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """Calculate model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


if __name__ == "__main__":
    # Test model
    model = TinyKWS(num_classes=2, input_size=49)
    dummy_input = torch.randn(1, 1, 98)  # (batch, channels, time)
    
    output = model(dummy_input)
    print(f"Model: TinyKWS")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Model size: {get_model_size_mb(model):.2f} MB")
    print(f"Output shape: {output.shape}")
    
    # Test minimal model
    minimal_model = MinimalKWS(num_classes=2)
    output2 = minimal_model(dummy_input)
    print(f"\nModel: MinimalKWS")
    print(f"Parameters: {count_parameters(minimal_model):,}")
    print(f"Model size: {get_model_size_mb(minimal_model):.2f} MB")
    print(f"Output shape: {output2.shape}")

