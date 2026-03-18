import torch
import torch.nn as nn
from .common import Conv

class YOLOv3ImageTransform(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):  # 1 channel for grayscale
        super().__init__()
        # YOLOv3 backbone
        self.backbone = YOLOv3Backbone(in_channels)
        
        # Adaptive decoder
        self.decoder = nn.Sequential(
            Conv(1024, 512, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv(512, 256, 3, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv(256, 128, 3, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv(128, 64, 3, 1),
            Conv(64, 32, 3, 1),
            nn.Conv2d(32, out_channels, kernel_size=1),  # 1 channel output
            nn.AdaptiveAvgPool2d((None, None))  # Maintain input spatial dimensions
        )
        
    def forward(self, x):
        # Get features from backbone
        features = self.backbone(x)
        
        # Use the deepest feature map
        x = features[-1]
        
        # Decode to image
        return self.decoder(x)

class YOLOv3Backbone(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        # Modified YOLOv3 backbone layers
        self.conv1 = Conv(in_channels, 32, 3, 1)
        self.conv2 = Conv(32, 64, 3, 2)
        # ... (add remaining backbone layers)
        
    def forward(self, x):
        # Forward pass through backbone
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        # ... (complete forward pass)
        return [x1, x2, x3]  # Return feature maps at different scales

# Example usage
if __name__ == '__main__':
    # Test with different input sizes
    model = YOLOv3ImageTransform(in_channels=1, out_channels=1)
    
    # Test 416x416 input
    input_416 = torch.randn(1, 1, 416, 416)
    output_416 = model(input_416)
    print(f"416x416 input -> output shape: {output_416.shape}")
    
    # Test 600x600 input
    input_600 = torch.randn(1, 1, 600, 600)
    output_600 = model(input_600)
    print(f"600x600 input -> output shape: {output_600.shape}")
    
    # Test 800x800 input
    input_800 = torch.randn(1, 1, 800, 800)
    output_800 = model(input_800)
    print(f"800x800 input -> output shape: {output_800.shape}")
