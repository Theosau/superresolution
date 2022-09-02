import torch.nn as nn
from torch.nn.modules.activation import ReLU

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x

class ConvAE(nn.Module):
    def __init__(self, input_size):
        super(ConvAE, self).__init__()

        # encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Interpolate(size=input_size//2, mode='bilinear'),
        )

        self.enc2 = nn.Sequential(
          nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          Interpolate(size=input_size//8, mode='bilinear'),
        )
        
        self.enc3 = nn.Sequential(
          nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          Interpolate(size=input_size//16, mode='bilinear'),
        )

        # decoder 
        self.dec1 = nn.Sequential(
          nn.Conv2d(in_channels=16, out_channels=12, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          Interpolate(size=input_size//8, mode='bilinear'),
        )
        self.dec2 = nn.Sequential(
          nn.Conv2d(in_channels=12, out_channels=8, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          Interpolate(size=input_size//2, mode='bilinear'),
        )
        self.dec3 = nn.Sequential(
          nn.Conv2d(in_channels=8, out_channels=5, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          Interpolate(size=input_size, mode='bilinear'),
        )
        self.dec4 = nn.Sequential(
          nn.Conv2d(in_channels=5, out_channels=3, kernel_size=3, stride=1, padding=1),
        )
    
    def compute_next_size(self, dimension, kernel, padding=0, stride=1):
        return int((dimension + 2*padding - kernel) / stride + 1)

    def forward(self, x):
        # encoding
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        
        # decoding
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        reconstruction = self.dec4(x)
        return reconstruction