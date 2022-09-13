from torch import cat
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
    def __init__(self, input_size, channels_init=16):
        super(ConvAE, self).__init__()

        # encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=channels_init, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            Interpolate(size=input_size//2, mode='bilinear'),
        )

        self.enc2 = nn.Sequential(
          nn.Conv2d(in_channels=channels_init, out_channels=channels_init*2, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          Interpolate(size=input_size//4, mode='bilinear'),
        )
        
        self.enc3 = nn.Sequential(
          nn.Conv2d(in_channels=channels_init*2, out_channels=channels_init*4, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          Interpolate(size=input_size//8, mode='bilinear'),
        )

        # decoder 
        self.dec1 = nn.Sequential(
          nn.Conv2d(in_channels=channels_init*4, out_channels=channels_init*2, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          Interpolate(size=input_size//4, mode='bilinear'),
        )
        self.dec2 = nn.Sequential(
          nn.Conv2d(in_channels=channels_init*2, out_channels=channels_init, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          Interpolate(size=input_size//2, mode='bilinear'),
        )
        self.dec3 = nn.Sequential(
          nn.Conv2d(in_channels=channels_init, out_channels=5, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          Interpolate(size=input_size, mode='bilinear'),
        )
        self.dec4 = nn.Sequential(
          nn.Conv2d(in_channels=5, out_channels=3, kernel_size=3, stride=1, padding=1),
        )

    def compute_next_size(self, dimension, kernel, padding=0, stride=1):
        return int((dimension + 2*padding - kernel) / stride + 1)

    def forward(self, x_input):
        # encoding
        x = self.enc1(x_input)
        x = self.enc2(x)
        x = self.enc3(x)
        
        # decoding
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        reconstruction = self.dec4(x)
        return reconstruction





class ConvUNet(nn.Module):
    def __init__(self, input_size, channels_init=16):
        super(ConvUNet, self).__init__()
        self.input_size = input_size
        self.channels_init = channels_init

        # encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=channels_init, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.inte1 = nn.Sequential(
          Interpolate(size=input_size//2, mode='bilinear'),
        )

        self.enc2 = nn.Sequential(
          nn.Conv2d(in_channels=channels_init, out_channels=channels_init*2, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
        )
        self.inte2 = nn.Sequential(
          Interpolate(size=input_size//4, mode='bilinear'),
        )
        
        self.enc3 = nn.Sequential(
          nn.Conv2d(in_channels=channels_init*2, out_channels=channels_init*4, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
        )
        self.inte3 = nn.Sequential(
          Interpolate(size=input_size//8, mode='bilinear'),
        )

        # linear layer internal
        self.lin1 = nn.Sequential(
            nn.Linear(in_features=(channels_init*4)*((input_size//8)**2), out_features=(channels_init*4)*((input_size//8)**2), bias=True),
            nn.ReLU()
        )

        # decoder 
        self.intd1 = nn.Sequential(
          Interpolate(size=input_size//4, mode='bilinear')
        )
        self.dec1 = nn.Sequential(
          nn.Conv2d(in_channels=2*channels_init*4, out_channels=channels_init*2, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
        )

        self.intd2 = nn.Sequential(
          Interpolate(size=input_size//2, mode='bilinear')
        )
        self.dec2 = nn.Sequential(
          nn.Conv2d(in_channels=2*channels_init*2, out_channels=channels_init, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
        )

        self.intd3 = nn.Sequential(
          Interpolate(size=input_size, mode='bilinear')
        )
        self.dec3 = nn.Sequential(
          nn.Conv2d(in_channels=2*channels_init, out_channels=5, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
        )


        self.dec4 = nn.Sequential(
          nn.Conv2d(in_channels=5, out_channels=3, kernel_size=3, stride=1, padding=1),
        )

    def compute_next_size(self, dimension, kernel, padding=0, stride=1):
        return int((dimension + 2*padding - kernel) / stride + 1)

    def forward(self, x_input):
        # encoding
        x_enc1 = self.enc1(x_input)
        x = self.inte1(x_enc1)
        x_enc2 = self.enc2(x)
        x = self.inte2(x_enc2)
        x_enc3 = self.enc3(x)
        x = self.inte3(x_enc3)
        
        # latent space operation
        x = self.lin1(x.view(-1, self.channels_init*4*((self.input_size//8)**2)))
        
        # decoder
        x = self.intd1(x.view(-1, self.channels_init*4, self.input_size//8, self.input_size//8))
        x = self.dec1(cat([x_enc3, x], dim=1))
        x = self.intd2(x)
        x = self.dec2(cat([x_enc2, x], dim=1))
        x = self.intd3(x)
        x = self.dec3(cat([x_enc1, x], dim=1))
        reconstruction = self.dec4(x)
        return reconstruction