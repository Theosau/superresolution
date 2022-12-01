from torch import cat
import torch.nn as nn

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x

class ConvUNetBis(nn.Module):
    def __init__(self, input_size, channels_in=4, channels_init=16, channels_out=20):
        super(ConvUNetBis, self).__init__()
        self.input_size = input_size
        self.channels_init = channels_init
        self.channels_out = channels_out

        # encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels=channels_in, out_channels=channels_init, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.inte1 = nn.Sequential(
          Interpolate(size=input_size//16, mode='trilinear'),
        )

        self.enc2 = nn.Sequential(
          nn.Conv3d(in_channels=channels_init, out_channels=channels_init*2, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
        )
        self.inte2 = nn.Sequential(
          Interpolate(size=input_size//32, mode='trilinear'),
        )
        
        self.enc3 = nn.Sequential(
          nn.Conv3d(in_channels=channels_init*2, out_channels=channels_init*4, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
        )
        self.inte3 = nn.Sequential(
          Interpolate(size=input_size//64, mode='trilinear'),
        )
        

        # linear layer internal
        self.lin1 = nn.Sequential(
            nn.Linear(in_features=(channels_init*4)*((input_size//64)**2), out_features=(channels_init*4)*((input_size//64)**2), bias=True),
            nn.ReLU()
        )

        # decoder 
        self.intd1 = nn.Sequential(
          Interpolate(size=input_size//32, mode='trilinear')
        )
        self.dec1 = nn.Sequential(
          nn.Conv3d(in_channels=2*channels_init*4, out_channels=channels_init*2, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
        )

        self.intd2 = nn.Sequential(
          Interpolate(size=input_size//16, mode='trilinear')
        )
        self.dec2 = nn.Sequential(
          nn.Conv3d(in_channels=2*channels_init*2, out_channels=channels_init, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
        )

        self.intd3 = nn.Sequential(
          Interpolate(size=input_size, mode='trilinear')
        )
        self.dec3 = nn.Sequential(
          nn.Conv3d(in_channels=2*channels_init, out_channels=3, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
        )
        
        self.dec4 = nn.Sequential(
          nn.Conv3d(in_channels=3, out_channels=channels_out, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
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
        x = self.lin1(x.view(-1, self.channels_init*4*((self.input_size//64)**2)))
        
        # decoder
        x = self.intd1(x.view(-1, self.channels_init*4, self.input_size//64, self.input_size//64, self.input_size//64))
        x = self.dec1(cat([x_enc3, x], dim=1))
        x = self.intd2(x)
        x = self.dec2(cat([x_enc2, x], dim=1))
        x = self.intd3(x)
        x = self.dec3(cat([x_enc1, x], dim=1))
        reconstruction = self.dec4(x)
        return reconstruction




class SmallLinear(nn.Module):

    def __init__(self, num_features, num_outputs):
        super(SmallLinear, self).__init__()
        self.lin1 = nn.Linear(num_features, 30)
        self.lin2 = nn.Linear(30, 20)
        self.lin3 = nn.Linear(20, 15)
        self.lin4 = nn.Linear(15, 10)
        self.lin5 = nn.Linear(10, num_outputs)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.lin1(x)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.activation(x)
        x = self.lin3(x)
        x = self.activation(x)
        x = self.lin4(x)
        x = self.activation(x)
        x = self.lin5(x)
        return x

