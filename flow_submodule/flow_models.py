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

class ConvNet(nn.Module):
    def __init__(self, input_size, channels_in=1, channels_init=3, channels_out=1, latent_space_size=20):
        super(ConvNet, self).__init__()
        self.input_size = input_size
        self.channels_init = channels_init
        self.channels_out = channels_out
        
        # encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels=channels_in, out_channels=channels_init, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(channels_init),
            nn.ReLU(),
        )
        self.inte1 = nn.Sequential(
          Interpolate(size=input_size//2, mode='trilinear'),
        )

        self.enc2 = nn.Sequential(
            nn.Conv3d(in_channels=channels_init, out_channels=channels_init*2, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(channels_init*2),
            nn.ReLU(),
        )
        self.inte2 = nn.Sequential(
            Interpolate(size=input_size//4, mode='trilinear'),
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv3d(in_channels=channels_init*2, out_channels=channels_init*4, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(channels_init*4),
            nn.ReLU(),
        )
        self.inte3 = nn.Sequential(
            Interpolate(size=input_size//8, mode='trilinear'),
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv3d(in_channels=channels_init*4, out_channels=channels_init*8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(channels_init*8),
            nn.ReLU(),
        )
        self.inte4 = nn.Sequential(
            Interpolate(size=input_size//16, mode='trilinear'),
        )
    
        self.enc5 = nn.Sequential(
            nn.Conv3d(in_channels=channels_init*8, out_channels=channels_init*16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(channels_init*16),
            nn.ReLU(),
        )
        self.inte5 = nn.Sequential(
            Interpolate(size=input_size//32, mode='trilinear'),
        )
    
        self.enc6 = nn.Sequential(
            nn.Conv3d(in_channels=channels_init*16, out_channels=channels_init*32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(channels_init*32),
            nn.ReLU(),
        )
        self.inte6 = nn.Sequential(
            Interpolate(size=input_size//64, mode='trilinear'),
        )
    
        self.enc7 = nn.Sequential(
            nn.Conv3d(in_channels=channels_init*32, out_channels=latent_space_size, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(latent_space_size),
            nn.ReLU(),
        )

        # decoder         
        self.intd00 = nn.Sequential(
            Interpolate(size=input_size//32, mode='trilinear')
        )
        self.dec00 = nn.Sequential(
            nn.Conv3d(in_channels=latent_space_size, out_channels=channels_init*32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(channels_init*32),
            nn.ReLU(),
        )
        
        self.intd0 = nn.Sequential(
            Interpolate(size=input_size//16, mode='trilinear')
        )
        self.dec0 = nn.Sequential(
            nn.Conv3d(in_channels=channels_init*32, out_channels=channels_init*16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(channels_init*16),
            nn.ReLU(),
        )
        
        self.intd1 = nn.Sequential(
            Interpolate(size=input_size//8, mode='trilinear')
        )
        self.dec1 = nn.Sequential(
            nn.Conv3d(in_channels=channels_init*16, out_channels=channels_init*8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(channels_init*8),
            nn.ReLU(),
        )


        self.intd2 = nn.Sequential(
            Interpolate(size=input_size//4, mode='trilinear')
        )
        self.dec2 = nn.Sequential(
            nn.Conv3d(in_channels=channels_init*8, out_channels=channels_init*4, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(channels_init*4),
            nn.ReLU(),
        )
 
        self.intd3 = nn.Sequential(
            Interpolate(size=input_size, mode='trilinear')
        )
        self.dec3 = nn.Sequential(
            nn.Conv3d(in_channels=channels_init*4, out_channels=channels_init*2, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(channels_init*2),
            nn.ReLU(),
        )

        self.intd4 = nn.Sequential(
            Interpolate(size=input_size//2, mode='trilinear')
        )
        self.dec4 = nn.Sequential(
            nn.Conv3d(in_channels=channels_init*2, out_channels=channels_init, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(channels_init),
            nn.ReLU(),
        )
  
        self.intd5 = nn.Sequential(
            Interpolate(size=input_size, mode='trilinear')
        )
        self.dec5 = nn.Sequential(
            nn.Conv3d(in_channels=channels_init, out_channels=channels_out, kernel_size=3, stride=1, padding=1),
        )

    def res_blox(self, in_channels, ):
        return int((dimension + 2*padding - kernel) / stride + 1)

    def forward(self, x):
        # encoding
        x = self.enc1(x)
        x = self.inte1(x)
        x = self.enc2(x)
        x = self.inte2(x)
        x = self.enc3(x)
        x = self.inte3(x)
        x = self.enc4(x)
        x = self.inte4(x)
        x = self.enc5(x)
        x = self.inte5(x)
        x = self.enc6(x)
        x = self.inte6(x)
        x = self.enc7(x)
        
        # decoder
        x = self.intd00(x)
        x = self.dec00(x)
        x = self.intd0(x)
        x = self.dec0(x)
        x = self.intd1(x)
        x = self.dec1(x)
        x = self.intd2(x)
        x = self.dec2(x)
        x = self.intd3(x)
        x = self.dec3(x)
        x = self.intd4(x)
        x = self.dec4(x)
        x = self.intd5(x)
        reconstruction = self.dec5(x)
        return reconstruction