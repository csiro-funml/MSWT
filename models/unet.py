# ---------------------------------------------------------------------------------------------
# Author: Xuesong
# Date: 08/28/2025
# This code is an ablation study of HFS (high frequency scaling)
#  https://github.com/SiaK4/HFS_ResUNet/blob/main/Models/ResUnet.py
# I will remove the HFS part and keep the ResNet remain unchanged.
# ---------------------------------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AdaptiveSwish(nn.Module):
    def __init__(self,beta_init=1.0):
        super(AdaptiveSwish,self).__init__()

        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self,x):
        return x*torch.sigmoid(self.beta*x)

class AdaptiveTanh(nn.Module):
    def __init__(self,alpha_init=1.0):
        super(AdaptiveTanh, self).__init__()
        self.alpha = nn.parameter(torch.tensor(alpha_init))

    def forward(self,x):
        return torch.tanh(self.alpha*x)

class Rowdy(nn.Module):
    def __init__(self, beta_init=1.0, cos_terms=2):
        super(Rowdy, self).__init__()
        self.amplitudes = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(cos_terms)])
        self.frequencies = nn.ParameterList([nn.Parameter(0.1*torch.ones(1)) for _ in range(cos_terms)])

        self.base_frequencies = torch.arange(10, 10*(cos_terms+1), 10, dtype=torch.float32)

        self.beta = nn.Parameter(torch.tensor(beta_init))

def get_activation(activation_name):
    if activation_name =='adaptive_swish':
        return AdaptiveSwish()
    if activation_name =='adaptive_tanh':
        return AdaptiveTanh()
    if activation_name =='Rowdy':
        return Rowdy()
    if activation_name =='GELU':
        return nn.GELU(approximate='tanh')

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super(ResidualBlock, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(approximate='tanh'),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(approximate='tanh')
        )

        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
            #nn.GELU(approximate='tanh')

        )
    def forward(self, x):
        shortcut = self.skip(x)
        out = self.residual(x)
        return out+shortcut

class ResidualBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super(ResidualBlock2, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1,out_channels),
            # nn.GELU(approximate='tanh'),
            activation,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1,out_channels),
            # nn.GELU(approximate='tanh')
            activation
        )

        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(1,out_channels)
            #nn.GELU(approximate='tanh')

        )
    def forward(self, x):
        shortcut = self.skip(x)
        out = self.residual(x)
        return out+shortcut

class UNet(nn.Module):
    def __init__(self, in_c,out_c, features = [64,128,256,512,512],bottleneck_feature=1024, patch_size_enc = [16,8,4,2,1], patch_size_dec=[16,8,4,2,1],activation_name='GELU'
                 ,device=torch.device('cpu')):
        super(UNet, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.lamb1_history = []
        self.lamb2_history = []
        self.activation = get_activation(activation_name)
        self.device = device
        #Encoder and featscale
        self.encoder = nn.ModuleList()

        num_layers = len(features)  # Assuming featscale lists match encoder layers

        for i,feature in enumerate(features):
            self.encoder.append(ResidualBlock2(in_c,feature,self.activation))
            in_c = feature

        # Bottleneck layer
        self.bottleneck = ResidualBlock2(features[-1], bottleneck_feature,self.activation)

        #Upsample and Decoder and featscale
        self.upsample = nn.ModuleList()
        self.decoder = nn.ModuleList()


        for i, feature in enumerate(reversed(features)):
            self.upsample.append(
                nn.ConvTranspose2d(bottleneck_feature, bottleneck_feature, kernel_size=2, stride=2)
            )
            self.decoder.append(ResidualBlock(bottleneck_feature+feature, feature,self.activation))
            bottleneck_feature = feature
        
        self.final_conv = nn.Conv2d(features[0],self.out_c,kernel_size=1)

    def save_lambdas(self):
        self.lamb1_history.append(self.lamb1.item())
        self.lamb2_history.append(self.lamb2.item())


    def get_grid(self, x):
        batchsize, size_x, size_y = x.shape[0], x.shape[1], x.shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).to(x.device)
        return grid
    
    def forward(self, x):
        # absort the time dimension into the channel dimensionx = x.view(*x.shape[:-2], -1)           #### B, X, Y, T*C
        B, H, W, T, C = x.shape
        x = x.view(*x.shape[:-2], -1)           #### B, H, W, T*C
        grid = self.get_grid(x)
        x = torch.cat((x, grid), dim=-1)        #### B, H, W, T*C +2
        x = x.permute(0, 3, 1, 2).contiguous() # (B, T*C+2, H, W)
        
        #Downsampling path
        skip_connections = []
        for i, down in enumerate(self.encoder):
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2)
        
        x = self.bottleneck(x)

        #Upsampling path
        skip_connections = skip_connections[::-1]
        for up in range(len(self.decoder)):
            x = self.upsample[up](x)
            x = torch.cat((x, skip_connections[up]),dim=1)
            x = self.decoder[up](x)
        out = self.final_conv(x)

        # reshape back to (B, C_out, H, W) -> (B, H, W, T, C)
        out = out.permute(0, 2, 3, 1).contiguous()
        out = out.view(B, H, W, -1, C)
        return out
    


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_channels = 3
    T_in = 7
    T_ar = 1
    model = UNet(in_c = n_channels * T_in + 2 ,out_c = n_channels, 
                 bottleneck_feature=512, 
                 device=device).to(device)
    x = torch.rand(2, 96, 192, T_in, n_channels)
    y = model(x)
    print(y.shape)