# note: the code for the GumbelSoftmax was inspired
# from https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
# visited on the 29Apr21

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys
sys.path.append(os.getcwd())
from models.generic_unet import UNet, SmallUNet

class ReshapeTheta(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return x


class Encoder(nn.Module):
    def __init__(self, unet_out_channels: int, unet_init_features: int,
                 temperature: int, beta_size: int, theta_size: int, img_size: tuple, small_unet: bool):
        super().__init__()

        # Gumbel Softmax variables
        self.temperature = temperature
        self.Gumbel = torch.distributions.gumbel.Gumbel(0.0, 1.0)
        self.sm = nn.Softmax2d()
        if small_unet:
            self.unet = SmallUNet(in_channels=1, out_channels=unet_out_channels, init_features=unet_init_features)
        else:
            self.unet = UNet(in_channels=1, out_channels=unet_out_channels, init_features=unet_init_features)
        self.beta_layers = torch.nn.ModuleList()
        self.theta_layers = torch.nn.ModuleList()
        # beta layers
        self.beta_layers.append(nn.Conv2d(unet_out_channels, beta_size, 3, padding=1))
        self.beta_layers.append(nn.Sigmoid())

        # theta layers
        self.theta_layers.append(nn.Conv2d(unet_out_channels, beta_size, 32, padding=0))
        size_out = [int((i_size - 32) + 1) for i_size in img_size]
        self.theta_layers.append(ReshapeTheta())
        self.theta_layers.append(nn.Linear(np.prod(size_out + [beta_size]), theta_size))


    def GumbelSoftmax(self, x):
        G = self.Gumbel.rsample(sample_shape=x.shape)
        x = x + G.to(x.device)
        x = x / self.temperature
        y = self.sm(x)
        # do the hardcoded one
        argy = torch.argmax(y, dim=1)
        y_hard = F.one_hot(argy, y.shape[1])
        y_hard = torch.moveaxis(y_hard, -1, 1)
        return (y_hard - y).detach() + y


    def forward(self, img):
        out = self.unet(img)
        #
        beta = torch.clone(out)
        for layer in self.beta_layers:
            beta = layer(beta)

        #
        theta = out
        for layer in self.theta_layers:
            theta = layer(theta)

        return beta, theta


class Decoder(nn.Module):
    def __init__(self, unet_init_features:int, beta_size:int, theta_size:int, small_unet: bool):
        super().__init__()

        # let's do  a four layer conv with leaky relu activations - super simple
        if small_unet:
            self.unet = SmallUNet(in_channels=beta_size + theta_size, out_channels=1, init_features=unet_init_features)
        else:
            self.unet = UNet(in_channels=beta_size + theta_size, out_channels=1, init_features=unet_init_features)

    def forward(self, beta, theta):
        # broadcast theta to the same size as beta - using a few tricks
        theta = theta.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, beta.shape[-2], beta.shape[-1])
        out = torch.cat((beta, theta), dim=1)
        out = self.unet(out)
        return out


