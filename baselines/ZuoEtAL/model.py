import torch
import torch.nn as nn
import numpy as np
import os, sys
from models.generic_unet import UNet

class ReshapeTheta(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return x


class BetaEncoder(nn.Module):
    def __init__(self, unet_init_features=4,
                 temperature=1, beta_size=5,):
        super().__init__()

        # Gumbel Softmax variables
        self.temperature = temperature
        self.Gumbel = torch.distributions.gumbel.Gumbel(0.0, 1.0)
        self.sm = nn.Softmax2d()

        self.unet = UNet(in_channels=1, out_channels=beta_size, init_features=unet_init_features)


    def GumbelSoftmax(self, x):
        G = self.Gumbel.rsample(sample_shape=x.shape)
        x = x + G.to(x.device)
        x = x / self.temperature
        return self.sm(x)


    def forward(self, img):
        out = self.unet(img)
        return out

class ThetaEncoder(nn.Module):
    def __init__(self, theta_size, img_size, kernel_size=3):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        in_shapes = [1, 2, 4, 16]
        out_shapes = [2, 4, 16, 32]
        size_out = list(img_size)
        for in_shape, out_shape in zip(in_shapes, out_shapes):
            self.layers.append(nn.Conv2d(in_shape, out_shape, kernel_size, padding=0))
            self.layers.append(nn.ReLU())
            size_out = [int((s - kernel_size) + 1) for s in size_out]
        self.layers.append(nn.MaxPool2d(kernel_size))
        size_out = [int((s - kernel_size) / kernel_size + 1) for s in size_out]
        self.layers.append(ReshapeTheta())
        self.layers.append(nn.Linear(np.prod(size_out) * out_shapes[-1], 32))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(32, 16))
        self.layers.append(nn.ReLU())
        self.mu_layer = nn.Linear(16, theta_size)
        self.var_layer = nn.Linear(16, theta_size)

    def forward(self, img):
        out = img
        for layer in self.layers:
            out = layer(out)
        mu = self.mu_layer(out)
        var = self.var_layer(out)
        return mu, var

class BetaDiscriminant(nn.Module):
    def __init__(self, beta_size, img_size, kernel_size):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        size_out = list(img_size)
        for in_shape, out_shape in zip([beta_size, 8, 8, 4], [8, 8, 4, 2]):
            self.layers.append(nn.Conv2d(in_shape, out_shape, kernel_size, padding=0))
            self.layers.append(nn.ReLU())
            size_out = [int((s - kernel_size) + 1) for s in size_out]
        self.layers.append(nn.MaxPool2d(kernel_size))
        size_out = [int((s - kernel_size) / kernel_size + 1) for s in size_out]
        self.layers.append(ReshapeTheta())
        self.layers.append(nn.Linear(np.prod(size_out) * 2, 16))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(16, 8))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(8, 2))

    def forward(self, beta):
        out = beta
        for layer in self.layers:
            out = layer(out)
        return out


class Decoder(nn.Module):
    def __init__(self, unet_init_features, beta_size=5, theta_size=2):
        super().__init__()

        # let's do  a four layer conv with leaky relu activations - super simple
        self.unet = UNet(in_channels=beta_size + theta_size, out_channels=1, init_features=unet_init_features)

    def forward(self, beta, theta):
        # broadcast theta to the same size as beta - using a few tricks
        theta = theta.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, beta.shape[-2], beta.shape[-1])
        out = torch.cat((beta, theta), dim=1)
        out = self.unet(out)
        return out


