import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
  def __init__(self, features, c):
    super(Encoder, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=c, kernel_size=4, stride=2, padding=1) # out: c x 14 x 14
    self.conv2 = nn.Conv2d(in_channels=c, out_channels=c*2, kernel_size=4, stride=2, padding=1) # out: c x 7 x 7
    self.fc_mu = nn.Linear(in_features=c*2*3*3, out_features=features)
    self.fc_logvar = nn.Linear(in_features=c*2*3*3, out_features=features)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    #print("conv1: ",x.shape)
    x = F.relu(self.conv2(x))
    #print("conv2: ",x.shape)
    x = x.view(x.size(0), -1)
    #print(x.shape) # flatten batch of multi-channel feature maps to a batch of feature vectors
    x_mu = self.fc_mu(x)
    x_logvar = self.fc_logvar(x)

    return x_mu, x_logvar


class Decoder(nn.Module):
  def __init__(self, features, c):
    super(Decoder, self).__init__()
    self.c = c
    self.features = features
    self.fc = nn.Linear(in_features=features, out_features=c*2*3*3)
    self.deconv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=5, stride=2, padding=1)
    self.deconv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)


  def forward(self, z):
    x = self.fc(z)
    x = x.view(x.size(0), self.c*2, 3, 3)
    #print(x.shape) # unflatten batch of feature vectors to a batch of multi-channel feature maps
    x = F.relu(self.deconv2(x))
    #print("deconv2: ",x.shape)
    reconstruction = torch.sigmoid(self.deconv1(x))

    return reconstruction


class VAE(nn.Module):
  def __init__(self, features, c):
    super(VAE, self).__init__()

    self.features = features
    self.c = c

    self.encoder = Encoder(features, c)

    self.decoder = Decoder(features, c)

  def reparamterize(self, mu, logvar):
    if self.training:
        # the reparameterization trick
        std = logvar.mul(0.5).exp_()
        eps = torch.empty_like(std).normal_()
        return eps.mul(std).add_(mu)
    else:
        return mu

  def forward(self, x):

    x_mu, x_logvar = self.encoder.forward(x)

    # print(x_mu.shape)
    z = self.reparamterize(x_mu, x_logvar)

    # print(z.shape)
    reconstruction = self.decoder.forward(z)


    return reconstruction, x_mu, x_logvar