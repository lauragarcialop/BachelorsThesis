import torch
import torch.nn as nn

from codes.autoencoders.Encoder import MyEncoder
from codes.autoencoders.Decoder import MyDecoder
    

# Autoencoder
class MyVariationalAutoencoder(nn.Module):
    def __init__(self, params):
        super(MyVariationalAutoencoder, self).__init__()

        self.encoder = MyEncoder(params)
        self.decoder = MyDecoder(params)

    
    def forward(self, x):
        mu, std = self.encoder(x)
        z = self.sample(mu, std)
        x_ = self.decoder(z)
        return x_
    

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def encode(self, x):
        mu, std = self.encoder(x)
        return self.sample(mu, std)
    

    def decode(self, z):
        return self.decoder(z)