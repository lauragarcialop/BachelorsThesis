import torch.nn as nn

from autoencoders.Encoder import MyEncoder
from autoencoders.Decoder import MyDecoder


# Autoencoder
class MyAutoencoder(nn.Module):
    def __init__(self, params):
        super(MyAutoencoder, self).__init__()

        self.encoder = MyEncoder(params['encoder'])
        self.decoder = MyDecoder(params['decoder'])

    
    def forward(self, x):
        z = self.encoder(x)
        x_ = self.decoder(z)
        return x_
    

    def encode(self, x):
        return self.encoder(x)
    

    def decode(self, z):
        return self.decoder(z)