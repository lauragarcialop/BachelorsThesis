import torch.nn as nn

from autoencoders.blocks.LinearBlock import LinearBlock
from autoencoders.blocks.RecurrentBlock import RecurrentBlock
from autoencoders.blocks.ConvolutionalBlock import ConvolutionalBlock
from autoencoders.blocks.MHATransformerBlock import TransformerBlock


# Decoder
class MyDecoder(nn.Module):
    def __init__(self, params):
        super(MyDecoder, self).__init__()

        self.linear_block = LinearBlock(params['linear_layers'])

        self.transformer = 'transformer_block' in params.keys()
        if self.transformer:
            self.transformer_block = TransformerBlock(params['transformer_block'])

        self.recurrent = 'recurrent_layers' in params.keys()
        if self.recurrent:
            self.recurrent_block = RecurrentBlock(params['recurrent_layers'], dec=True)
        
        self.convolutional = 'convolutional_layers' in params.keys()
        if self.convolutional:
            self.convolutional_block = ConvolutionalBlock(params['convolutional_layers'], dec=True)
    
    def forward(self, z):
        z = z.view((z.shape[0], -1))
        x_ = self.linear_block(z)

        if self.transformer:
            x = self.transformer_block(x)
    
        if self.recurrent:
            x_ = self.recurrent_block(x_)
        
        if self.convolutional:
            x_ = self.convolutional_block(x_.view((x_.shape[0], 1, -1)))
            x_ = x_.view(x_.shape[0], -1)

        return x_