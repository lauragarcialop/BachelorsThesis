import torch.nn as nn
from autoencoders.blocks.LinearBlock import LinearBlock
from autoencoders.blocks.RecurrentBlock import RecurrentBlock
from autoencoders.blocks.ConvolutionalBlock import ConvolutionalBlock
from autoencoders.blocks.MHATransformerBlock import TransformerBlock


# Encoder
class MyEncoder(nn.Module):
    def __init__(self, params):
        super(MyEncoder, self).__init__()

        self.transformer = 'transformer_block' in params.keys()
        if self.transformer:
            self.transformer_block = TransformerBlock(params['transformer_block'])

        self.recurrent = 'recurrent_layers' in params.keys()
        if self.recurrent:
            self.recurrent_block = RecurrentBlock(params['recurrent_layers'])
        
        self.convolutional = 'convolutional_layers' in params.keys()
        if self.convolutional:
            self.convolutional_block = ConvolutionalBlock(params['convolutional_layers'])
        
        self.linear_block = LinearBlock(params['linear_layers'])
    
    def forward(self, x):
        if self.transformer:
            x = self.transformer_block(x)
            
        if self.recurrent:
            x = self.recurrent_block(x)
        
        if self.convolutional:
            x = x.view((x.shape[0], 1, -1))
            x = self.convolutional_block(x)
        
        x = x.view(x.shape[0], -1)
        return self.linear_block(x)