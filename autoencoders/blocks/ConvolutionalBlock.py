import torch.nn as nn


# Convolutional Block
class ConvolutionalBlock(nn.Module):
    def __init__(self, params, dec=False):
        super(ConvolutionalBlock, self).__init__()
        
        conv_layers, prev_channel = [], params['input_channels']
        for k, hidden_channel in enumerate(params['hidden_channels']):
            if dec:
                conv_layers.append(nn.ConvTranspose1d(prev_channel, hidden_channel, params['kernels'][k]))
            else:
                conv_layers.append(nn.Conv1d(prev_channel, hidden_channel, params['kernels'][k]))
            conv_layers.append(nn.ReLU())
            prev_channel = hidden_channel
        
        # last layer without activation function
        if dec:
            conv_layers.append(nn.ConvTranspose1d(prev_channel, params['output_channels'], params['kernels'][-1]))
        else:
            conv_layers.append(nn.Conv1d(prev_channel, params['output_channels'], params['kernels'][-1]))

        self.convolutional_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        return self.convolutional_layers(x)