import torch.nn as nn


# Recurrent Block
class RecurrentBlock(nn.Module):
    def __init__(self, params, dec=False):
        super(RecurrentBlock, self).__init__()
        
        rec_layers, prev_size = [], params['input_size']
        for hidden_size in params['hidden_sizes']:
            rec_layers.append(nn.RNN(prev_size, hidden_size))
            prev_size = hidden_size
        self.recurrent_layers = nn.Sequential(*rec_layers)
        
        self.dec = dec
        if self.dec:
            self.last_layer = nn.Linear(prev_size, params['output_size'])
        else:
            self.last_layer = nn.RNN(prev_size, params['output_size'])

    def forward(self, x):
        for rnn in self.recurrent_layers:
            x, _ = rnn(x)
        
        if self.dec:
            x = self.last_layer(x)
        else:
            x, _ = self.last_layer(x)
        
        return x