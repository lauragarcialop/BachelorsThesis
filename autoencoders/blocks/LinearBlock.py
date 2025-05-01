import torch.nn as nn


# Linear Block
class LinearBlock(nn.Module):
    def __init__(self, params):
        super(LinearBlock, self).__init__()
        hidden_layers = params['hidden_layers']
        
        linear_layers, prev_dim = [], params['input_dim']
        for hidden_dim in hidden_layers:
            linear_layers.append(nn.Linear(prev_dim, hidden_dim))
            linear_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # last layer without activation function
        linear_layers.append(nn.Linear(prev_dim, params['output_dim']))

        self.linear = nn.Sequential(*linear_layers)

    def forward(self, x):
        return self.linear(x)