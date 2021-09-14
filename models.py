import torch
import torch.nn as nn

class Model1(nn.Module):
    def __init__(self, n_input_dim, params):
        super(Model1, self).__init__()
        self.layer_1 = nn.Linear(n_input_dim, params['layer_1'])
        self.layer_2 = nn.Linear(params['layer_1'], params['layer_2'])
        self.layer_out = nn.Linear(params['layer_2'], 1)
        if params['activation'] == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=params['dropout'])
        self.batchnorm1 = nn.BatchNorm1d(params['layer_1'])
        self.batchnorm2 = nn.BatchNorm1d(params['layer_2'])

    def forward(self, inputs):
        x = self.activation(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.activation(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.sigmoid(self.layer_out(x))

        return x