"""
An implementation of GRU and LSTM.
Author: Xander Song
"""

import torch
import torch.nn as nn
import random
import math
import pudb

class LSTM(nn.Module):
    pass

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.hidden = self._init_hidden(hidden_size)
        self.linear_x = nn.Linear(input_size, 4 * hidden_size,
                                  bias=self.bias)
        self.linear_h = nn.Linear(hidden_size, 4 * hidden_size,
                                  bias=self.bias)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def _init_hidden(self, hidden_size):
        """
        Initializes the hidden state of the LSTM.

        The elements of the hidden state are initialized using
        Args:
            hidden_size (int): Size of hidden layer.

        Returns:
            hidden (torch.tensor): An initialized 
        """
        k = 1 / hidden_size
        sqrt_k = math.sqrt(k)
        hidden = 2 * sqrt_k * torch.rand(hidden_size) + sqrt_k
        return hidden

    def forward(self, input_, hidden_layers=None):
        """
        Forward pass of LSTM cell.

        Args:
            input (torch.tensor): An input tensor of shape (batch,
                input_size).
            hidden_layers (tuple): A tuple of torch tensors, each of
                dimensions (batch, input_size). The first tensor is the
                hidden state and the second is the cell state. If no
                argument is passed, then both hidden and cell states are
                initialized to 0.
        
        Returns:
             
        """
        
        # Check input size.
        if input_.shape[1] != self.input_size:
            raise ValueError("Unexpected shape for input.")

        if hidden_layers:
            h0, c0 = hidden_layers
        else:
            batch = input_.shape[0]
            h0 = torch.zeros(batch, self.hidden_size)
            c0 = torch.zeros(batch, self.hidden_size)

        W = self.linear_x(input_) + self.linear_h(h0)
        i, f, g, o = torch.chunk(W, 4, dim=1)
        i = self.sigmoid(i)
        f = self.sigmoid(f)
        g = self.tanh(g)
        o = self.sigmoid(o)
        c1 = f * c0 + i * g 
        h1 = o * self.tanh(c1)
        return (h1, c1)


class GRU(nn.Module):
    pass

class GRUCell(nn.Module):
    pass


def main():
    torch.manual_seed(0)
    input_size = 256
    hidden_size = 64
    bias = True
    batch = 32

    # Torch LSTM Cell.
    torch_lstm_cell = nn.LSTMCell(input_size, hidden_size, bias=bias)
    lstm_cell = LSTMCell(input_size, hidden_size, bias=bias)

    lstm_cell.linear_x._parameters['weight'] = \
        torch_lstm_cell._parameters['weight_ih']
    lstm_cell.linear_x._parameters['bias'] = \
        torch_lstm_cell._parameters['bias_ih']
    lstm_cell.linear_h._parameters['weight'] = \
        torch_lstm_cell._parameters['weight_hh']
    lstm_cell.linear_h._parameters['bias'] = \
        torch_lstm_cell._parameters['bias_hh']

    import numpy
    import numpy.testing as np_test

    input_ = torch.rand(batch, input_size)
    h1, c1 = lstm_cell(input_)
    h1_, c1_ = torch_lstm_cell(input_)
    
    np_test.assert_array_almost_equal(h1.detach().numpy(),
                                      h1_.detach().numpy())
    np_test.assert_array_almost_equal(c1.detach().numpy(),
                                      c1_.detach().numpy())
    print("Done")




if __name__ == "__main__":
    main()
