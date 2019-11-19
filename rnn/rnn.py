"""
An implementation of RNNs, LSTMs, and GRUs.
Author: Xander Song
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import random
import math
import pudb

__all__ = ['RNN', 'RNNCell', 'LSTM', 'LSTMCell', 'GRU', 'GRUCell']

class AbstractRNNCell(ABC, nn.Module):
    @abstractmethod
    def __init__(self, input_size, hidden_size, bias):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

    @abstractmethod
    def forward(self):
        pass

class AbstractRNN(ABC, nn.Module):
    @abstractmethod
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

    @abstractmethod
    def forward(self):
        pass

class RNN(AbstractRNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, X):
        pass
        
class RNNCell(AbstractRNNCell):
    def __init__(self, *args, **kwargs):
        self.nonlinearity = kwargs.pop('nonlinearity', 'tanh')
        super().__init__(*args, **kwargs)
        self.Wih = nn.Linear(self.input_size, self.hidden_size, bias=self.bias)
        self.Whh = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        if self.nonlinearity == 'tanh':
            self.activation = nn.Tanh()
        elif self.nonlinearity == 'relu':
            self.activation = F.relu()
        else:
            raise ValueError("Invalid nonlinearity.")

    def forward(self, input, hidden=None):
        if hidden is None:
            batch = input.shape[0]
            hidden = torch.zeros(batch, self.hidden_size)
        return self.activation(self.Wih(input) + self.Whh(hidden))
        
class LSTM(AbstractRNN):
    def __init__(self, *args, **kwargs):    
        super().__init__(*args, **kwargs)
        self.num_directions = 2 if bidirectional else 1
        if self.bidirectional:
            raise Exception("Bidirectional LSTM not currently supported.")
        else:
            # Unidirectional LSTM.
            self.layers = [LSTMCell(self.input_size, self.hidden_size,
                bias=self.bias)]
            self.layers.extend([
                LSTMCell(self.hidden_size, self.hidden_size, bias=self.bias)
                for _ in range(self.num_layers - 1)
            ])

    def forward(self, input_, hidden_layers=None):
        """
        Args:
            input_ (torch.tensor): Input tensor of dimension (seq_len,
                batch, input_size)
            hidden_layers (tuple): Tuple of torch tensors, each of
                dimension (num_layers * self.num_directions, batch,
                hidden_size), representing the initial hidden and cell
                states for each element in the batch. If hidden_layers
                is None, then the hidden and cell states are initiliazed
                to zero.
        """

        self.check_input(input_)
        seq_len, batch, _ = input_.shape

        if hidden_layers:
            self.check_hidden(hidden_layers)
        else:
            h0 = torch.zeros(self.num_layers * self.num_directions, batch,
                             self.hidden_size)
            c0 = torch.zeros(self.num_layers * self.num_directions, batch,
                             self.hidden_size)
            t = (h0, c0)

        if self.bidirectional:
            pass
        else:
            h, c = h0, c0
            output = list()
            for s in range(seq_len):
                h[0,:,:], c[0,:,:] = self.layers[0](input_[s,:,:], (h[0,:,:], c[0,:,:]))
                for i, layer in enumerate(self.layers[1:]):
                    i += 1
                    h[i,:,:], c[i,:,:] = layer(h[i-1,:,:], (h[i,:,:], c[i,:,:]))
                output.append(h[-1,:,:])
            return output

    def check_input(self, input_):
        """
        Raises a ValueError if input_ has unexpected batch or input_size
        dimensions.

        Args:
            input_ (torch.tensor): An input tensor of dimensions
                (seq_len, batch, input_size).

        Returns:
            None

        Raises:
            ValueError
        """

        seq_len, batch, input_size = input_.shape
        if input_size != self.input_size:
            raise ValueError("Incorrect input size.")
                
class LSTMCell(AbstractRNNCell):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden = self._init_hidden(self.hidden_size)
        self.linear_x = nn.Linear(self.input_size, 4 * self.hidden_size) #bias=self.bias)
        self.linear_h = nn.Linear(self.hidden_size, 4 * self.hidden_size,
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

class GRU(AbstractRNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class GRUCell(AbstractRNNCell):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def main():
    input_size = 10
    hidden_size = 5
    bias = True
    pudb.set_trace()
    rnn_cell = RNNCell(input_size, hidden_size, bias=bias)
    print("Done")




if __name__ == "__main__":
    main()
