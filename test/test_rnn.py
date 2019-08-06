"""
This script tests the RNN architectures defined in rnn.py.

Author: Xander Song
"""

import rnn
import pytest
import numpy
import numpy.testing as np_test


def test_lstm_cell_forward(input_size, hidden_size, bias, batch):
    input_size = 256
    hidden_size = 64
    bias = True
    batch = 32

    # Define Torch LSTM Cell and my lstm_cell.
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

    # Test that output of 
    torch.manual_seed(0)
    input_ = torch.rand(batch, input_size)
    h1, c1 = lstm_cell(input_)
    h1_, c1_ = torch_lstm_cell(input_)
    
    np_test.assert_array_almost_equal(h1.detach().numpy(),
                                      h1_.detach().numpy())
    np_test.assert_array_almost_equal(c1.detach().numpy(),
                                      c1_.detach().numpy())


