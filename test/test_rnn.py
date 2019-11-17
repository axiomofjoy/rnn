"""
This script tests the RNN architectures defined in rnn.py against the
official PyTorch implementations.

Author: Xander Song
"""

import random
import numpy
import numpy.testing as np_test
import pytest
import torch
import torch.nn as nn
import rnn


NUM_TESTS = 10

@pytest.mark.parametrize("input_size,hidden_size,bias,batch",
    [
        (random.randrange(1, 256), random.randrange(1, 64),
         random.choice([True, False]), random.randrange(1, 128))
        for _ in range(NUM_TESTS)
    ]
)
def test_rnn_cell_forward(input_size, hidden_size, bias, batch):
    """
    A test for the forward method of the LSTMCell.
    """

    # Define Torch RNN Cell and my lstm_cell.
    torch_rnn_cell = nn.RNNCell(input_size, hidden_size, bias=bias)
    rnn_cell = rnn.RNNCell(input_size, hidden_size, bias=bias)

    rnn_cell.Wih._parameters['weight'] = \
        torch_rnn_cell._parameters['weight_ih']
    rnn_cell.Wih._parameters['bias'] = \
        torch_rnn_cell._parameters['bias_ih']
    rnn_cell.Whh._parameters['weight'] = \
        torch_rnn_cell._parameters['weight_hh']
    rnn_cell.Whh._parameters['bias'] = \
        torch_rnn_cell._parameters['bias_hh']

    # Test that output of forward matches the expected output.
    torch.manual_seed(0)
    input = torch.rand(batch, input_size)
    hidden = torch.rand(batch, hidden_size)
    h1 = rnn_cell(input, hidden)
    h1_ = torch_rnn_cell(input, hidden)
    
    np_test.assert_array_almost_equal(h1.detach().numpy(),
                                      h1_.detach().numpy())
    
@pytest.mark.parametrize("input_size,hidden_size,bias,batch",
    [
        (random.randrange(1, 256), random.randrange(1, 64),
         random.choice([True, False]), random.randrange(1, 128))
        for _ in range(NUM_TESTS)
    ]
)
def test_lstm_cell_forward(input_size, hidden_size, bias, batch):
    """
    A test for the forward method of the LSTMCell.
    """

    # Define Torch LSTM Cell and my lstm_cell.
    torch_lstm_cell = nn.LSTMCell(input_size, hidden_size, bias=bias)
    lstm_cell = rnn.LSTMCell(input_size, hidden_size, bias=bias)

    lstm_cell.linear_x._parameters['weight'] = \
        torch_lstm_cell._parameters['weight_ih']
    lstm_cell.linear_x._parameters['bias'] = \
        torch_lstm_cell._parameters['bias_ih']
    lstm_cell.linear_h._parameters['weight'] = \
        torch_lstm_cell._parameters['weight_hh']
    lstm_cell.linear_h._parameters['bias'] = \
        torch_lstm_cell._parameters['bias_hh']

    # Test that output of forward matches the expected output.
    torch.manual_seed(0)
    input_ = torch.rand(batch, input_size)
    h1, c1 = lstm_cell(input_)
    h1_, c1_ = torch_lstm_cell(input_)
    
    np_test.assert_array_almost_equal(h1.detach().numpy(),
                                      h1_.detach().numpy())
    np_test.assert_array_almost_equal(c1.detach().numpy(),
                                      c1_.detach().numpy())


#def test_lstm_forward(input_size, hidden_size, bias, batch, num_layers):
#    """
#    A test for the forward method of the LSTM.
#    """
#    pass
#    torch_lstm = nn.LSTM(input_size, hidden_size, bias, num_layers=num_layers)
#
