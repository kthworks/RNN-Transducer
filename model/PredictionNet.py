# Author: Taehyoung Kim
# Last updated: 2022-09-16
# Original paper: Sequence Transduction with Recurrent Neural Networks, Alex Graves, 2012 
# Link: https://arxiv.org/pdf/1211.3711.pdf
# Reference:
#   Sooftware: https://github.com/sooftware/kospeech/tree/latest/kospeech/models/rnnt    
#   Jiho Jeong: https://github.com/fd873630/RNN-Transducer/tree/c1158689ba23ccf11091107c69f52f1e28219b62    
#   upskyy: https://github.com/upskyy/RNN-Transducer/tree/main/rnn_transducer

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PredictionNet(nn.Module):
    '''
    Prediction Network of RNN Transducer.
    
    Args: 
        - vocab_size (int) : size of vocabulary (number of classes)
        - hidden_dim (int, optional) : dimension of hidden layer (default: 512)
        - output_dim (int, optional): output dimension of PredictionNet (default: 512)
        - num_layers (int, optional): number of layers (default: 1)
        - dp (float, optional): dropout probability (default: 0.2)
    
    Input: 
        Batch of y (torch.LongTensor): Index of y_onehot = (y1, ... , yu). size: (batch, max_seq_length)
            - Zero-padded indexes based on maximum sequence length in the batch.
            - Each rows are sorted by sequence length. (To put it into pack_padded_sequence)
            - For example:
                y = tensor([[3, 2, 1, 3], [2, 1, 3, 0]]). size: (batch=2, max_seq_length=4)
        
        input_lengths (torch.LongTensor): The length of input tensor. size = (batch)
            - for example: [4, 3]
            
        hidden_states (torch.FloatTensor): A previous hidden state. size = (batch, seq_length, hidden_dim)
        
    Output: 
        Batch of g (torch.FloatTensor): g = (g0, g1, ... ,  gu). size = (batch, seq_length, out_dim)
    '''
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 512,
        output_dim: int = 512, 
        num_layers: int = 1, 
        dp: float = 0.2,
        ):
               
        super(PredictionNet, self).__init__()
        
        self.embedded =  nn.Embedding(vocab_size, hidden_dim, padding_idx=0) 
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            bias=True,
            batch_first=True,
            dropout=dp,
        )
        
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(
        self,
        inputs: Tensor,
        input_lengths: Tensor = None,
        hidden_states: Tensor = None,
    )-> Tuple[Tensor, Tensor]:
        
        embedded = self.embedded(inputs) 

        if input_lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
            outputs, hidden_states = self.lstm(embedded, hidden_states)
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, hidden_states = self.lstm(embedded, hidden_states)
        
        outputs = self.out_proj(outputs)
        
        return outputs, hidden_states
