# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 22:36:59 2022

@author: Taehyoung Kim
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 21:20:37 2022

@author: Taehyoung Kim
"""

from model.TranscriptionNet import TranscriptionNet
from data.data_loader import AudioDataset, AudioDataLoader

import torch

# Get input features from the dataloader
train_dataset = AudioDataset(transcript_filepath = './data/transcripts.txt')
train_loader = AudioDataLoader(dataset = train_dataset,
    batch_size = 5)


for data in train_loader:
     
    inputs, targets, inputs_lengths, targets_lengths = data

    inputs_lengths = torch.IntTensor(inputs_lengths)
    targets_lengths = torch.IntTensor(targets_lengths)
     
    break


input_dim = 80


TranscriptionNet = TranscriptionNet(input_dim=input_dim)
out, input_lengths = TranscriptionNet(inputs, inputs_lengths)


print('TranscriptionNet Test')
print('------------------------------------------')
print('Input_shape = \n', inputs.shape, '\n'), 
print('Input_lengths = \n', inputs_lengths, '\n')
print('Output_shape = n', out.shape)
