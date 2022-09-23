# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 21:20:37 2022

@author: Taehyoung Kim
"""

from model.PredictionNet import PredictionNet
from data.data_loader import AudioDataset, AudioDataLoader

import torch
import torch.nn as nn
import numpy as np


# Get input features from the dataloader
train_dataset = AudioDataset(transcript_filepath = './data/transcripts.txt')
train_loader = AudioDataLoader(dataset = train_dataset,
    batch_size = 5)


for data in train_loader:
     
    inputs, targets, inputs_lengths, targets_lengths = data

    inputs_lengths = torch.IntTensor(inputs_lengths)
    targets_lengths = torch.IntTensor(targets_lengths)
     
    break

vocab_size = 110

PredictionNet = PredictionNet(vocab_size=vocab_size)
out, hidden = PredictionNet(targets)


print('PredictionNet Test')
print('------------------------------------------')
print('Input_shape = \n', targets.shape, '\n'), 
print('Input_lengths = \n', targets_lengths, '\n')
print('Output_shape = n', out.shape)