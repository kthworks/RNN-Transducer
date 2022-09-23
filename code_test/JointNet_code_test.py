# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 22:43:37 2022

@author: Taehyoung Kim
"""
from model.TranscriptionNet import TranscriptionNet
from model.PredictionNet import PredictionNet
from model.Transducer import JointNet
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


input_dim = 80
vocab_size = 110

TranscriptionNet = TranscriptionNet(input_dim=input_dim)
PredictionNet = PredictionNet(vocab_size=vocab_size)

transNet_out, input_lengths = TranscriptionNet(inputs, inputs_lengths)
predNet_out, hidden = PredictionNet(targets)

JointNet = JointNet(vocab_size=vocab_size)
output = JointNet(transNet_out, predNet_out)


print('JointNet Test')
print('------------------------------------------')
print('transNet_out_shape = \n', transNet_out.shape, '\n'), 
print('predNet_out_shape = \n', predNet_out.shape, '\n'), 
print('Output_shape = n', output.shape)