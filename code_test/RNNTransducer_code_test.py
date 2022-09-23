# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 02:41:29 2022

@author: Taehyoung Kim
"""

from model.TranscriptionNet import TranscriptionNet
from model.PredictionNet import PredictionNet
from model.Transducer import JointNet
from model.Transducer import RNNTransducer
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

RNNTransducer = RNNTransducer(TranscriptionNet, PredictionNet, vocab_size)
predict = RNNTransducer(inputs, inputs_lengths, targets, targets_lengths)


print('RNNTransducer Test')
print('------------------------------------------')
print('Output_shape = n', predict.shape)