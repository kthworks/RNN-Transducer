# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 22:15:46 2022

@author: Taehyoung Kim
"""

from data.data_loader import AudioDataset, AudioDataLoader


train_dataset = AudioDataset(transcript_filepath = './data/transcripts.txt')
data_loader = AudioDataLoader(dataset = train_dataset,
    batch_size = 2)


for data in data_loader:
    
    inputs, targets, inputs_lengths, targets_lengths = data

    break


print('batch_size = 2')
print('input shape: ', inputs.shape)
print('target shape: ', targets.shape)
print('input length: ', inputs_lengths)
print('target length: ', targets_lengths)
