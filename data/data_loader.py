# Author: Taehyoung Kim
# Last updated: 2022-09-17
# Original paper: Sequence Transduction with Recurrent Neural Networks, Alex Graves, 2012 
# Link: https://arxiv.org/pdf/1211.3711.pdf
# Reference:
#   Sooftware: https://github.com/sooftware/kospeech/tree/latest/kospeech/models/rnnt    
#   Jiho Jeong: https://github.com/fd873630/RNN-Transducer/tree/c1158689ba23ccf11091107c69f52f1e28219b62    


import torch
import pandas as pd
import scipy.signal
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader

PAD = 0
data_path = 'C:\Kosponspeech_train'

def make_vocab():
    try:
        vocab_file = pd.read_csv('./aihub_labels.csv')
        
        idx2grapheme = {}
        grapheme2idx = {}
        for i,j in zip(vocab_file['id'], vocab_file['grpm']):
            idx2grapheme[i] = j
            grapheme2idx[j] = i
       
        return idx2grapheme, grapheme2idx
   
    except:
        print("There is no 'aihub_labels.csv' file in the data folder!")
        print("You should preprocess the dataset first.")
    
              
class AudioParser():
    def __init__(self):
        super(AudioParser, self).__init__()
        self.window_size = 0.02
        self.window_stride = 0.01
        self.window_function = scipy.signal.hamming
        self.sample_rate = 16000
        
    def parse_audio(self, audio_path):
        with open(audio_path, 'rb') as opened_pcm_file:
            
            buf = opened_pcm_file.read()
            pcm_data = np.frombuffer(buf, dtype = 'int16')
            wav_data = librosa.util.buf_to_float(pcm_data)
        
        n_fft = int(self.sample_rate * self.window_size)
        win_length = n_fft
        hop_length = int(self.sample_rate * self.window_stride)
         
        # log_mel spec
        D = np.abs(librosa.stft(wav_data, n_fft=n_fft, hop_length=hop_length,
                win_length=win_length, window=self.window_function))
        
        mel_spec = librosa.feature.melspectrogram(S=D, sr=self.sample_rate, n_mels=80, hop_length=hop_length, win_length=win_length)
        mel_spec = np.log1p(mel_spec)
        spect = torch.FloatTensor(mel_spec)
        
        return spect

    
class AudioDataset(Dataset, AudioParser):
    def __init__(self, transcript_filepath):
        with open(transcript_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split('\t') for x in ids]

        self.ids = ids
        self.size = len(ids)
        
        super(AudioDataset, self).__init__()
        
        
    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path = data_path + '\\' + sample[0]
        target = sample[2]
        target = torch.LongTensor([int(t) for t in target.split(' ')])
        
        spect = self.parse_audio(audio_path)
        spect = torch.transpose(spect, 0, 1)
        
        return spect, target
        
    def __len__(self):
        return self.size
    

class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
        
def _collate_fn(batch):
    
    def seq_length_(p):
        return len(p[0])
    
    def target_length_(p):
        return len(p[1])
    
    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]
    
    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)
    
    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)

    targets = torch.zeros(batch_size, max_target_size).to(torch.long) # +2 for <sos> and <eos>
    targets.fill_(PAD)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        
        seq_length = tensor.size(0)
        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(target)
    
    idx = sorted(range(len(seq_lengths)), key=lambda k: seq_lengths[k])
    seq_lengths = sorted(seq_lengths, reverse=True)
    seqs = seqs[idx]
    
    idx = sorted(range(len(target_lengths)), key=lambda k: target_lengths[k])
    target_lengths = sorted(target_lengths, reverse=True)
    targets = targets[idx]
    
    return seqs, targets, seq_lengths, target_lengths