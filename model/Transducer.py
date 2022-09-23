# Author: Taehyoung Kim
# Last updated: 2022-09-19
# Original paper: Sequence Transduction with Recurrent Neural Networks, Alex Graves, 2012 
# Link: https://arxiv.org/pdf/1211.3711.pdf
# Reference:
#   Sooftware: https://github.com/sooftware/kospeech/tree/latest/kospeech/models/rnnt    
#   Jiho Jeong: https://github.com/fd873630/RNN-Transducer/tree/c1158689ba23ccf11091107c69f52f1e28219b62



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from model.PredictionNet import PredictionNet
from model.TranscriptionNet import TranscriptionNet


class JointNet(nn.Module):
    '''
    Joint Network of RNN Transducer.
    
    Args: 
        - input_dim (int): dimension of input vector = (out_dim of TranscriptionNet + out_dim of PredictionNet = 1024)
        - hidden_dim (int, optional) : dimension of hidden layer (default: 512)
        - vocab_size (int, optional): output dimension of JointNet = (vocab size = 149)
    
    Input: 
        TranscriptionNet_output (torch.FloatTensor): (batch, seq_length, transcriptionNet_out_dim=512)
       
        PredictionNet_output (torch.FloatTensor): (batch, seq_length, PredictionNet_out_dim=512)
                   
    Output: 
        JointNet_output (torch.FloatTensor): (batch, seq_length, out_dim = vocab size = 149)
    '''
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        vocab_size = int,
        ):
        
        super(JointNet, self).__init__()
        
        self.forward_layer = nn.Linear(input_dim, hidden_dim, bias=True)
        self.tanh = nn.Tanh()
        self.project_layer = nn.Linear(hidden_dim, vocab_size, bias=True)
        
        
    def forward(self, transNet_out, predNet_out):
        
        # Calcualte All probability of combination (t,u) only during Training
        if transNet_out.dim() == 3 and predNet_out.dim() == 3:
            transNet_out = transNet_out.unsqueeze(2)
            predNet_out = predNet_out.unsqueeze(1)
        
            t = transNet_out.size(1)
            u = predNet_out.size(2)
        
            transNet_out = transNet_out.repeat([1,1,u,1])
            predNet_out = predNet_out.repeat([1,t,1,1])
        
        concat_out = torch.cat((transNet_out, predNet_out), dim = -1)
        
        outputs = self.forward_layer(concat_out)
        outputs = self.tanh(outputs)
        outputs = self.project_layer(outputs)
        
        return outputs
        

class RNNTransducer(nn.Module):
    def __init__(
        self, 
        transNet: TranscriptionNet,
        predNet: PredictionNet,
        vocab_size: int,
        output_dim: int = 1024,
        hidden_dim: int = 512,
        ):
        
        super(RNNTransducer,self).__init__()
        
        self.sos_id = 1
        
        self.transNet = transNet
        self.predNet = predNet
        self.jointNet = JointNet(vocab_size=vocab_size)
        
        
    def forward(
        self,
        inputs: Tensor,
        input_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        
        transNet_out, input_lengths = self.transNet(inputs, input_lengths)
        predNet_out, hidden = self.predNet(targets, target_lengths)
        jointNet_out = self.jointNet(transNet_out, predNet_out)
        
        return jointNet_out
    
    # Greedy search 
    @torch.no_grad()
    def decode(
        self,
        transNet_out: Tensor, 
        max_seq_len: int,
        ):
            
        token = torch.LongTensor([[self.sos_id]])
        # if torch.cuda.is_available(): token = token.cuda() 
        
        predicted_token = []
        hidden = None
        
        for i in range(max_seq_len):
            predNet_out, hidden = self.predNet(token, hidden_states=hidden)
            output = self.jointNet(transNet_out[i], predNet_out.squeeze())
            output = F.softmax(output, dim=0)
            pred = output.topk(1)[1]
            pred = int(pred.item())
            
            if pred != 0:
                predicted_token.append(pred)
                token = torch.LongTensor([[pred]])
            
        return torch.LongTensor(predicted_token)
    
    @torch.no_grad()
    def recognize(
        self,
        inputs: Tensor,
        input_lengths: Tensor,
        )-> Tensor:
        
        outputs = []
        
        transNet_out, input_lengths = self.transNet(inputs, input_lengths)
        
        max_seq_len = transNet_out.size(1) # for stacking
        
        for output_ in transNet_out:
            output = self.decode(output_, max_seq_len)
            outputs.append(output)
            
            print(len(outputs))
            
        outputs = torch.stack(outputs, dim=0)
        return output