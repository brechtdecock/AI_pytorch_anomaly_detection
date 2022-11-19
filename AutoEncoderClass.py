import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as Functional
import torch
from torch.utils.data import DataLoader

class AutoEncoder(nn.Module):  #1 second of waveform has 16 000 inputs
    def __init__(self, length):
        super().__init__()

        self.encoder = torch.nn.Sequential(

            nn.Linear(length,2000),
            #nn.BatchNorm1d
            nn.Tanh(), #mss tanh om altijd terug te normaliseren van -1 naar 1
            nn.Linear(2000, 1000),
            nn.Tanh(),
            nn.Linear(1000, 500),
            nn.Tanh(),
            nn.Linear(500, 250),
            nn.Dropout(0.1)
           #no function after this???
        )
         
        self.decoder = torch.nn.Sequential(
            nn.Linear(250, 500),
            nn.Tanh(),
            nn.Linear(500, 1000),
            nn.Tanh(),
            nn.Linear(1000, 2000),
            nn.Tanh(),
            nn.Linear(2000, length)
            #nn.Sigmoid()#for value between 0 and 1
        )
 
    def forward(self, data):
        encoded = self.encoder(data)
        decoded = self.decoder(encoded)
        return decoded
  