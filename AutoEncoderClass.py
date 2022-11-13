import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as Functional
import torch
from torch.utils.data import DataLoader

class AutoEncoder(nn.Module):  #1 second of waveform has 16 000 inputs
    def __init__(self, length):
        super().__init__()

        self.encoder = torch.nn.Sequential(

            nn.Linear(length,1000),
            nn.ReLU(),
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, 20),
            #nn.Dropout(0.1)
           #no function after this???
        )
         
        self.decoder = torch.nn.Sequential(
            nn.Linear(20, 100),
            nn.ReLU(),
            nn.Linear(100, 1000),
            nn.ReLU(),
            nn.Linear(1000, length),
            nn.Sigmoid()#for value between 0 and 1
        )
 
    def forward(self, data):
        encoded = self.encoder(data)
        decoded = self.decoder(encoded)
        return decoded
    
    
