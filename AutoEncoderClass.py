import torch.nn as nn
from torch.autograd import Variable as V
#import torch.nn.functional as Functional
import torch
#from torch.utils.data import DataLoader

class AutoEncoder(nn.Module):  #0.5 second of waveform has 16 000 inputs
    def __init__(self, length):
        super().__init__()

        
        self.lin1 = nn.Sequential(nn.Linear(length,4000),
                                  nn.Tanh())
        self.lin2 = nn.Sequential(nn.Linear(4000,2000),
                                  nn.Tanh())
        self.lin3 = nn.Sequential(nn.Linear(2000,1000),
                                  nn.Tanh())
        self.lin4 = nn.Sequential(nn.Linear(1000,500),
                                  nn.Tanh())
        self.lin5 = nn.Sequential(nn.Linear(500,250),
                                  nn.Tanh())
        self.lin6 = nn.Sequential(nn.Linear(250,100),
                                  nn.Tanh())
        self.lin7 = nn.Sequential(nn.Linear(100,50),
                                  nn.Tanh())
        
        self.drop = nn.Dropout(0.1)
        
        self.delin1 = nn.Sequential(nn.Linear(50,100),
                                  nn.Tanh())
        self.delin2 = nn.Sequential(nn.Linear(100,250),
                                  nn.Tanh())
        self.delin3 = nn.Sequential(nn.Linear(250,500),
                                  nn.Tanh())
        self.delin4 = nn.Sequential(nn.Linear(500,1000),
                                  nn.Tanh())
        self.delin5 = nn.Sequential(nn.Linear(1000,2000),
                                  nn.Tanh())
        self.delin6 = nn.Sequential(nn.Linear(2000,4000),
                                  nn.Tanh())
        self.delin7 = nn.Sequential(nn.Linear(4000,length),
                                  nn.Tanh())
        
        
 
    def forward(self, data): #data is 8000 long
        out_4000_enc = self.lin1(data)
        out_2000_enc = self.lin2(out_4000_enc)
        out_1000_enc = self.lin3(out_2000_enc)
        out_500_enc = self.lin4(out_1000_enc)
        out_250_enc = self.lin5(out_500_enc)
        out_100_enc = self.lin6(out_250_enc)
        out_50_enc = self.lin7(out_100_enc)
        
        out_drop = self.drop(out_50_enc)
        
        #with skip connections implemented: add the data from the layers of the encoder
        # out_100_dec = out_100_enc+ self.delin1(out_drop)
        # out_250_dec = out_250_enc+ self.delin2(out_100_dec)
        # out_500_dec = out_500_enc+ self.delin3(out_250_dec)
        # out_1000_dec = out_1000_enc+ self.delin4(out_500_dec)
        # out_2000_dec = out_2000_enc+ self.delin5(out_1000_dec)
        # out_4000_dec = out_4000_enc+ self.delin6(out_2000_dec)
        # out_8000_dec = data + self.delin7(out_4000_dec)
        
        out_100_dec =self.delin1(out_drop)
        out_250_dec =self.delin2(out_100_dec)
        out_500_dec =self.delin3(out_250_dec)
        out_1000_dec = self.delin4(out_500_dec)
        out_2000_dec = self.delin5(out_1000_dec)
        out_4000_dec = self.delin6(out_2000_dec)
        out_8000_dec =  self.delin7(out_4000_dec)

        return out_8000_dec
  