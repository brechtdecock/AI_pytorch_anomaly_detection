import torch
import torch.nn as nn


class LinearVAE(nn.Module):  #0.5 second of waveform has 8 000 inputs
    def __init__(self,length):
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
        
        #not sequential!
        self.z_mean = nn.Linear(50,25)
        self.z_log_var = nn.Linear(50,25)
        
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
        
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
 
    def forward(self, data):
        # encoding
        out_4000_enc = self.lin1(data)
        out_2000_enc = self.lin2(out_4000_enc)
        out_1000_enc = self.lin3(out_2000_enc)
        out_500_enc = self.lin4(out_1000_enc)
        out_250_enc = self.lin5(out_500_enc)
        out_100_enc = self.lin6(out_250_enc)
        out_50_enc = self.lin7(out_100_enc)
        
        """
        dimensions of latent spaces mu and log_var?
        """
        mu = self.z_mean(out_50_enc)
        logvar = self.z_log_var(out_50_enc)
    
        z = self.reparameterize(mu, logvar)
 
        # decoding
        out_100_dec = self.delin1(z)
        out_250_dec = self.delin2(out_100_dec)
        out_500_dec = self.delin3(out_250_dec)
        out_1000_dec =  self.delin4(out_500_dec)
        out_2000_dec =  self.delin5(out_1000_dec)
        out_4000_dec =  self.delin6(out_2000_dec)
        out_8000_dec = self.delin7(out_4000_dec)
        
        return out_8000_dec, mu, logvar
    

class ConvolutedVAE(nn.Module):  #0.5 second of waveform has 8 000 inputs
    def __init__(self):
        super().__init__()
        
        
         
        
    def forward(self, data):
        pass
        
        return 