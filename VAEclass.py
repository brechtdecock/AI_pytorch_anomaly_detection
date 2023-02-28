import torch
import torch.nn as nn


class LinearVAE(nn.Module):  #0.5 second of waveform has 8 000 inputs
    def __init__(self,length):
        super().__init__()
    
        self.KL_Loss = 0
        
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
        
        
        self.delin1 = nn.Sequential(nn.Linear(25,100),
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
        log_var = self.z_log_var(out_50_enc)

        z = self.reparameterize(mu, log_var)
        
        
        #self.KL_Loss = torch.sum(sigma**2 + mu**2 - torch.log(sigma) - 1/2)
        
        self.KL_Loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp()))
        
        # decoding
        # out_100_dec = self.delin1(z)
        # out_250_dec = self.delin2(out_100_dec)
        # out_500_dec = self.delin3(out_250_dec)
        # out_1000_dec =  self.delin4(out_500_dec)
        # out_2000_dec =  self.delin5(out_1000_dec)
        # out_4000_dec =  self.delin6(out_2000_dec)
        # out_8000_dec = self.delin7(out_4000_dec)
        out_100_dec = out_100_enc+ self.delin1(z)
        out_250_dec = out_250_enc+ self.delin2(out_100_dec)
        out_500_dec = out_500_enc+ self.delin3(out_250_dec)
        out_1000_dec = out_1000_enc+ self.delin4(out_500_dec)
        out_2000_dec = out_2000_enc+ self.delin5(out_1000_dec)
        out_4000_dec = out_4000_enc+ self.delin6(out_2000_dec)
        out_8000_dec = data + self.delin7(out_4000_dec)
        
        return out_8000_dec
    



class ConvolutedVAE(nn.Module):  #0.5 second of waveform has 8 000 inputs
    def __init__(self):
        super().__init__()
        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.KL_Loss = 0
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size=(3,1), stride=1, padding="same"),
                                    nn.Tanh(), #default neg slope of 0.01
                                    nn.MaxPool2d(kernel_size=(2,1), stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels = 32,out_channels = 64, kernel_size=(3,1), stride=1, padding="same"),
                                   nn.Tanh(),
                                   nn.MaxPool2d(kernel_size=(2,1), stride=2))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels = 64,out_channels = 128, kernel_size=(3,1), stride=1, padding="same"),
                                   nn.Tanh(),
                                   nn.MaxPool2d(kernel_size=(2,1), stride=2))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels = 128,out_channels = 256, kernel_size=(3,1), stride=1, padding="same"),
                                   nn.Tanh(),
                                   nn.MaxPool2d(kernel_size=(2,1), stride=2))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels = 256,out_channels = 512, kernel_size=(3,1), stride=1, padding="same"),
                                   nn.Tanh(),
                                   nn.MaxPool2d(kernel_size=(2,1), stride=2))
        
        self.z_mean = nn.Linear(512*250*1,256)
        self.z_log_var = nn.Linear(512*250*1,256)
        

        # Initializing the fully-connected layer and 2 convolutional layers for decoder
        self.delin = nn.Sequential(nn.Linear(256, 512*250*1),
                                   nn.Tanh()
                                    )   
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size = (2,1), stride=2),
                                     nn.Tanh()
                                     )
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size = (2,1), stride=2),
                                     nn.Tanh()
                                     )
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size = (2,1), stride=2),
                                     nn.Tanh()
                                     )
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size = (2,1), stride=2),
                                     nn.Tanh()
                                     )
        self.deconv5 = nn.Sequential(nn.ConvTranspose2d(32, 1, kernel_size = (2,1), stride=2),
                                     nn.Tanh()
                                     )
        

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps


    def forward(self, data):
        #encoder
        enc_32 = self.conv1(data)
        enc_64 = self.conv2(enc_32)
        enc_128= self.conv3(enc_64)
        enc_256 = self.conv4(enc_128)
        enc_512 = self.conv5(enc_256)
        
        z_lin = enc_512.view(-1, 512*250*1)
        
        mu = self.z_mean(z_lin)
        log_var = self.z_log_var(z_lin)
        
        z_out = self.reparameterize(mu, log_var)
        self.KL_Loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp()))

        #decoder
        x = self.delin(z_out)
        dec_512 = x.view(-1, 512, 250, 1)
        dec_256 = self.deconv1(dec_512)
        dec_128 = self.deconv2(dec_256)
        dec_64 = self.deconv3(dec_128)
        dec_32 = self.deconv4(dec_64)
        dec_1 = self.deconv5(dec_32)

        return dec_1 #back to (-1,1,8000,1)
         
    
    
    
    


