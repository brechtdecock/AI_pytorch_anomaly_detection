import torch
import torch.nn as nn

class Trim(nn.Module): #dont know if necessary but just in case
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :8000, :]

class CAE(nn.Module):  #0.5 second of waveform has 8 000 inputs
    def __init__(self):
        super().__init__()
        """
        use 1D components or use 2D components with [length,1] as input and [3,1] as kernel?
        
        often relu is used an only sigmoid(tanh in my case) at the end==better??? doesn't seem like it
        
        use tanh at the end but use a Relu in between
        """
        
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
        
        self.drop  = nn.Dropout(0.1)
        
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
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
        
        
        #(batch size, channels, x-dim,  y-dim)
        # vector in shape=(?, 1, 8000, 1)
        #    Conv     -> (?, 32, 8000, 1)
        #    Pool     -> (?, 32, 4000, 1)
        
        #    shape =     (?, 32, 4000, 1)
        #    Conv      ->(?, 64, 4000, 1)
        #    Pool      ->(?, 64, 2000, 1)
                
        #    shape   =  (?, 64, 2000, 1)
        #    Conv      ->(?, 128, 2000, 1)
        #    Pool      ->(?, 128, 1000, 1)
               
        #    shape   =  (?, 128, 1000, 1)
        #    Conv      ->(?, 256, 1000, 1)
        #    Pool      ->(?, 256, 500, 1)
       
        
        #    shape   =  (?, 256, 500, 1)
        #    Conv      ->(?, 512, 500, 1)
        #    Pool      ->(?, 512, 250, 1)
         
        
        """"
        doe een VAE
        doe statistics op resutls allle cases channel1 van SVDD, AE, CAE
        """
 
    def forward(self, data):
        out_32_enc = self.conv1(data)
        out_64_enc = self.conv2(out_32_enc)
        out_128_enc = self.conv3(out_64_enc)
        out_256_enc = self.conv4(out_128_enc)
        out_512_enc = self.conv5(out_256_enc)
      
        out_drop = self.drop(out_512_enc)
        
        ##with skip connections implemented: add the data from the layers of the encoder
        # out_256_dec = out_256_enc+ self.deconv1(out_drop)
        # out_128_dec = out_128_enc+ self.deconv2(out_256_dec)
        # out_64_dec = out_64_enc+ self.deconv3(out_128_dec)
        # out_32_dec = out_32_enc+ self.deconv4(out_64_dec)
        # out_1_dec = data+ self.deconv5(out_32_dec)
        
        
        out_256_dec = self.deconv1(out_drop)
        out_128_dec = self.deconv2(out_256_dec)
        out_64_dec = self.deconv3(out_128_dec)
        out_32_dec = self.deconv4(out_64_dec)
        out_1_dec = self.deconv5(out_32_dec)
        
        
        return out_1_dec