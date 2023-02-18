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
        """
        
        #(batch size, channels, x-dim,  y-dim)
        self.encoder = torch.nn.Sequential(    
        # vector in shape=(?, 1, 8000, 1)
        #    Conv     -> (?, 32, 8000, 1)
        #    Pool     -> (?, 32, 4000, 1)
        nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size=(3,1), stride=1, padding="same"),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=(2,1), stride=2),
        
        #    shape =     (?, 32, 4000, 1)
        #    Conv      ->(?, 64, 4000, 1)
        #    Pool      ->(?, 64, 2000, 1)
        nn.Conv2d(in_channels = 32,out_channels = 64, kernel_size=(3,1), stride=1, padding="same"),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=(2,1), stride=2),
        #torch.nn.Dropout(p=1 - keep_prob))
        
        #    shape   =  (?, 64, 2000, 1)
        #    Conv      ->(?, 128, 2000, 1)
        #    Pool      ->(?, 128, 1000, 1)
        nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size=(3,1), stride=1, padding="same"),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=(2,1), stride=2),
        #nn.Dropout(p=1 - keep_prob))
        
        #    shape   =  (?, 128, 1000, 1)
        #    Conv      ->(?, 256, 1000, 1)
        #    Pool      ->(?, 256, 500, 1)
        nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size=(3,1), stride=1, padding="same"),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=(2,1), stride=2),
        
        
        #    shape   =  (?, 256, 500, 1)
        #    Conv      ->(?, 512, 500, 1)
        #    Pool      ->(?, 512, 250, 1)
        nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size=(3,1), stride=1, padding="same"),
        nn.Tanh(),
        nn.MaxPool2d(kernel_size=(2,1), stride=2),
        )  
         
        self.decoder = torch.nn.Sequential( 
        nn.ConvTranspose2d(512, 256, kernel_size = (3,1), stride=2),
        nn.Tanh(),
        nn.ConvTranspose2d(256, 128, kernel_size = (3,1), stride=2),
        nn.Tanh(),
        nn.ConvTranspose2d(128, 64, kernel_size = (3,1), stride=2),
        nn.Tanh(),
        nn.ConvTranspose2d(64, 32, kernel_size = (3,1), stride=2),
        nn.Tanh(),
        nn.ConvTranspose2d(32, 1, kernel_size = (3,1), stride=2),
        nn.Tanh(),
        Trim(),
        #nn.Flatten() #??????
        )
 
    def forward(self, data):
        encoded = self.encoder(data)
        decoded = self.decoder(encoded)
        return decoded
  