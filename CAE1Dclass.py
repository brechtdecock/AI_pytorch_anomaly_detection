import torch
import torch.nn as nn

class Trim(nn.Module): #dont know if necessary but just in case
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :8000]

class CAE1D(nn.Module):  #0.5 second of waveform has 8 000 inputs
    def __init__(self):
        super().__init__()
        """
        use 1D components or use 2D components with [length,1] as input and [3,1] as kernel?
        
        often relu is used an only sigmoid(tanh in my case) at the end==better??? doesn't seem like it
        """
        
        #(batch size, channels, x-dim,  y-dim)
        self.encoder = torch.nn.Sequential(    
        
        nn.Conv1d(1,32, kernel_size=3, padding = "same"),
        nn.Tanh(),
        nn.MaxPool1d(kernel_size=2, stride=2),
        
        nn.Conv1d(32,64, kernel_size=3,padding = "same"),
        nn.Tanh(),
        nn.MaxPool1d(kernel_size=2, stride=2),

        nn.Conv1d(64,128, kernel_size=3,padding = "same"),
        nn.Tanh(),
        nn.MaxPool1d(kernel_size=2, stride=2),
        
        nn.Conv1d(128,256, kernel_size=3,padding = "same"),
        nn.Tanh(),
        nn.MaxPool1d(kernel_size=2, stride=2),

        nn.Conv1d(256,512, kernel_size=3,padding = "same"),
        nn.Tanh(),
        nn.MaxPool1d(kernel_size=2, stride=2),

        )  
         
        self.decoder = torch.nn.Sequential( 
        nn.ConvTranspose1d(512, 256, kernel_size = 3, stride=2),
        nn.Tanh(),
        nn.ConvTranspose1d(256, 128, kernel_size = 3, stride=2),
        nn.Tanh(),
        nn.ConvTranspose1d(128, 64, kernel_size = 3, stride=2),
        nn.Tanh(),
        nn.ConvTranspose1d(64, 32, kernel_size = 3, stride=2),
        nn.Tanh(),
        nn.ConvTranspose1d(32, 1, kernel_size = 3, stride=2),
        nn.Tanh(),
        Trim()
        )
 
    def forward(self, data):
        encoded = self.encoder(data)
        decoded = self.decoder(encoded)
        return decoded
  