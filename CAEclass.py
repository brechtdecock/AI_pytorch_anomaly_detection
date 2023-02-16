import torch
import torch.nn as nn


class CAE(nn.Module):  #0.5 second of waveform has 8 000 inputs
    def __init__(self):
        super().__init__()
        """
        use 1D components or use 2D components with [length,1] as input and [3,1] as kernel?
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
        
        
        
        )  
          
    # input_shape=(28,28,1)
    # n_channels = input_shape[-1]
    # model = Sequential()
    # model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    # model.add(MaxPool2D(padding='same'))
    # model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
    # model.add(MaxPool2D(padding='same'))
    # model.add(Conv2D(8, (3,3), activation='relu', padding='same'))
    # model.add(UpSampling2D())
    # model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
    # model.add(UpSampling2D())
    # model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    # model.add(Conv2D(n_channels, (3,3), activation='sigmoid', padding='same'))
       
  
        
         
        self.decoder = torch.nn.Sequential( 
        nn.ConvTranspose2d(128, 64, kernel_size = (3,1), stride=2, padding = 0),
        nn.Tanh(),
        nn.ConvTranspose2d(64, 32, kernel_size = (3,1), stride=2, padding = 0),
        nn.Tanh(),
        nn.ConvTranspose2d(32, 1, kernel_size = (3,1), stride=2, padding = 0),
        nn.Tanh(),
        nn.Flatten() #??????
        )
 
    def forward(self, data):
        encoded = self.encoder(data)
        decoded = self.decoder(encoded)
        return decoded
  