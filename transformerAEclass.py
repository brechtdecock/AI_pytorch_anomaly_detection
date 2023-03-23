import torch
from torch import Tensor
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

class AudioEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
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
        #    shape   =  (?, 256, 500, 1)
        #    Conv      ->(?, 512, 500, 1)
        #    Pool      ->(?, 512, 250, 1)-->squeeze en dan transpose for 250x512
        
        self.drop  = nn.Dropout(0.1)
          
    def getPositionEncoding(self,rows, cols,n=10000):
        P = torch.zeros((rows, cols))
        for k in range(rows):
            for i in torch.arange(int(cols/2)):
                denominator = torch.pow(n, 2*i/cols)
                P[k, 2*i] = torch.sin(k/denominator)
                P[k, 2*i+1] = torch.cos(k/denominator)
        return P
    
    
        
    def forward(self, data):
       
        out_32_enc = self.conv1(data)
        out_64_enc = self.conv2(out_32_enc)
        out_128_enc = self.conv3(out_64_enc)
        out_256_enc = self.conv4(out_128_enc)
        out_512_enc = self.conv5(out_256_enc)
      
        out_drop = self.drop(out_512_enc)
        
        #typically the vector is stored horizontally, while the sequence is stored vertically
        conv_feature_embedding = torch.squeeze(out_drop) #so length of 512, each a vector of 250 long 
        
        
        position_embedding = self.getPositionEncoding(512,250).to(device=conv_feature_embedding.device)
        return torch.add(conv_feature_embedding,position_embedding) 
    
    
class deconvEmbedding(nn.Module):
        def __init__(self):
                super().__init__()
        #(?, 512, 250, 1) back all the way upto 
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
                
        def forward(self, x):
                out_256_dec = self.deconv1(x)
                out_128_dec = self.deconv2(out_256_dec)
                out_64_dec = self.deconv3(out_128_dec)
                out_32_dec = self.deconv4(out_64_dec)
                out_1_dec = self.deconv5(out_32_dec)
                return out_1_dec
            
class SelfAttention(nn.Module):
    def __init__(self, emb_dim, heads=5, mask=False): 
        super().__init__()
        
        assert emb_dim % heads == 0
        
        self.emb_dim = emb_dim
        self.heads = heads
        
        # These compute the queries, keys and values for ALL heads
        self.to_keys    = nn.Linear(emb_dim, emb_dim, bias=False)
        self.to_queries = nn.Linear(emb_dim, emb_dim, bias=False)
        self.to_values  = nn.Linear(emb_dim, emb_dim, bias=False)

        # This will be applied after the multi-head self-attention operation.
        self.unifyheads = nn.Linear(emb_dim, emb_dim)
        
    def forward(self, x):

        batch, seq_len, emb_dim = x.size()
        heads = self.heads #double code?

        queries = self.to_queries(x)
        keys    = self.to_keys(x)   
        values  = self.to_values(x)
        
            
        s = emb_dim // heads  #to divide the query(and key and value) matrix that contains ALL head info to the query of a single head

        """"torch.view: returns a new tensor with the same data as the self tensor but of a different shape.
        This simply reshapes the tensors to add a dimension that iterations over the heads. 
        For a single vector in our sequence you can think of it as reshaping a vector of 
        dimension emb_dim into a matrix of (head x emb_dim//head)
        
        before: all HEADS info: b x seq_len x emb_dim
        after: all HEADS info but iterable per head: b x seq_len x heads x (emb_dim//heads)
        """
        keys    = keys.view(batch, seq_len, heads, s)
        queries = queries.view(batch, seq_len, heads, s)
        values  = values.view(batch, seq_len, heads, s)
        
        keys = keys.transpose(1, 2).contiguous().view(batch * heads, seq_len, s)
        queries = queries.transpose(1, 2).contiguous().view(batch * heads, seq_len, s)
        values = values.transpose(1, 2).contiguous().view(batch * heads, seq_len, s)
        
            
        """
        Dit volgt gewoon de attention formula: https://storrs.io/attention/#:~:text=the%20attention%20operation%3F-,Attention,-Equation
        """
        # Get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2)) #not (0,1) since the zero dimesion are the batches
        # -- dot has size (batch*heads, seq_len, seq_len) containing raw weights

        # scale the dot product
        dot = dot / (emb_dim ** (1/2))
        
        
        """
        if mask is used: create upper triangluar matrix of -inf(this will set softmax to zero at those positions)
        """
        indices = torch.triu_indices(seq_len, seq_len, offset=1)
        dot[:, indices[0], indices[1]] = float('-inf')

        # normalize 
        dot = F.softmax(dot, dim=2)
        # - dot now contains row-wise normalized weights
         
        # apply the self attention to the values
        out = torch.bmm(dot, values).view(batch, heads, seq_len, s)
        
        # swap heads, seq_len back, unify heads
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, s * heads)  #batch x seq_len x emb_dim
    
        return self.unifyheads(out)
    
    
class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, heads):
        super().__init__()

        self.attention = SelfAttention(emb_dim, heads=heads)

        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)


        """
        We’ve made the relatively arbitrary choice of making the hidden layer of the feedforward 4 times as big as the input and output. 
        Smaller values may work as well, and save memory, but it should be bigger than the input/output layers.
        """
        self.ff = nn.Sequential(
        nn.Linear(emb_dim, 4 * emb_dim),
        nn.ReLU(),
        nn.Linear(4 * emb_dim, emb_dim))

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)

        fedforward = self.ff(x)
        return self.norm2(fedforward + x)
    
    
class Transformer(nn.Module):
    def __init__(self, emb_dim, heads, nb_transformer_blocks, seq_length):
        super().__init__()

        self.embedding = AudioEmbedding()

		# The sequence of transformer blocks that does all the
		# heavy lifting
        tblocks = []
        for i in range(nb_transformer_blocks):
            tblocks.append(TransformerBlock(emb_dim=emb_dim, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)

        self.toAudio = deconvEmbedding()

    def forward(self, x):
        token = self.embedding(x) #token == embedded data, includes vectorization and position data, naming might be off here

        x = self.tblocks(token)
        
        x = self.toAudio(x.unsqueeze(-1)) ## add a dimension so [batch, 512, 250] is of the expected size [batch, 512, 250,1] 
        return x