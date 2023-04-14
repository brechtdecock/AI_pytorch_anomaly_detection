import torch
from torch import Tensor
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import logging
import argparse

parser = argparse.ArgumentParser(prog='transformer')
parser.add_argument('--TB', help='Number of transformer blocks', required=True)
parser.add_argument('--mask', help='Type of inference mask', required=True)
parser.add_argument('--epochs', help='Number of epochs', required=True)
args = vars(parser.parse_args())

nb_transformer_blocks = int(args.get('TB'))
epochs = int(args.get('epochs'))

if args.get('mask') == 'single':
    mask = "single diag matrix"
elif args.get('mask') == 'upper':
    mask = "upper diag matrix"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


# configure the logging system
logging.basicConfig(filename='loggingFile.log', level=logging.INFO)

# create a stream handler to print log messages to the terminal
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


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
        conv_feature_embedding = torch.squeeze(out_drop, dim = -1) #so length of 512, each a vector of 250 long 
        
        
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
        
        # global mask  #to use it in the logging
        # #mask = "upper diag matrix"
        # mask = "single diag matrix"
        
        if mask =="single diag matrix":
            # convert numpy array to torch tensor    
            mat = np.eye(seq_len, k=1)
            indices = torch.from_numpy(mat).nonzero() #gets indices but in the wrong dimension
            indices = indices.transpose(0,1)
            
        elif mask == "upper diag matrix":
            indices = torch.triu_indices(seq_len, seq_len, offset=1)
        print("mask =" ,mask)
        
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
    def __init__(self, heads, nb_transformer_blocks, emb_dim, seq_length):
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
    
import pandas as pd
import numpy as np
import librosa
import os
import tqdm

import torch
from torch.utils.data import DataLoader

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


def load_dataset(dataset_path):
    """Loads a dataset from a CSV file."""
    return pd.read_csv(dataset_path)

def filter_dataset_by_cases_and_channels(dataset, cases, channels):
    """Filters a dataset to keep only the rows that correspond to the specified cases and channels."""
    selected_rows = pd.DataFrame()
    for case_number in cases:
        rows_for_case = dataset[dataset['Case'] == f'case{case_number}']
        selected_rows = pd.concat([selected_rows, rows_for_case])
    selected_rows = selected_rows[selected_rows['Channel'].isin(channels)]
    return selected_rows

def split_dataset_for_train_val_test(data_pd):
    """Splits a dataset into three parts: train, validation, and test."""
    normal_data = data_pd[data_pd['norm/ab'] == 'normal']
    abnormal_data = data_pd[data_pd['norm/ab'] == 'abnormal']
    
    # We only need normal data for training, but validation and test need both normal and abnormal data.
    train_data, intermediate_data = train_test_split(normal_data, test_size=0.2, shuffle=True)
    validation_data, test_data = train_test_split(pd.concat([abnormal_data, intermediate_data]), test_size=0.8, shuffle=True)

    return train_data, validation_data, test_data

def train_val_test(dataset_path=None, cases=[], channels=[]):
    """Loads a dataset, filters it, and splits it for training, validation, and test."""
    dataset = load_dataset(dataset_path)
    filtered_dataset = filter_dataset_by_cases_and_channels(dataset, cases, channels)
    train_data, validation_data, test_data = split_dataset_for_train_val_test(filtered_dataset)
    return train_data, validation_data, test_data


datapath = r'/data/leuven/350/vsc35045/ToyADMOS/ToycarCSV.csv'
cases = [1]
channels = ['ch1']
train_data, validation_data, test_data = train_val_test(dataset_path=datapath, cases=cases, channels=channels)

import librosa
import ffmpeg
#helper functions 
def find_path_to_wav(full_sample_name):
    for root, dirs, files in os.walk(os.path.dirname(datapath)):
        for name in files:
            if name == full_sample_name:
                path_to_wavFile = os.path.abspath(os.path.join(root, name))
                return path_to_wavFile

def get_sample_waveform_normalised(full_sample_name, start = 0, stop = 11):
    #returns waveform values, cut to seconds going from start to stop
    sample_path = find_path_to_wav(full_sample_name)
    waveform, sample_rate = librosa.load(sample_path, sr= None)
    waveform = waveform[int(start*sample_rate): int(stop*sample_rate)]
    return librosa.util.normalize(waveform)


X_train_wav = train_data["Full Sample Name"].values
X_test_wav = test_data["Full Sample Name"].values
X_valid_wav = validation_data["Full Sample Name"].values

batch_train = np.array([get_sample_waveform_normalised(elem,4,4.5) for elem in X_train_wav]) 
batch_test = np.array([get_sample_waveform_normalised(elem,4,4.5) for elem in X_test_wav])
batch_val = np.array([get_sample_waveform_normalised(elem,4,4.5) for elem in X_valid_wav])

batch_train_reshaped =  np.reshape(batch_train,(len(batch_train),1,8000,1))
batch_test_reshaped =  np.reshape(batch_test,(len(batch_test),1,8000,1))
batch_val_reshaped =  np.reshape(batch_val,(len(batch_val),1,8000,1))

X_train = DataLoader(batch_train_reshaped, batch_size=64, shuffle=False)  # comes from 64
X_test = DataLoader(batch_test_reshaped, batch_size=64, shuffle=False)
X_val = DataLoader(batch_val_reshaped, batch_size=64, shuffle=False)

Y_train = train_data["norm/ab"]
Y_train = np.array([1 if i == "normal" else -1 for i in Y_train]).reshape(-1, 1)

Y_val = validation_data["norm/ab"]
Y_val = np.array([1 if i == "normal" else -1 for i in Y_val]).reshape(-1, 1)

Y_test = test_data["norm/ab"]
Y_test = np.array([1 if i == "normal" else -1 for i in Y_test]).reshape(-1, 1)


heads = 5
#nb_transformer_blocks = 6
model = Transformer(heads, nb_transformer_blocks, emb_dim=250,seq_length=512).to(device=device)


model_loss = nn.MSELoss()    #?nn.L1Loss() best type of loss for sound?, MSE loss seems to result in lower loss
learning_rate = 0.0001  #0.0001 seems best so far
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

#epochs = 50
losses = []
avg_val_losses = []

def train(epochs, model, model_loss):
    for epoch in tqdm.tqdm(range(epochs)):
        
        for batch_idx, data in enumerate(X_train):
            model.train(True)
            # Zero your gradients for every batch!
            model.zero_grad()
            
            #for param in model.parameters(): #instead of model.zero_grad: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#:~:text=implement%20this%20optimization.-,Use%20parameter.grad,-%3D%20None%20instead%20of
            #    param.grad = None
            
            # Make predictions for this batch
            data_gpu = data.to(device= device)
            outputs = model(data_gpu)
    
            # Compute the loss and its gradients
            loss = model_loss(outputs, data_gpu)
            
            loss.backward()
            optimizer.step()
           
            losses.append(loss.item())
            #del loss 
            #del data #free memory
            #del outputs
            
            model.train(False)

        
        #hier validation data gebruiken, gets run once per epoch
        #get average loss value of the validation data
        running_val_loss = []
        for val_data in X_val:
            val_data_gpu = val_data.to(device=device)
            val_outputs = model(val_data_gpu)
            val_loss = model_loss(val_outputs, val_data_gpu)
            
            running_val_loss.append(val_loss.item())

        avg_val_losses.append(np.average(running_val_loss))

train(model=model, epochs=epochs, model_loss=model_loss)

plt.figure(figsize = (15,10))
plt.subplot(2,1,1)
plt.xlabel("epochs*nb_of_batches")
plt.ylabel("Training losses")
plt.plot(losses)

plt.subplot(2,1,2)
plt.plot(avg_val_losses)
plt.xlabel("epochs")
plt.ylabel("Validation losses")
plt.show()
plt.savefig('TrainLossValLoss'+ str(epochs) +'_'+ str(nb_transformer_blocks)+'_'+str(mask) +'.png', dpi=400)


def score(dataset, scoring_function): 
    scores_normal = [] #scores of each waveform in the test datase
    scores_abnormal = []
    
    for line_of_data in dataset.iloc():
        waveform = np.array(get_sample_waveform_normalised(line_of_data["Full Sample Name"], 4, 4.5))
        waveform = np.reshape(waveform,(-1,1,8000,1))
        waveform_gpu = torch.FloatTensor(waveform).to(device=device)

        predicted_waveform = model(waveform_gpu)
        error = scoring_function(predicted_waveform,waveform_gpu) 
        
        if line_of_data["norm/ab"] == "normal":
            scores_normal.append(error.detach().cpu().numpy().item()) 
        
        if line_of_data["norm/ab"] == "abnormal":
            scores_abnormal.append(error.detach().cpu().numpy().item()) 
   
    return scores_normal, scores_abnormal

MSE_scores_normal, MSE_scores_abnormal = score(test_data, scoring_function = nn.MSELoss())
L1_scores_normal, L1_scores_abnormal = score(test_data, scoring_function = nn.L1Loss())
CEL_scores_normal, CEL_scores_abnormal =score(test_data, scoring_function =nn.CrossEntropyLoss()) 

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


#RESCALE DATA! met standardscaler
L1_scores_normal = np.array(L1_scores_normal).reshape(-1, 1)
L1_scores_abnormal = np.array(L1_scores_abnormal).reshape(-1, 1)

scaler_normal = StandardScaler() #necessary?
scaler_normal.fit_transform(L1_scores_normal)

scaler_abnormal = StandardScaler()
scaler_abnormal.fit_transform(L1_scores_abnormal)

L1_all_scores = np.append(L1_scores_abnormal, L1_scores_normal).reshape(-1, 1) # first abnormal(-1), then normal(1) # test scores
L1_all_results = np.ravel(np.concatenate((np.ones_like(L1_scores_abnormal)*(-1), np.ones_like(L1_scores_normal)), axis=0)) #true result

# confusion matrix and ROC curve
fpr, tpr, _ = roc_curve(L1_all_results,L1_all_scores )  #y_true, y_score
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=3, label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", lw=3, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.grid()
plt.show()
plt.savefig('AUCscore'+ str(epochs) +'_'+ str(nb_transformer_blocks)+'_'+str(mask) +'.png', dpi=400)


logging.info(f"model = {model} \n case = {cases} \n heads = {heads} \n transformer_blocks = {nb_transformer_blocks} \n mask = {mask} \n learning rate = {learning_rate} \n losses = {losses} \n avg_val_losses = {avg_val_losses} \n AUC score = {roc_auc}")
