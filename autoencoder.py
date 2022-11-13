#%%imports
import os
from re import sub
from unittest import case
import pandas as pd
from tqdm import tqdm
import librosa
import librosa.display
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch import *
from torch.utils.data import DataLoader
from AutoEncoderClass import *
import wavio

def check_cuda():  
    if torch.cuda.is_available(): 
        print("cuda is available")
        dev = "cuda" 
    else: 
        dev = "cpu" 
    return dev
device = torch.device(check_cuda())
print("current device:", device)


#header = ['Full Sample Name', 'Toytype','Case', 'norm/ab', 'IND/CNT', 'Channel', 'sample_ID']
data_csv = pd.read_csv(r'C:\Users\brech\THESIS_local\ToyADMOS\ToycarCSV.csv')
def find_path_to_wav(full_sample_name):
    for root, dirs, files in os.walk(r'C:\Users\brech\THESIS_local\ToyADMOS'):
        for name in files:
            if name == full_sample_name:
                path_to_wavFile = os.path.abspath(os.path.join(root, name))
                return path_to_wavFile
     

def get_waveform_seconds(full_sample_name, start = 0, stop = 11):
    #returns waveform values, cut to seconds going from start to stop
    
    sample_path = find_path_to_wav(full_sample_name)
    waveform, sample_rate = librosa.load(sample_path, sr= 16000)
    waveform = waveform[start*sample_rate : stop*sample_rate]
    return waveform

#%% 
#data preprocessing
data_case1_normal = data_csv[(data_csv["norm/ab"] == "normal") & (data_csv["Case"] == "case1")]
data_case1_abnormal = data_csv[(data_csv["norm/ab"] == "abnormal") & (data_csv["Case"] == "case1")]

train_dataset_normal, test_dataset_normal=train_test_split(data_case1_normal, test_size=0.2, shuffle=True)  #percentages, sklearn function
train_dataset_abnormal, test_dataset_abnormal=train_test_split(data_case1_abnormal, test_size=0.2, shuffle=True)

train_dataset = pd.concat([train_dataset_normal, train_dataset_abnormal])
test_dataset = pd.concat([test_dataset_normal, test_dataset_abnormal])
train_dataset, validation_dataset = train_test_split(train_dataset, test_size=0.2, shuffle=True) #further split into train and validation

X_train = DataLoader(train_dataset["Full Sample Name"].values, batch_size = 64, shuffle = True)
X_test = DataLoader(test_dataset["Full Sample Name"].values, batch_size = 64, shuffle = True)
X_train = DataLoader(validation_dataset["Full Sample Name"].values, batch_size = 64, shuffle = True)

Y_train = train_dataset["norm/ab"]
Y_test = test_dataset["norm/ab"]
Y_val = validation_dataset["norm/ab"]
#%%
model = AutoEncoder(16000).to(device=device)#assume 1 sec of waveform
loss = nn.L1Loss()
#nn.MSELoss()?? best type of loss for sound?
learning_rate = 0.01
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
print(model)
#%%
def score(x): # not yet in use
    y_pred = model(V(x))
    x1 = V(x)
    return loss(y_pred,x1).item()

epochs = 10
outputs = []
losses = []


#%%
def train(epochs, model, model_loss):
    
    for epoch in tqdm(arange(0, epochs)):
        for waveform_Sample_name in X_train: #X_train is dataloader of the "names" of the wavefiles
            waveform = get_waveform_seconds(waveform_Sample_name[0], 4, 5)
            waveform = torch.FloatTensor(waveform).to(device=device)
            
            reconstructed = model(waveform) 
            loss = model_loss(reconstructed, waveform)
                
            # Storing the losses in a list for plotting
            losses.append(loss)
            outputs.append((epoch, waveform, reconstructed))
            # The gradients are set to zero,
            # the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            


train(model=model, epochs=epochs, model_loss=loss)

print(outputs)
last_waveform = outputs[-1][1].detach().cpu().numpy()
last_reconstructed = outputs[-1][2].detach().cpu().numpy()
print(last_waveform, last_reconstructed) #reconstruted gives all zeros!!!!

wavio.write("last_waveform_signal.wav", last_waveform, 16000, sampwidth=4)
wavio.write("last_reconstructed_signal.wav", last_reconstructed, 16000, sampwidth=4)


print([X.detach().cpu().numpy().item() for X in losses])
#plt.plot(x, [X.detach().cpu().numpy().item() for X in losses], label="loss")
#plt.legend()
plt.plot([X.detach().cpu().numpy().item() for X in losses])
plt.show()

