#%%imports
import os
from re import sub
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
#todo normalise to make it possible to train over different cases(that are of different loudness)
data_case1_normal = data_csv[(data_csv["norm/ab"] == "normal") & (data_csv["Case"] == "case1")]
data_case1_abnormal = data_csv[(data_csv["norm/ab"] == "abnormal") & (data_csv["Case"] == "case1")]

train_dataset_normal, test_dataset_normal=train_test_split(data_case1_normal, test_size=0.2, shuffle=True)  #percentages, sklearn function
test_dataset_abnormal = data_case1_abnormal

train_dataset = train_dataset_normal  
test_dataset = pd.concat([test_dataset_normal, test_dataset_abnormal])
test_dataset, validation_dataset = train_test_split(test_dataset, test_size=0.2, shuffle=True) #further split into test and validation

#shuffle all datasets for good measure
train_dataset= shuffle(train_dataset)
validation_dataset= shuffle(validation_dataset)
test_dataset= shuffle(test_dataset)

X_train = DataLoader(train_dataset["Full Sample Name"].values, batch_size = 64, shuffle = False)
X_test = DataLoader(test_dataset["Full Sample Name"].values, batch_size = 64, shuffle = False)
X_train = DataLoader(validation_dataset["Full Sample Name"].values, batch_size = 64, shuffle = False)

Y_train = train_dataset["norm/ab"]
Y_test = test_dataset["norm/ab"]
Y_val = validation_dataset["norm/ab"]
#%%
model = AutoEncoder(16000).to(device=device)#assume 1 sec of waveform
loss = nn.MSELoss()    #nn.L1Loss()? best type of loss for sound?
learning_rate = 0.01
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)
print(model)

#%%
epochs = 10
original_wav = []
reconstructed_wav = []

losses = []

def train(epochs, model, model_loss):
    
    for epoch in tqdm(arange(0, epochs)):
        for waveform_sample_name in X_train: #X_train is dataloader of the "names" of the wavefiles
            waveform = get_waveform_seconds(waveform_sample_name[0], 4, 5)
            waveform = np.interp(waveform, (waveform.min(), waveform.max()), (-1, +1)) #scale the date between -1 and +1
            waveform = torch.FloatTensor(waveform).to(device=device)
            
            print(waveform)
            reconstructed = model(waveform)
             
            loss = model_loss(reconstructed, waveform)
                
            # Storing the losses in a list for plotting
            losses.append(loss)
            original_wav.append(waveform)
            reconstructed_wav.append(reconstructed)
            # The gradients are set to zero,
            # the gradient is computed and stored.
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        #per epoch de loss checken via de validation set
        #code hier
        

train(model=model, epochs=epochs, model_loss=loss)

#%%
last_waveform = original_wav[-1].detach().cpu().numpy()
last_reconstructed = reconstructed_wav[-1].detach().cpu().numpy()
print(last_waveform, last_reconstructed)

wavio.write("last_waveform_signal.wav", last_waveform, 16000, sampwidth=4)
wavio.write("last_reconstructed_signal.wav", last_reconstructed, 16000, sampwidth=4)

print([X.detach().cpu().numpy().item() for X in losses])
#plt.plot(x, [X.detach().cpu().numpy().item() for X in losses], label="loss")
#plt.legend()
plt.plot([X.detach().cpu().numpy().item() for X in losses])
plt.show()
  
#%% 
def score(dataset):
    scores_normal = [] #scores of each waveform in the test dataset
    scores_abnormal = []
    
    for line_of_data in dataset.iloc():
        waveform = get_waveform_seconds(line_of_data["Full Sample Name"], 4, 5)
        waveform = np.interp(waveform, (waveform.min(), waveform.max()), (-1, +1)) #scale the date between -1 and +1
        waveform = torch.FloatTensor(waveform).to(device=device)
        
        predicted_waveform = model(waveform)
        errorfunc= nn.L1Loss()
        error = errorfunc(predicted_waveform,waveform)
        
        if line_of_data["norm/ab"] == "normal":
            scores_normal.append(error.detach().cpu().numpy().item()) 
        
        if line_of_data["norm/ab"] == "abnormal":
            scores_abnormal.append(error.detach().cpu().numpy().item()) 
   
    return scores_normal, scores_abnormal

scores_normal, scores_abnormal = score(validation_dataset)
plt.plot(scores_normal, scores_normal, 'bo') 
plt.plot(scores_abnormal, scores_abnormal, 'rx') 
plt.show
# %%
#next up try to draw a boundary around the 'normal'scores, go to higher dimension, use svm with a kernel

from sklearn import svm
from sklearn.svm import OneClassSVM
from sklearn.inspection import DecisionBoundaryDisplay

#clf = svm.SVC(kernel='poly', degree=4)
#clf = OneClassSVM(gamma = 'auto', kernel = 'rbf')
clf = svm.SVC(kernel= 'rbf') # Kernel


correct_classification = np.concatenate((np.ones_like(scores_abnormal)*(-1), np.ones_like(scores_normal)), axis=0) 

scores_normal = np.r_['1,2,0', scores_normal, scores_normal]#np.zeros_like(scores_normal)]
scores_abnormal = np.r_['1,2,0', scores_abnormal,scores_abnormal]#np.zeros_like(scores_abnormal)]
all_scores = np.concatenate((scores_abnormal,scores_normal), axis= 0)
#abnormal = -1, normal = 1

clf.fit(all_scores, correct_classification)
result = clf.predict(all_scores)

disp = DecisionBoundaryDisplay.from_estimator(clf, all_scores, response_method = "predict",alpha=0.5,)
disp.ax_.scatter(all_scores[:,0],all_scores[:,1,] , c = correct_classification ,edgecolor="black")

plt.show()
# %%
