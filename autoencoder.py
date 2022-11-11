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

#%% data preprocessing
data_case1_normal = data_csv[(data_csv["norm/ab"] == "normal") & (data_csv["Case"] == "case1")]
data_case1_abnormal = data_csv[(data_csv["norm/ab"] == "abnormal") & (data_csv["Case"] == "case1")]

train_dataset_normal, test_dataset_normal=train_test_split(data_case1_normal, test_size=0.2, shuffle=True)  #percentages, sklearn function
train_dataset_abnormal, test_dataset_abnormal=train_test_split(data_case1_abnormal, test_size=0.2, shuffle=True)

train_dataset = pd.concat([train_dataset_normal, train_dataset_abnormal])
test_dataset = pd.concat([test_dataset_normal, test_dataset_abnormal])
train_dataset, validation_dataset = train_test_split(train_dataset, test_size=0.2, shuffle=True) #further split into train and validation

X_train = train_dataset["Full Sample Name"]
X_test = test_dataset["Full Sample Name"]
X_val = validation_dataset["Full Sample Name"]

Y_train = train_dataset["norm/ab"]
Y_test = test_dataset["norm/ab"]
Y_val = validation_dataset["norm/ab"]


# %%
