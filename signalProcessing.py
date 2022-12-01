#%%
import csv
import os
from re import sub
from unittest import case
import wave
import pandas as pd
import pathlib
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


#header = ['Full Sample Name', 'Toytype','Case', 'norm/ab', 'IND/CNT', 'Channel', 'sample_ID']
data_csv = pd.read_csv(r'C:\Users\brech\THESIS_local\ToyADMOS\ToycarCSV.csv')
def find_path_to_wav(full_sample_name):
    for root, dirs, files in os.walk(r'C:\Users\brech\THESIS_local\ToyADMOS'):
        for name in files:
            if name == full_sample_name:
                path_to_wavFile = os.path.abspath(os.path.join(root, name))
                return path_to_wavFile
            
            
data_case_normal = data_csv[(data_csv["norm/ab"] == "normal") & (data_csv["Case"] == "case1") & (data_csv["Channel"] == "ch2")]
data_case_abnormal = data_csv[(data_csv["norm/ab"] == "abnormal") & (data_csv["Case"] == "case1") & (data_csv["Channel"] == "ch2")]
#%%retain only 1 second of waveform
sample_path = find_path_to_wav(data_case_normal.iloc()[1][0])
waveform, sample_rate = librosa.load(sample_path, sr= 16000)
waveform = waveform[0+4*sample_rate:5*sample_rate]
plt.figure()
librosa.display.waveshow(waveform, sr = sample_rate)
plt.title("Current sample of only 1 second " )
plt.show()


#%% rms for normal data
rms_normal = []
for i in range(len(data_case_normal)):     #len(data_case_normal)):
    sample_path = find_path_to_wav(data_case_normal.iloc()[i][0])
    waveform, sample_rate = librosa.load(sample_path, sr= 16000)
    S, phase = librosa.magphase(librosa.stft(waveform,window=np.ones,center=False, n_fft=len(waveform)))
    rms = librosa.feature.rms(S=S, frame_length=len(waveform))
    rms_normal.append(rms.item())

#%% rms for abnormal data
rms_abnormal = []
for i in range(len(data_case_abnormal)):     #len(data_case_normal)):
    sample_path = find_path_to_wav(data_case_abnormal.iloc()[i][0])
    waveform, sample_rate = librosa.load(sample_path, sr= 16000)
    S, phase = librosa.magphase(librosa.stft(waveform,window=np.ones,center=False, n_fft=len(waveform)))
    rms = librosa.feature.rms(S=S, frame_length=len(waveform))
    rms_abnormal.append(rms.item())
#%% plot rms values'
plt.figure()
plt.plot(rms_normal[:100], color="r")
plt.plot(range(len(rms_normal), len(rms_normal)+len(rms_abnormal)), rms_abnormal, color="b")
plt.title(f"RMS of healthy and unhealthy samples from case: 1 and channel: 2")
plt.show()

# %%
