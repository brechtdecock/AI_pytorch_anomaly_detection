#%%

import csv
import os
from re import sub
import pandas as pd
import pathlib
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

header = ['Full Sample Name', 'Toytype','Case', 'norm/ab', 'IND/CNT', 'Channel', 'sample_ID']

with open(r'C:\Users\brech\THESIS_local\ToyADMOS\ToycarCSV.csv', 'w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    
    
    dataset_base_path = r"C:\Users\brech\THESIS_local\ToyADMOS\Toycar all data"
    for folder in os.listdir(dataset_base_path):
        subPath = os.path.join(dataset_base_path, folder)
        for subfolder in os.listdir(subPath):
            subsubpath = os.path.join(subPath,subfolder)
            for wavefile in os.listdir(subsubpath):
               waveID, Toytype, case, norm_or_ab, IND_or_CNT, channel, sampleID = wavefile.split("_")
               if "ab" in norm_or_ab:
                   norm_or_ab = "abnormal"
               sampleID = sampleID.split(".")[0]
               data = [wavefile,Toytype, case, norm_or_ab, IND_or_CNT, channel, sampleID]          
               
               writer.writerow(data)
    
data_csv = pd.read_csv(r'C:\Users\brech\THESIS_local\ToyADMOS\ToycarCSV.csv')

#find the path of the .wav file based on the full name sample of the pandas dataset
def find_path_to_wav(full_sample_name):
    for root, dirs, files in os.walk(r'C:\Users\brech\THESIS_local\ToyADMOS'):
        for name in files:
            if name == full_sample_name:
                path_to_wavFile = os.path.abspath(os.path.join(root, name))
                return path_to_wavFile
            
            
data_case_1 = data_csv[(data_csv["Case"] == "case1") & (data_csv["norm/ab"] == "normal")]
print(data_case_1)
print(find_path_to_wav(data_case_1.iloc()[0][0]))

#%%
#average the waveform per frame, using standard frame_size = 2048 and hop_length = 512
#is the same as librosa.feature.rms
def waveform_averager(waveform, frame_size = 2048, hop_length = 512):
    averaged_waveform = []
    
    # calculate average for each frame
    for i in range(0, len(waveform), hop_length): 
        averaged_frame_waveform = np.average(np.abs(waveform[i:i+frame_size]))
        averaged_waveform.append(averaged_frame_waveform)   
    return np.array(averaged_waveform) 

waveform, sample_rate = librosa.load(find_path_to_wav(data_case_1.iloc()[0][0]))
avg_waveform = waveform_averager(waveform)
avg_frames = range(len(avg_waveform))
t_avg = librosa.frames_to_time(avg_frames)

plt.figure()
librosa.display.waveshow(waveform)
plt.plot(t_avg, avg_waveform, color="r")
plt.title("Averaged waveform")

# %%
print(librosa.get_duration(y= waveform, sr = sample_rate))
# %%
