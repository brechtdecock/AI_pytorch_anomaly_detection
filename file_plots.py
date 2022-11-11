#%%
from cgi import print_arguments
from sqlite3 import Row
import torch 
import torchaudio
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

print("Is cuda available:", torch.cuda.is_available())

data_case = 1
data_case_channel = 1

dataset_base_path = r"C:\Users\brech\THESIS_local\ToyADMOS\Toycar all data"
dataset_total_path = dataset_base_path + f"\expCase{data_case}_ch{data_case_channel}_dataset_ToyCar"

dataset_total_path_test_anomaly = dataset_total_path +r"\test_anomaly"
dataset_total_path_train_normal = dataset_total_path +r"\train_normal"
dataset_total_path_test_normal = dataset_total_path +r"\test_normal"


sample_waves = os.listdir(dataset_total_path_test_normal)
sample_path = dataset_total_path_test_normal+"\\" + sample_waves[0]
print("the current .wav signal is:"+ sample_path)

print(torchaudio.info(sample_path))#different value than librosa?

"""
librosa resamples to 22050 instead of native sampling rate: what to do?
"""
waveform, sample_rate = librosa.load(sample_path)
print("sample rate is "+ str(sample_rate))

#%%%%%%%%%%%%%
#plot waveform
plt.figure()
librosa.display.waveshow(waveform, sr = sample_rate)
plt.title("Current sample: " + sample_waves[0])
plt.show()

#plot spectogram
plt.figure()
D = librosa.amplitude_to_db(np.abs(librosa.stft(waveform)))
print("size of D" + str(D.shape))
librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sample_rate)
plt.title("Current sample: " + sample_waves[0])
plt.show()

#plot magnitude spectrum
def plot_magnitude_spectrum(signal, sr):
    Y = np.fft.fft(signal)
    Y_mag = np.absolute(Y)
    print(Y_mag,len(Y_mag), sr)
    plt.figure()
    
    freq = np.linspace(0, sr, len(Y_mag))
    
    plt.plot(freq,Y_mag)
    plt.xlabel('Frequency (Hz)')
    plt.title("Current sample: " + sample_waves[0])
    plt.show()
plot_magnitude_spectrum(waveform, sample_rate)

#%% 
#librosa default frame and hop lengths are set to 2048 and 512 
#root mean square of sample
rms = librosa.feature.rms(waveform)[0]


frames = range(len(rms))
t = librosa.frames_to_time(frames)

plt.figure()
plt.plot(t, rms, color="r")
librosa.display.waveshow(waveform, sr = sample_rate)
plt.title("RMS added on waveform")
plt.show()



# %%
#zero crossing rate, often used to classify percussive sounds.
zcr = librosa.feature.zero_crossing_rate(waveform)[0]

plt.figure()
plt.plot(t, zcr, color="g")
librosa.display.waveshow(waveform, sr = sample_rate)
plt.title("ZCR of waveform")
plt.show()

# %%
spectral_centroid =  librosa.feature.spectral_centroid(y=waveform)[0]
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=waveform)[0]
plt.figure()
plt.plot(t, spectral_centroid, color="g")
plt.plot(t, spectral_bandwidth, color="y")
plt.title("SC of sample")
plt.legend(["spectral centroid", "spectral bandwidth"])
plt.show()

# %%
