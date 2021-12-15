"""
Author: Grey Hutchinson
Date: 12/14/2021

This script allows the user to load a wav file and display the mfccs.
"""
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt

FIG_SIZE = (15,10)

file_number = "XC25566"
file_name = 'trues/' + str(file_number) + '.wav'

# load audio file with Librosa
signal, sample_rate = librosa.load(file_name, sr=22050)
noise=np.random.normal(0, 0.001, signal.shape[0])
# FFT -> power spectrum
# perform Fourier transform
# STFT -> spectrogram
hop_length = 256 # in num. of samples
n_fft = 2048 # window in num. of samples

# WAVEFORM
# display waveform
# plt.figure(figsize=FIG_SIZE)
# librosa.display.waveplot(signal, sample_rate, alpha=0.4)
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.title("Waveform")
MFCCs = librosa.feature.mfcc(signal, sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

# display MFCCs
plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(MFCCs, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC coefficients")
plt.colorbar()
plt.title("MFCCs")


signal=signal+noise

# MFCCs
# extract 13 MFCCs
MFCCs = librosa.feature.mfcc(signal, sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

# display MFCCs
plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(MFCCs, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC coefficients")
plt.colorbar()
plt.title("MFCCs")

# show plots
plt.show()