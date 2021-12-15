# import the pyplot and wavfile modules

import matplotlib.pyplot as plot
from scipy.io import wavfile
import os
dir = ".\\resources\\kaggle_datasets\\dataset_2\\"

def make_spectro(signalData, samplingFrequency, path):
    plot.specgram(signalData, Fs=samplingFrequency)
    plot.savefig(path, dpi = 300)


# Read the wav file (mono)
for file in os.listdir(dir + "comter_wav"):
    print("HERE:    " + dir + "comter_wav\\" + file)
    filename = file[:-4]
    samplingFrequency, signalData = wavfile.read(dir + "comter_wav\\" + file)
    print(signalData.shape)

    make_spectro(signalData, samplingFrequency, dir + "comter_spectro\\" + filename + ".png")
    # print(filename)
