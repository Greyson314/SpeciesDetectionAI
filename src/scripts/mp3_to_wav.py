"""
This is a procedure designed to convert mp3 files to WAV files. 

"""
from pydub import AudioSegment
import os

for file in os.listdir(".\\resources\\kaggle_datasets\\dataset_2\\nomter"):
    filename = file[:-4]
    sound = AudioSegment.from_mp3(".\\resources\\kaggle_datasets\\dataset_2\\nomter\\" + file)
    sound = sound.set_channels(1)
    sound.export(".\\resources\\kaggle_datasets\\dataset_2\\nomter_wav\\" + filename + ".wav", format="wav")
    print(filename)
