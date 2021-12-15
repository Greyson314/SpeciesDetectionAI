"""
Convert mp3 files to WAV files. 

"""
from pydub import AudioSegment
import os

for file in os.listdir("falses"):
    filename = file[:-4]
    sound = AudioSegment.from_mp3("falses\\" + file)
    sound = sound.set_channels(1)
    sound.export(".falses_wav\\" + filename + ".wav", format="wav")
    print(filename)
