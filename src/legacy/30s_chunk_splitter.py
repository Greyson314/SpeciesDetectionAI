from pydub import AudioSegment
import os
import math

working_in = 'comter'
os.chdir(f"resources/kaggle_datasets/dataset_2/")

num_seconds = 30
chunk = num_seconds*1000 #30 seconds

oldAudio = AudioSegment.from_wav(f"megalodon_{working_in}.wav")
num_divisions = math.floor(len(oldAudio)/chunk)
print(num_divisions)
t1 = 0
t2 = chunk
for x in range(num_divisions):
    newAudio = oldAudio[t1:t2]
    newAudio.export(f'{working_in}_chunk/{x}.wav', format="wav") #Exports to a wav file in the current path.
    t1 += chunk
    t2 += chunk
