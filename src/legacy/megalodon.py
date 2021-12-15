import wave
import os

infiles = os.listdir("resources/kaggle_datasets/dataset_2/nomter/")
# infiles = ["XC134724.wav", "XC134786.wav"]
outfile = "megalodon_nomter.wav"

data= []
for infile in infiles:
    w = wave.open("resources/kaggle_datasets/dataset_2/nomter/" + infile, 'rb')
    data.append( [w.getparams(), w.readframes(w.getnframes())] )
    w.close()
    
output = wave.open(outfile, 'wb')
output.setparams(data[0][0])
for i in range(len(data)):
    output.writeframes(data[i][1])
output.close()