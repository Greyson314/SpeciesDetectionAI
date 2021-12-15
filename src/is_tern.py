#should load a wav file or a set of wav files
#for every one of those files, identify whether or not it thinks it is a tern, based on the model
import keras
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt


model = keras.models.load_model('resources/kaggle_datasets/dataset_2/saved_model')
file_number = 76
file_name = 'resources/kaggle_datasets/dataset_2/comter_chunk_full/' + str(file_number) + '.wav'

signal, sample_rate = librosa.load(file_name, sr=28000)

hop_length = 512 #the default spacing between frames
n_fft = 2048 #number of samples #TODO look into this
# n_mfcc = 128

fft = np.fft.fft(signal)

# def padding(array, xx, yy):
#     h = array.shape[0]
#     w = array.shape[1]
#     a = (xx - h) // 2
#     aa = xx - a - h
#     b = (yy - w) // 2
#     bb = yy - b - w
#     return np.pad(array, pad_width=((a, aa), (b, bb)), mode="constant")


# def get_features(filename, is_tern=True):
#     features=[] #list to save features
#     labels=[] #list to save labels
#       #get the filename        
#     #load the file
#     _labels, sr = librosa.load(filename,sr=28000)
#     #cut the file from tstart to tend 
#     data = np.array([padding(librosa.feature.mfcc(_labels, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc), n_mfcc, 1641)])
#         #takes about 12 minutes as of 10/4

#     print("Data Shape: ", data.shape) 
#     features.append(data)
#     labels.append(is_tern)
#     output=np.concatenate(features,axis=0)
#     return(np.array(output), labels)

# features, labels = get_features(file_name, is_tern=True)

# result = model.predict(features)
# print(result)
# classes = np.argmax(result)
# print(classes)

FIG_SIZE = (15,10)

MFCCs = librosa.feature.mfcc(y=file_name, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
plt.figure(figsize=FIG_SIZE)
librosa.display.specshow(MFCCs, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC coefficients")
plt.colorbar()
plt.title("MFCCs")

# show plots
plt.show()


