"""
Author: Grey Hutchinson
Date: 12/14/2021

This script is used to generate the model for the project.  
This model is what is used by app.py to make predictions.

It creates a rnn model that takes in a sequence of mfccs and
predicts whether there is a common tern.

The input is the dataframe.csv file. To see how dataframe.csv is
generated, see the dataframe_maker.py script.
"""
# Std lib Imports
import os
from copy import copy
import pickle
import warnings

# Used to show progress of the script
from progress.bar import Bar

# Used to generate the mfccs
import librosa
import librosa.display

# Used to do basic data analysis
import pandas as pd
import numpy as np

# Used to create the models
import keras
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense

# Used to plot the results (visualization)
import matplotlib.pyplot as plt

# Used to train the model
from sklearn.model_selection import train_test_split



warnings.filterwarnings("ignore")

# os.chdir("resources/kaggle_datasets/dataset_2/")
df = pd.read_csv("dataframe.csv")
print(df.head())

USE_CACHE = False #Set to TRUE if you would like to skip preprocessing. Only works after having been run with set to FALSE at least once.
USE_MODEL_CACHE = False #Set to TRUE if you would like to skip the modelling step and use your previously cached model. Only works after having been run with set to FALSE at least once.

#MODEL PARAMETERS:
epoch_count = 12000  # current best is >10,000, takes about 30 minutes
batch_size = 256  # current best is 256
learning_rate = 0.001 # current best is 0.001
decay_rate = 0 # Mess with it if you want to but it sucks
make_plot = True # Leave as true 

hop_length = 512  # the default spacing between frames, current best is 512
n_fft = 2048  # number of samples, current best is 2048
n_mfcc = 13 # Number of MFCC 
pad_length = 840_000 #Set to your sample rate * your desired file length. 840,000 = 28,000 * 30, for example. 


def preprocess(df_in):
    bar = Bar("Getting Features", max=673) #just a progress bar
    features = []  # list to save features
    labels = []  # list to save labels
    for index in range(0, len(df_in)):
        # get the filename
        filename = df_in.iloc[index]["id"]
        # save labels
        label = df_in.iloc[index]["label"]
        # load the file
        wav_data, sr = librosa.load(filename, sr=28000)

        #if the file is a true (i.e. your target species), duplicate and add random noise to help the algorithm learn
        if label == True:
            copy_label = copy(wav_data)
            # add noise here
            noise=np.random.normal(0, 0.001, wav_data.shape[0])
            copy_data = np.array(
                [
                    padding_2(
                        librosa.feature.mfcc( #Fournier transforms your WAV files to be a numerical array of size n_mels (y) by 1641 (x). This is what the NN uses to learn. 
                            padding(copy_label+noise), #pads your wav file to be 30 seconds
                            n_fft=n_fft,
                            hop_length=hop_length,
                            n_mfcc=n_mfcc,
                            n_mels=128,
                        ),
                        n_mfcc,
                        1641, #length of 30s file MFCC. No idea how to calculate it, but it will tell you if you're wrong. 
                    )
                ]
            )
        # cut the file from tstart to tend
        data = np.array(
            [
                padding_2(
                    librosa.feature.mfcc( #Fournier transforms your WAV files to be a numerical array of size n_mels (y) by 1641 (x). This is what the NN uses to learn. 
                        padding(wav_data), #pads your wav file to be 30 seconds
                        n_fft=n_fft,
                        hop_length=hop_length,
                        n_mfcc=n_mfcc,
                        n_mels=128,
                    ),
                    n_mfcc,
                    1641,
                )
            ]
        )
        # takes about 12 minutes as of 10/4
        print(data.shape)
        os.system("cls")
        bar.next() #just progress bar stuff
        features.append(data) #add our array, which we created from the sound files, to an array of arrays. Hooray!
        labels.append(label) #assign said array a label--yay or nay. 
        if label == True:
            features.append(copy_data) #add the copied noisy data
            labels.append(label) #label the copied noisy data
    bar.finish()
    output = np.concatenate(features, axis=0) #turn the arrays into an NP array for processing
    return (np.array(output), labels)


def padding_2(array, xx, yy): #this is here to pad the Y axis, but I tried getting rid of it's X padding and it broke everything. 
    """
    pad the array
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w
    return np.pad(array, pad_width=((a, aa), (b, bb)), mode="constant")


def padding(waveform): 
    if waveform.shape[0] == pad_length: #if the length == pad length, return itself
        return waveform

    if waveform.shape[0] > pad_length: #if it's longer, take the difference, divide it by 2, then chop that off both sides to preserve the middle. 
        diff = (waveform.shape[0] - pad_length) // 2
        waveform = waveform[diff:][: (-1 * diff)]
        return waveform
        # get rid of diff number of columns from both sides

    if waveform.shape[0] < pad_length: #if it's shorter than the pad_length, pad each side with 0s
        diff = pad_length - waveform.shape[0]
        waveform = np.pad(waveform, diff // 2, mode="constant")
        return waveform


features, labels = None, None #this is where the pickling happens. 
if USE_CACHE:
    with open("Xy.pkl", "rb") as f:
        features, labels = pickle.load(f)
else:
    features, labels = preprocess(df)
    with open("Xy.pkl", "wb") as f:
        pickle.dump([features, labels], f)

labels = [
    1 if x else 0 for x in labels
]  # converting all "False" to 0 and all "True" to 1

# print(labels)

print("Normalizing Data...") 
features = np.array(
    (features - np.min(features)) / (np.max(features) - np.min(features))
)
features = features / np.std(features)
labels = np.array(labels)

# Split twice to get the validation set
print("Splitting Training and Test Sets...")
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.25, random_state=231, stratify=labels
)

# Print the shapes
print("Data Shapes:")
print(features_train.shape, features_test.shape, len(labels_train), len(labels_test))

print("Creating Model...")

model = None
if USE_MODEL_CACHE: #accesses the model cache 
    model = keras.models.load_model("saved_model")
    with open("history.pkl", "rb") as f:
        history_dict = pickle.load(f)
else:
    input_shape = (n_mfcc, 1641) #128x1641 array being used to train our model. Each array has 128*1641=210,000 features
    model = keras.Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=False)) #Uncomment at your leisure, but make sure return_sequences is true for all but the last layer. 
                                                                            #also only use LSTM layers or dropout layers; and I don't recommend dropouts. 1 layer LSTM rocks btw
    # model.add(LSTM(64, input_shape=input_shape, return_sequences=True))

    # model.add(LSTM(32, input_shape=input_shape, return_sequences=True))

    # # model.add(LSTM(16, input_shape=input_shape, return_sequences=True))

    # # model.add(LSTM(32, input_shape=input_shape, return_sequences=True))

    # model.add(LSTM(64, input_shape=input_shape, return_sequences=True))

    # model.add(LSTM(128, input_shape=input_shape, return_sequences=False))


    # model.add(LSTM(32, input_shape=input_shape, return_sequences=True))
    # # model.add(Dropout(0.2))
    # model.add(LSTM(16, input_shape=input_shape, return_sequences=True))
    # # model.add(Dropout(0.2))
    # model.add(LSTM(32, input_shape=input_shape, return_sequences=True))
    # # model.add(Dropout(0.2))
    # model.add(LSTM(128, input_shape=input_shape, return_sequences=False))



    # model.add(LSTM(256, input_shape=input_shape, return_sequences=False))
    # model.add(Dropout(0.5))
    # model.add(Dropout(0.4))
    # model.add(Dense(264, activation="relu"))
    # model.add(Dropout(0.4))

    model.add(Dense(1, activation="sigmoid")) #sigmoid gives us a 1 or 0!
    model.summary()


    print("Compiling Model...")
    tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=0.0, nesterov=False, name="SGD" #setting our optimizer. SGD is the best, but you can also try "adam"
    )
    # tf.keras.optimizers.RMSprop(learning_rate=0.01, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False, name="RMSprop")
    model.compile(optimizer="SGD", loss="BinaryCrossentropy", metrics=["acc", tf.keras.metrics.FalseNegatives(), tf.keras.metrics.FalsePositives()]) #compiling model!

    print("Fitting the Model...")
    history = model.fit(
        features_train,
        labels_train,
        epochs=epoch_count,
        batch_size=batch_size,
        shuffle=True, #shuffles data every epoch. Keep true. 
    )
    model.save("saved_model")
    history_dict = history.history
    with open("history.pkl", "wb") as f:
        pickle.dump(history_dict, f)

# history_dict=history.history
loss_values = history_dict["loss"]
acc_values = history_dict["acc"]
fp_values = history_dict["false_positives"]
fn_values = history_dict["false_negatives"]
epochs = range(1, epoch_count + 1)


if make_plot: #just making a gross little plot for the model. Very helpful, don't click the X when it pops up. Click the save button. :)
    print("Creating Plot...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 5))
    ax1.plot(epochs, loss_values, "co", label="Training Loss")
    # ax1.plot(epochs,val_loss_values,'m', label='Validation Loss')
    ax1.set_title("Training loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs, acc_values, "co", label="Training accuracy")
    ax2.set_title("Training accuracy")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    ax3.plot(epochs, fp_values, "co", label="False Positives")
    ax3.set_title("False Positives")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Frequency")
    ax3.legend()

    ax4.plot(epochs, fn_values, "co", label="False Negatives")
    ax4.set_title("False Negatives")
    ax4.set_xlabel("Epochs")
    ax4.set_ylabel("Frequency")
    ax4.legend()

    plt.show()


print("Evaluating Model...")
TrainLoss, Trainacc = model.evaluate(features_train, labels_train)
TestLoss, Testacc = model.evaluate(features_test, labels_test)
print("\nTest accuracy: ", Testacc)

print("Predicting...")
labels_pred = model.predict(features_test)

# print("Prediction[0]: ", y_pred[0])
# print('Confusion_matrix: ',tf.math.confusion_matrix(y_test, np.argmax(y_pred,axis=1))) #confusion matrix is really annoying for BinaryCrossentropy, just check accuracy


#Now the model is ready to be applied in app.py! If you aren't happy with your model results, feel free to run it again. I recommend setting USE_CACHE to True at this point
#that will speed it up a lot. 