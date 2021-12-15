"""
Author: Grey Hutchinson
Date: 12/14/2021

This script is used to generate the model for the project.  
This model is what is used by app.py to make predictions.

It creates a rnn model that takes in a sequence of mfccs and
predicts if there is a common tern.

The inputs are the dataframe.csv file.  To see how dataframe.csv is
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

USE_CACHE = True
USE_MODEL_CACHE = False
epoch_count = 12000  # current best is 50-250
batch_size = 256  # current best is 255
learning_rate = 0.001
decay_rate = 0
make_plot = True

hop_length = 512  # the default spacing between frames
n_fft = 2048  # number of samples, 2048
n_mfcc = 13
pad_length = 840000


def padding_2(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]

    print("h: ", h)
    print("w: ", w)

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w
    return np.pad(array, pad_width=((a, aa), (b, bb)), mode="constant")


def padding(waveform):
    if waveform.shape[0] == pad_length:
        return waveform

    if waveform.shape[0] > pad_length:
        diff = (waveform.shape[0] - pad_length) // 2
        waveform = waveform[diff:][: (-1 * diff)]
        return waveform
        # get rid of diff number of columns from both sides

    if waveform.shape[0] < pad_length:
        diff = pad_length - waveform.shape[0]
        waveform = np.pad(waveform, diff // 2, mode="constant")
        return waveform


def get_features(df_in):
    bar = Bar("Getting Features", max=673)
    features = []  # list to save features
    labels = []  # list to save labels
    for index in range(0, len(df_in)):
        # get the filename
        filename = df_in.iloc[index]["id"]
        # save labels
        label = df_in.iloc[index]["label"]
        # load the file
        wav_data, sr = librosa.load(filename, sr=28000)

        #if the file is comter, duplicate and add random noise
        if label == True:
            copy_label = copy(wav_data)
            # add noise here
            noise=np.random.normal(0, 0.001, wav_data.shape[0])
            copy_data = np.array(
                [
                    padding_2(
                        librosa.feature.mfcc(
                            padding(copy_label+noise),
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
        # cut the file from tstart to tend
        data = np.array(
            [
                padding_2(
                    librosa.feature.mfcc(
                        padding(wav_data),
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
        bar.next()
        features.append(data)
        labels.append(label)
        if label == True:
            features.append(copy_data)
            labels.append(label)
    bar.finish()
    output = np.concatenate(features, axis=0)
    return (np.array(output), labels)


# features,y=get_features(df)

features, labels = None, None
if USE_CACHE:
    with open("Xy.pkl", "rb") as f:
        features, labels = pickle.load(f)
else:
    features, labels = get_features(df)
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
print("Splitting Training, Test, and Validation Sets...")
features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, test_size=0.25, random_state=231, stratify=labels
)
# features_train, features_val, labels_train, labels_val = train_test_split(features_train, labels_train, test_size=0.25, random_state=321) #with val

# Print the shapes
print("Data Shapes:")
# print(features_train.shape, features_test.shape, features_val.shape, len(labels_train), len(labels_test), len(labels_val)) #with val
print(features_train.shape, features_test.shape, len(labels_train), len(labels_test))

print("Creating Model...")

model = None
if USE_MODEL_CACHE:
    model = keras.models.load_model("saved_model")
    with open("history.pkl", "rb") as f:
        history_dict = pickle.load(f)
else:
    input_shape = (n_mfcc, 1641)
    model = keras.Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=False))
    # model.add(Dropout(0.2))

    # model.add(LSTM(64, input_shape=input_shape, return_sequences=True))

    # model.add(LSTM(32, input_shape=input_shape, return_sequences=True))

    # # model.add(LSTM(16, input_shape=input_shape, return_sequences=True))

    # # model.add(LSTM(32, input_shape=input_shape, return_sequences=True))

    # model.add(LSTM(64, input_shape=input_shape, return_sequences=True))

    # model.add(LSTM(128, input_shape=input_shape, return_sequences=False))


    # model.add(Dropout(0.2))
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
    model.add(Dense(1, activation="sigmoid"))
    model.summary()


    print("Compiling Model...")

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=epoch_count,
    decay_rate=decay_rate)

    tf.keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=0.0, nesterov=False, name="SGD"
    )
    # tf.keras.optimizers.RMSprop(learning_rate=0.01, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False, name="RMSprop")
    model.compile(optimizer="SGD", loss="BinaryCrossentropy", metrics=["acc", tf.keras.metrics.FalseNegatives(), tf.keras.metrics.FalsePositives()])

    print("Fitting the Model...")
    history = model.fit(
        features_train,
        labels_train,
        epochs=epoch_count,
        batch_size=batch_size,
        shuffle=True,
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


if make_plot:
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
# print('Confusion_matrix: ',tf.math.confusion_matrix(y_test, np.argmax(y_pred,axis=1)))


# pickle the model, then give it 10 terns and 10 non-terns, and use the model.predict on those files at the end.
