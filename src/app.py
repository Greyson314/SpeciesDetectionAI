"""
Take the trained model and apply it to analyze a passive acoustic monitoring file. 
The MVP is that this file takes 1) our trained model and 2) a passive acoustic monitoring file, and uses the PAM file as a test set.
This function should return the timestamps in the PAM file where the model believes there to be common terns (or whatever your species is) 
-Still operating in 30 second chunks.
-Offsetting [test frames] by 20 seconds to give it some overlap.  

take file path
load file, convert it to the dataframe

"""
from pydub import AudioSegment
import os
import math
import shutil
import keras
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import csv
from multiprocessing import Pool



model = None
def predictor(file, threshold = 0.6): #threshold = at what confidence point you want it to return True. 0.6 = anything over 60%
    data = preprocessing(file)
    result = model.predict(data) #predict this single file's label using the saved_model. 
    # print(result[0][0])
    if result[0][0] > threshold:
        confidence = result[0][0]
    else:
        confidence = 1-result[0][0]

    return result[0][0] > threshold, round(confidence*100, 2)

#A lot of this is repetitive from Murmur! We have to re-do the preprocessing on the 30s chunks of the PAM files in order for the model to be able to use it. 

hop_length = 512  # the default spacing between frames
n_fft = 2048  # number of samples, 2048
n_mfcc = 13
pad_length = 840000

def hms(seconds):
    h = seconds // 3600
    m = seconds % 3600 // 60
    s = seconds % 3600 % 60
    return '{:02d}:{:02d}:{:02d}'.format(h, m, s)

def padding_2(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]

    # print("h: ", h)
    # print("w: ", w)

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

def preprocessing(file_name): #the same as the Murmur preprocessing file, but without the copies. 
    # load the file
    wav_data, sr = librosa.load(file_name, sr=28000)

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
    return data

def test_frame_maker(pam_file, chunk_seconds = 30, offset_seconds = 20): #splits the code into chunk_seconds-second chunks. Offsets the chunks by offset_seconds. Default 30, 20
    file_name = pam_file[:-4]
    try:
        shutil.rmtree(file_name)
    except:
        pass
    os.mkdir(file_name)
    chunk = chunk_seconds*1000 #30 seconds
    offset = offset_seconds*1000

    oldAudio = AudioSegment.from_wav(f"{pam_file}") #grabs the PAM file
    num_divisions = math.floor(len(oldAudio)/offset) #figures out how many times to divide it up
    # print(num_divisions)
    t1 = 0
    t2 = t1 + chunk
    for x in range(num_divisions): #creates a new file representing every 30 second chunk. 
        newAudio = oldAudio[t1:t2] 
        newAudio.export(f'{file_name}/{x}.wav', format="wav") #Exports to a wav file in the current path.
        t1 += offset
        t2 = t1+chunk

def main():
    global model
    model = keras.models.load_model('saved_model', compile = True) #load our completed Murmur model!
    pam_file = 'pam_1.wav' #this is your Passive Acoustic Monitoring file. Rename it if need be! 
    folder_name = pam_file[:-4]
    # test_file = 25
    os.chdir(f"resources/kaggle_datasets/dataset_2/")
    chunk_seconds = 30 #slightly repetitive, sorry
    offset_seconds = 20 
    test_frame_maker(pam_file, chunk_seconds, offset_seconds)
    files = os.listdir(folder_name)
    files = sorted(files, key = lambda file: int(file[:-4])) #sorting by the number associated with each file (e.g. 0, 1, 2 instead of 1, 10, 100) (thanks stackoverflow)
    with open("timestamps.csv", "w", newline='', encoding='utf-8') as f: #create a CSV for our timestamps!
        writer = csv.writer(f)
        writer.writerow(["Prediction", "Confidence", "Start Time", "End Time", "File Number"])
        for file in files:
            result = predictor("pam_1/" + file) #THIS IS WHERE WE DO THE PREDICTION
            start_time = (int(file[:-4]))*offset_seconds
            end_time = start_time + chunk_seconds
            print(result[0], ", Confidence = ", result[1], "%, file =", file, "Start Time = ", hms(start_time)) #From the predictor it grabs the final results! HMS= hours mins secs
            writer.writerow([result[0], result[1], start_time, end_time, file]) #writes them in the CSV

    with open("timestamps.csv", "r") as f: 
        """
        The piece of my code that most resembles a data structures homework problem.
        Makes it so that if there are multiple "True" or "False" chunks in a row, it combines them + tells you the duration + start and end time.
        Took me 3 hours
        """
        reader = csv.reader(f)
        next(reader)
        len_reader = len(list(reader))
        f.seek(0)
        reader = csv.reader(f)
        next(reader) #repopulating the reader (just used it all for the len function lol)
        timestamp_outputs = []
        prev_bool = None
        curr_bool = None
        curr_start = None
        curr_end = None
        duration = chunk_seconds
        for index, line in enumerate(reader):
            curr_bool = line[0]
            if index != 0:
                if prev_bool == curr_bool: #if the previous stamp and current stamp are the same
                    duration += offset_seconds
                    curr_end += offset_seconds
                    output = [line[0], line[1], hms(int(curr_start)), hms(int(curr_end)), line[4], hms(int(duration))]

                else: #switching between true and false! this is where we add to the list. 
                    timestamp_outputs.append(output)
                    duration = chunk_seconds
                    prev_bool = curr_bool
                    curr_start = int(line[2])
                    curr_end = int(line[3])
                    output = [line[0], line[1], hms(int(line[2])), hms(int(line[3])), line[4], hms(int(duration))]

            if index == 0: #if first thing in list, just establishing our outputs
                prev_bool = line[0]
                curr_bool = line[0]
                curr_start = int(line[2])
                curr_end = int(line[3])
                output = [line[0], line[1], hms(int(line[2])), hms(int(line[3])), line[4], hms(int(duration))]


            if index == len_reader-1: #if last thing in the list
                print("Done.")
                timestamp_outputs.append(output)
        # print(timestamp_outputs)

    with open("final_output.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Prediction", "Confidence", "Start Time", "End Time", "Initial File", "Duration"])
        writer.writerows(timestamp_outputs)
            



if __name__ == "__main__":
    main()

