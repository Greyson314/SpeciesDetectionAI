import os
import random
import shutil
import csv

def singles():
    base_dir = "C:/Users/greys/Dropbox/thesis/resources/kaggle_datasets/dataset_2/A-Z"
    # print(os.listdir(base_dir))
    x = 0
    for dir in os.listdir(base_dir):
        file = random.choice(os.listdir(base_dir + "/" + dir))
        if dir == "comter":
            continue
        shutil.copyfile(base_dir + "/" + dir + "/" + file, "C:/Users/greys/Dropbox/thesis/resources/kaggle_datasets/dataset_2/nomter/" + file)
        print(file)
        x += 1
    print(x)
# singles()

def giveclass():
    base_dir = "C:/Users/greys/Dropbox/thesis/resources/kaggle_datasets/dataset_2/"
    with open(base_dir + "labeled_data.csv", "w", newline = "") as f:
        writer = csv.writer(f)
        for filename in os.listdir(base_dir + "nomter"):
            writer.writerow(["nomter/" + filename, "False"])
        for filename in os.listdir(base_dir + "comter"):
            writer.writerow(["comter/" + filename, "True"])        

giveclass()
        

