#look through the dataset
#remove the files that have the associated label "hasbird"

import os

goodfiles = []

os.chdir(f"resources/kaggle_datasets/dataset_3/")
with open("not_hasbird.csv", "r") as f:
    for row in f:
        #keep associated file in /wav
        goodfiles.append(row[:-1] + ".wav") 
# print(goodfiles)
q = 0
for i in os.listdir("wav"):
    if i not in goodfiles:
        print("removing file...", i)
        os.remove("wav/" + i)
        #nuke the rest
        q+=1
        print(q)
    else:
        print("keeping file...", i)


