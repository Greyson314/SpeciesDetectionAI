"""
Author: Grey Hutchinson
Date: 12/14/2021

Given a directory of subdirectories with the name comter and nomter, this script
will generate a dataframe.csv file that contains the file paths and TRUE/FALSE
for whether the file is a common tern or not.

It can also write a 'short' version for quick testing.
"""
import os

os.chdir(f"resources/kaggle_datasets/dataset_2/")
with open("dataframe.csv", "w") as f:
    f.write("id,label\n")
    for i in os.listdir(f"comter/"):
        f.write(f"comter/{i},TRUE\n")
    for i in sorted(os.listdir(f"nomter/")):
        f.write(f"nomter/{i},FALSE\n")

with open("dataframe_short.csv", "w") as f:
    f.write("id,label\n")
    x = 0
    for i in os.listdir(f"comter/"):
        if x % 10 == 0:
            f.write(f"comter/{i},TRUE\n")
        x += 1
    x = 0
    for i in sorted(os.listdir(f"nomter/")):
        if x % 10 == 0:
            f.write(f"nomter/{i},FALSE\n")
        x += 1
