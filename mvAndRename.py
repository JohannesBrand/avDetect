import random
import glob, os
from shutil import copyfile

# path = "../Pictures/skitour+site:gipfelbuch.ch/"
pTestData = 0.05 # percent of data used for test
picturePath = "../Pictures/"
trainPath = "../train_pics/"
testPath = "../test_pics/"

classes = ["avalanche","no_avalanche"]

for cf in classes:
    folders = os.listdir(picturePath+cf)
    for fo in folders:
        fopath = picturePath + cf + "/" + fo + "/"
        files = glob.glob(fopath + "*.jpg")
        random.shuffle(files)
        for i, infile in enumerate(files):
            path, file = os.path.split(infile)
            if i < pTestData*len(files):
                copyfile(infile, testPath + cf + "/" + fo + file)
            else:
                copyfile(infile, trainPath + cf + "/" + fo + file)