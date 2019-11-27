import imgLib as il
import numpy as np
import os

print("Pick a pgm file to make a reference library for")
file_name = input()
print("Experiment Directory to be created if one does not exist\n")
prj_dir = file_name[:-4] + "_Exp"  # name of the project directory
il.makeDir(prj_dir)
matList = il.makeMatrix(file_name)
os.chdir(prj_dir)
mat = matList[0]
maxPix = matList[1]
row = np.shape(mat)[0]
col = np.shape(mat)[1]
print("Please select a low threshold for noise\n",
      "reduction in uncanny edge detect")
low = int(input())
print("Please select a high threshold for noise\n",
      "reduction in uncanny edge detect")
high = int(input())
found = False
checkName = file_name + '_Uncanny_' + str(low) + '_' + str(high)
for files in os.scandir(path='.'):
    if(files == checkName):
        found = True
if not found:
    print("No Uncanny Edge Detection for picture and thresholds, making one")
    il.makeImage(file_name, "Uncanny_{0}_{1}".format(str(low), str(high)),
                 maxPix, il.uncannyEdge(mat, maxPix, low, high))
else:
    print("Uncanny edge detect for image already exsists")

print("Please enter the number of trials you would like to attempt.\n",
      "Each trial breaksdown an image into smaller submatrices to scan\n",
      "starting with the entire image as the first submatrix, each trial\n",
      "breaking down the image by half each time",
      "\n(ie  4 trials  = last trial scans 9 quardrants)\n")
trials = int(input())
for t in range(1, trials+1):
    il.makeCircLib(file_name[:-4], int(row/t), int(col/t), maxPix)

print("~~~~~~~~~~~~Process Completed~~~~~~~~~~~~")
