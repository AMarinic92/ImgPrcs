import imgLib as il  # Andrew Marinic's image function library
import imagefunctions as imf  # Andrea's Abellera's image function library
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
checkName = file_name[:-4] + '_Uncanny_' + str(low) + '_' + str(high) + ".pgm"
print(checkName)
if not os.path.exists(checkName):
    print("No Uncanny Edge Detection for picture and thresholds, making one")
    il.makeImage(file_name, "Uncanny_{0}_{1}".format(str(low), str(high)),
                 maxPix, il.uncannyEdge(mat, maxPix, low, high))
else:
    print("Uncanny edge detect for image already exsists.\n We can just",
          "convert it to a np.array for use")
    matList = il.makeMatrix(checkName)
    mat = matList[0]

print("Please enter the number of trials you would like to attempt.\n",
      "Each trial breaksdown an image into smaller submatrices to scan\n",
      "starting with the entire image as the first submatrix, each trial\n",
      "breaking down the image by half each time",
      "\n(ie  4 trials  = last trial scans 9 quardrants)\n")
trials = int(input())
trow = row  # trial row
tcol = col  # trial column
for t in range(1, trials+1):
    if t > 1:
        if trow >= tcol:
            trow = int(trow/2)
        else:
            tcol = int(tcol/2)
    il.makeCircLib(file_name[:-4], trow, tcol, maxPix)
    print(os.getcwd())
    if t > 1:
        print("need to iterate over sub matrices and run trials")
    else:
        imf.calcD(mat, maxPix, os.getcwd())
    os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir))
    print(os.getcwd())
print("~~~~~~~~~~~~Process Completed~~~~~~~~~~~~")
