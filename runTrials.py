import imgLib as il  # Andrew Marinic's image function library
import imagefunctions as imf  # Andrea Abellera's image function library
import numpy as np
import os

file_name = input("Pick a pgm file to make a reference library for\n")
print("Experiment Directory to be created if one does not exist\n")
prj_dir = file_name[:-4] + "_Exp"  # name of the project directory
il.makeDir(prj_dir)
matList = il.makeMatrix(file_name)
os.chdir(prj_dir)
mat = imf.auto_brighten(matList[0])  # our matrix we will be testing
maxPix = np.amax(mat)  # after brightening grabbing the new largest value
row = np.shape(mat)[0]  # rows of test image
col = np.shape(mat)[1]  # columns of test image

# Selecting thresholds for uncanny
print("Please select a low threshold for noise\n",
      "reduction in uncanny edge detect")
low = int(input())
print("Please select a high threshold for noise\n",
      "reduction in uncanny edge detect")
high = int(input())
# lets not make more images than we need, if the source image with the the
# selected thresholds exsist lets just convert it to a np.array
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


# picking how many trials you want to do will do finer scans each trial. The
# first trial is just looking at the image as whole
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
    if t > 1:
        print("need to iterate over sub matrices and run trials")
        hCol = int(tcol/2)  # half column
        hRow = int(trow/2)  # half row

        for x in range(hCol, col-hCol,
                       int(tcol/4)):
            for y in range(hRow, row-hRow,
                           int(trow/4)):
                submat = mat[y-hRow:y+hRow, x-hCol:x+hCol]
                imf.calcD(submat, maxPix, os.getcwd())
        os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir))
    else:
        imf.calcD(mat, maxPix, os.getcwd())
    os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir))
print("~~~~~~~~~~~~Process Completed~~~~~~~~~~~~")
