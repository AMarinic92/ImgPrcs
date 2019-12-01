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
maxPix = matList[1]
mat = imf.auto_brighten(matList[0], maxPix)  # our matrix we will be testing
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
                 maxPix, imf.uncannyEdge(mat, maxPix, low, high))
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
      "breaks down the image into smaller squares")
trials = int(input())
bSqr = int(min(row, col))  # what is the biggest square for first scan
for t in range(1, trials+1):
    # performing tests against a computer generated library of circles
    if t > 2:
        # 2nd trial will scan with bSqr, all other trials will shrink
        # bSqr and use that
        bSqr = int(bSqr*.75)
    if t == 1:
        # if we are doing first trial testing entire image
        trow = row
        tcol = col
    else:
        # else make library of bSqr
        trow = bSqr
        tcol = bSqr
    tname = "trial_{0}".format(t)
    il.overloadLib(tname, trow, tcol, maxPix)
    if t > 1:
        # if this is not our first trial we are testing slices
        inc = int(bSqr/4)  # we will increment by a quarter of the square
        hSqr = int(bSqr/2)  # half the square for calcs
        print("We need to iterate over sub matrices and run trials")
        # I want to change to while loops so we can take a bigger jump if we
        # find a ball that way we dont test the same general area twice for the
        # same ball
        for x in range(hSqr, col - hSqr+1, inc):
            for y in range(hSqr, row - hSqr+1, inc):
                cord = "({0}, {1})".format(y, x)
                submat = il.getSub(mat, y, x, bSqr)
                print("\nTest trial {0} ".format(t),
                      "at the (y,x) coordinate: {0} ".format(cord),
                      "looking at a {0} x {0} square\n".format(bSqr))
                # test against entire library
                imf.calcD(submat, maxPix, os.getcwd())
                print("Testing against individual images in library\n")
                # test against indvidual images
                imf.calcD_all(submat, os.getcwd())
        os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir))
    else:
        # else we are testing the image as a whole
        print("Testing entire image against library of circles\n")
        imf.calcD(mat, maxPix, os.getcwd())
        print("Testing entire image against each in library\n")
        imf.calcD_all(mat, os.getcwd())
    os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir))
print("~~~~~~~~~~~~Process Completed~~~~~~~~~~~~")
