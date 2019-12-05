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
'''
#   This section is for selecting Uncanny edge detection thresholds
#   If a picture with these thresholds exists we convert it to an np array
#   Else we make it
'''
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


'''
#  This block handles the trials, their directories,  and the images in them
#  It also runs the indvidual trials on each slice
'''


# picking how many trials you want to do will do finer scans each trial. The
# first trial is just looking at the image as whole
print("Please enter the number of trials you would like to attempt.\n",
      "Each trial breaksdown an image into smaller submatrices to scan\n",
      "starting with the largest square as the first submatrix, each trial\n",
      "breaks down the image into smaller squares. A value less than 1 runs\n",
      "the entire image against an entire directory, it is very CPU intensive",
      "\n")
trials = int(input())
bSqr = int(min(row, col))  # what is the biggest square for first scan
'''
#   Here the individual trials are run. We test our slice agaisnt a library of
#   images with varying position
'''

if trials < 1:
    '''
    #   This is for later implemantation but currently is too cpu intensive
    #   It can still be used if used appriopriatly sized image like a small
    #   cell or slices of a cell. It allows for a custom directory to be fed in
    '''
    print("Testing entire image against library\n",
          "WARNING PROCESS VERY CPU INTESIVE FOR LARGE IMAGE\n",
          "Type yes/YES/y/Y to continue\n")
    proceed = input()
    if (proceed == "YES" or proceed == "y" or proceed == "Y"
       or proceed == "yes"):
        tDir = input('input test directory name')
        il.adv_calcD(mat, maxPix, tDir)
        print("Testing entire image against each in library\n")
        # imf.calcD_all(mat, tDir)
'''
# This runs the difference calculations for each appropriate slice in each
# trial one trial may have multiple slices taken from it
'''
for t in range(1, trials+1):
    # performing tests against a computer generated library of circles
    if t > 1:
        # 1st trial will scan with bSqr, all other trials will shrink
        # bSqr and use that
        bSqr = int(bSqr*.75)
    hSqr = int(bSqr/2)  # half of bSqr
    # else make library of bSqr
    trow = hSqr*2  # can be turned into rectangular positions in later use
    tcol = hSqr*2
    tname = "trial_{0}".format(t)  # used for naming an calling differnt libs
    otname = "trial_{0}_Overload".format(t)
    if bSqr % 2 > 0:
        #  odd square
        imf.find_radii(mat, tname)
        # the following line of code is a safe way to go back to the parent dir
        # on most modern OS even though ".." should work on the important ones
        os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir))
        il.overloadLib(otname, trow+1, tcol+1, maxPix)
        os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir))
    else:
        #  even square
        imf.find_radii(mat, tname)
        os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir))
        il.overloadLib(otname, trow, tcol, maxPix)
        os.chdir(os.path.normpath(os.getcwd() + os.sep + os.pardir))
    if t > 0:
        # all trials now scan instead of testing entire image. Largest radius
        # ball can only be as big as smallest dimension
        inc = int(bSqr/4)  # we will increment by a quarter of the square
        hSqr = int(bSqr/2)  # half the square for calcs
        print("We need to iterate over sub matrices and run trials")
        x = hSqr
        y = hSqr
        #  while loops allow us to make larger jumps if a ball is found
        while y < row-hSqr+1:
            while x < col-hSqr+1:
                cord = "({0}, {1})".format(y, x)
                if bSqr % 2 > 0:
                    #  if an odd square
                    submat = mat[y-hSqr:y+hSqr+1, x-hSqr:x+hSqr+1]
                else:
                    #  else an even square
                    submat = mat[y-hSqr:y+hSqr, x-hSqr:x+hSqr]
                print("\nTest trial {0} ".format(t),
                      "at the (y,x) coordinate: {0} ".format(cord),
                      "looking at a {0} x {0} square\n".format(bSqr))
                # test against entire library
                imf.calcD(submat, maxPix, (otname+"_Lib"))
                print("Testing against individual images in library\n")
                # test against indvidual images
                found = imf.calcD_all(submat, (tname+"_Lib"))
                if found:
                    # if we find a ball lets move over further in the x cord
                    x = x + hSqr
                    # If we find an image here we can add our submat to our
                    # library to increase our test accuracy and library of
                    # balls
                else:
                    # else we didnt find one lets scan finer in the x cord
                    x = x + inc
            # allow us to progress
            y = y + inc
print("~~~~~~~~~~~~Process Completed~~~~~~~~~~~~")
