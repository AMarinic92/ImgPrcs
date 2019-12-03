import numpy as np
import re
import math
import os

# imageLibrary - Sci2000 - T01 - Andrew Marinic - 7675509


def overloadLib(name, row, col, maxPix):
    maxRadius = int(min(row, col)/2)  # maximum radius to fit the picture
    inc = max(1, int(maxRadius/5))  # increments
    lib_dir = name + "_Lib"
    exsist = makeDir(lib_dir)
    os.chdir(lib_dir)  # we will enter the directory if it was created or not
    if not exsist:
        # if the directory was just created it must be empty so lets create the
        # images for the library
        for radius in range(inc, maxRadius, inc):
            out = makeCircle(row, col, radius, maxPix, int(col/2), int(row/2),
                             black=False)
            imgNm = str(radius) + "_" + str(int(col/2)) + "_" + str(int(row/2))
            imgNm = imgNm + ".pgm"
            makeImage(imgNm, "", 255, out)
            for x in range(radius, col-radius+1, radius*2):
                for y in range(radius, row-radius+1, radius*2):
                    out = makeCircle(row, col,
                                     radius, maxPix, x, y, black=False)
                    imgNm = str(radius) + "_" + str(x) + "_" + str(y) + ".pgm"
                    makeImage(imgNm, "", 255, out)


# This is what I use for generating a new folder and a bunch of different
# radius ceneterd circles. name is used for making the folder row and col are
# dimensions and maxPix is the maxPixel value of the circles
def makeCircLib(name, row, col, maxPix):
    maxRadius = int(min(row, col)/2)  # #maximum radius to fit the picture
    inc = max(1, int(maxRadius/25))  # increments
    lib_dir = name + "_" + str(maxRadius) + "_Lib"
    makeDir(lib_dir)
    os.chdir(lib_dir)
    for radius in range(inc, maxRadius, inc):
        out = makeCircle(row, col, radius, maxPix, int(col/2), int(row/2),
                         black=False)
        imgName = str(radius) + ".pgm"
        makeImage(imgName, "", 255, out)


# this function makes a circle matrix that is a ring of the selected colour
# either black(if true) on a white background or white(if false) on a black
# background. matRow and matCol are the dimension of the matrix, radius is the
# radius if the desired ring, maxPix is the maxPixel value of the image, xCord
# and yCord is where you want the image to be centerd, black is described above
def makeCircle(matRow, matCol, radius, maxPix, xCord, yCord, black=True):
    mat = np.zeros((matRow, matCol))
    midX = xCord
    midY = yCord
    if xCord < int(radius/2) or xCord > matCol-int(radius/2):
        midX = int(matCol/2)
    if yCord < int(radius/2) or yCord > matRow-int(radius/2):
        midY = int(matRow/2)
    """
    By default, the pixels making up the circle are set to 0.  If black is
    set to False, the pixels making how to test if I can connect to serverup
    the circle are set to MAXPIXEL.
    """
    inradius = int(radius*.99)
    for ay in range(-radius, radius):
        # ay is the y-coordinate, measured with respect to the center
        # of the circle.
        # at height ay, the horizontal distance from the y-axis to the
        # boundary of the circle is sqrt(r^2 - ay^2)

        boundx = math.sqrt(radius*radius - ay*ay)
        for ax in range(-radius, radius):
            if -boundx < ax and ax < boundx:
                if black:
                    mat[midY+ay, midX+ax] = 0
                else:
                    mat[midY+ay, midX+ax] = maxPix
    for ay in range(-inradius, inradius):
        # ay is the y-coordinate, measured with respect to the center
        # of the circle.
        # at height ay, the horizontal distance from the y-axis to the
        # boundary of the circle is sqrt(r^2 - ay^2)

        boundx = math.sqrt(inradius*inradius - ay*ay)
        for ax in range(-inradius, inradius):
            if -boundx < ax and ax < boundx:
                if not black:
                    mat[midY+ay, midX+ax] = 0
                else:
                    mat[midY+ay, midX+ax] = maxPix
    return mat


#  uncanny edge detection algo, input a matrix and and upper and lower
#  threshold for noise reduction returns a np.array
def uncannyEdge(matrix, maxPix, low, high):
    out = guassianBlur(matrix, maxPix)
    out = np.clip(10*centDiffConvolve(out), 0, maxPix)
    out = thinEdge(out)
    return noiseThresh(out, maxPix, low, high)


# this function gets a sub matrix of size n x n at posistion y, x of the value
# inputted
# matric and returns an appropriate sub matrix usually used with nditer
def getSub(matrix, y, x, n):
    row = np.shape(matrix)[0]
    col = np.shape(matrix)[1]
    if n <= row and n <= col:
        left = max(0, x - int(n/2))
        right = min(col, x+int(n/2)+1)
        up = max(0, y - int(n/2))
        down = min(row, y + int(n/2)+1)
        return matrix[up:down, left:right]


# This function produces a guassian blur of the image with a sigma of 1.5
def guassianBlur(matrix, maxPix):
    kern = twoDKernel(maxPix, 3)
    return twoDConvolve(matrix, kern)


# This function applies the central difference kernel to an image
def centDiffConvolve(matrix):
    kern = np.array([-.5, 0, .5])
    return horiConvolve(matrix, kern)


# This function is a edge thinning process for the uncanny edge detection
def thinEdge(matrix):
    row = np.shape(matrix)[0]
    col = np.shape(matrix)[1]
    delX = 0
    delY = 0
    out = []
    outVal = 0
    with np.nditer(matrix, flags=['multi_index']) as loop:
        for pix in loop:
            x = loop.multi_index[1]
            y = loop.multi_index[0]
            if (x > 0 and x < col-1) and (
                    y > 0 and y < row-1):
                delX = (matrix[y, x+1] - matrix[y, x-1])/2
                delY = (matrix[y+1, x] - matrix[y-1, x])/2
            else:
                if x == 0:
                    delX = (matrix[y, x+1] - pix)
                elif x == col-1:
                    delX = (pix - matrix[y, x-1])
                else:
                    delX = (matrix[y, x+1] - matrix[y, x-1])/2
                if y == 0:
                    delY = (matrix[y+1, x] - pix)
                elif y == row-1:
                    delY = (pix - matrix[y-1, x])
                else:
                    delY = (matrix[y+1, x] - matrix[y-1, x])/2
            theta = abs(45 * round((np.arctan2(delY, delX)*(180/math.pi))/45))
            if (theta == 180):
                theta = 0
            outVal = pix  # set it to pix val and only change to 0 when needed
            #  compare left and right pix
            if theta == 0:
                if x > 0 and x < col-1:
                    if matrix[y, x-1] > pix or matrix[y, x+1] > pix:
                        outVal = 0
                else:
                    if x == 0 and matrix[y, x+1] > pix:
                        outVal = 0
                    if x == col-1 and matrix[y, x-1] > pix:
                        outVal = 0
            #  compare slope 1
            if theta == 45:
                if x > 0 and y > 0 and x < col-1 and y < row-1:
                    if matrix[y-1, x+1] > pix or matrix[y+1, x-1] > pix:
                        outVal = 0
                else:
                    if (x == 0 and y > 0) and matrix[y-1, x+1] > pix:
                        outVal = 0
                    if x == (col-1 and y < row-1) and matrix[y+1, x-1] > pix:
                        outVal = 0
                    if y == 0 and x > 0:
                        if matrix[y+1, x-1] > pix:
                            outVal = 0
                    if y == row-1 and x < col-1:
                        if matrix[y-1, x+1] > pix:
                            outVal = 0
            #  compare verticle
            if theta == 90:
                if (y > 0 and y < row-1) and (matrix[y+1, x] > pix or
                                              matrix[y-1, x] > pix):
                    outVal = 0
                else:
                    if y == 0 and matrix[y+1, x] > pix:
                        outVal = 0
                    if y == row-1 and matrix[y-1, x] > pix:
                        outVal = 0

            #  compare slope -1
            if theta == 135:
                if x > 0 and y > 0 and x < col-1 and y < row-1:
                    if(matrix[y-1, x-1] > pix or matrix[y+1, x+1] > pix):
                        outVal = 0
                else:
                    if (x == 0 and y < row-1) and matrix[y+1, x+1] > pix:
                        outVal = 0
                    if (x == col-1 and y > 0) and matrix[y-1, x-1] > pix:
                        outVal = 0
                    if (y == 0 and x < col-1) and matrix[y+1, x+1] > pix:
                        outVal = 0
                    if (y == row-1 and x > 0) and matrix[y-1, x-1] > pix:
                        outVal = 0
            out.append(outVal)
    return np.array(out).reshape(matrix.shape)


# this function compares a pixel to a threshold set from a max pixel
# it is used to reduce noise. It takes in a np.array, maxPix val and upper and
# lower thresholds as percentages omitting the % eg ..., 10, 50) would
# translate to 10% and 50%
def noiseThresh(matrix, maxPix, upper, lower):
    out = []
    outVal = 0
    high = maxPix*(upper/100)
    low = maxPix*(lower/100)
    with np.nditer(matrix, flags=['multi_index']) as loop:
        for pix in loop:
            outVal = pix
            x = loop.multi_index[1]
            y = loop.multi_index[0]
            if(pix < lower):
                outVal = 0
            elif pix < upper and pix > lower:
                outVal = noiseHelper(matrix, x, y, high, low)
            out.append(outVal)
    return np.array(out).reshape(matrix.shape)


# in an attempt to prevent what my error checker is calling too complex of
# functions I broke this up into multiple functions. It still complains
def noiseHelper(matrix, x, y, high, low):
    row = np.shape(matrix)[0]
    col = np.shape(matrix)[1]
    pix = matrix[y, x]
    outVal = 0
    if (x > 0 and y > 0) and (x < col-1 and y > row-1):
        if matrix[y-1, x-1] >= high:
            outVal = pix
        if matrix[y-1, x] >= high:
            outVal = pix
        if matrix[y-1, x+1] >= high:
            outVal = pix
        if matrix[y, x-1] >= high:
            outVal = pix
        if matrix[y, x+1] >= high:
            outVal = pix
        if matrix[y+1, x-1] >= high:
            outVal = pix
        if matrix[y+1, x] >= high:
            outVal = pix
        if matrix[y+1, x+1] >= high:
            outVal = pix
    elif x == 0 and (y > 0 and y < row-1):
        if matrix[y-1, x] >= high:
            outVal = pix
        if matrix[y-1, x+1] >= high:
            outVal = pix
        if matrix[y, x+1] >= high:
            outVal = pix
        if matrix[y+1, x] >= high:
            outVal = pix
        if matrix[y+1, x+1] >= high:
            outVal = pix
    elif x == col-1 and (y > 0 and y < row-1):
        if matrix[y-1, x-1] >= high:
            outVal = pix
        if matrix[y-1, x] >= high:
            outVal = pix
        if matrix[y, x-1] >= high:
            outVal = pix
        if matrix[y+1, x-1] >= high:
            outVal = pix
        if matrix[y+1, x] >= high:
            outVal = pix
    elif y == 0 and (x > 0 and x < col-1):
        if matrix[y, x-1] >= high:
            outVal = pix
        if matrix[y, x+1] >= high:
            outVal = pix
        if matrix[y+1, x-1] >= high:
            outVal = pix
        if matrix[y+1, x] >= high:
            outVal = pix
        if matrix[y+1, x+1] >= high:
            outVal = pix
    elif y == row-1 and (x > 0 and x < col-1):
        if matrix[y-1, x-1] >= high:
            outVal = pix
        if matrix[y-1, x] >= high:
            outVal = pix
        if matrix[y-1, x+1] >= high:
            outVal = pix
        if matrix[y, x-1] >= high:
            outVal = pix
        if matrix[y, x+1] >= high:
            outVal = pix
    elif x == 0 and y == 0:
        if matrix[y, x+1] >= high:
            outVal = pix
        if matrix[y+1, x] >= high:
            outVal = pix
        if matrix[y+1, x+1] >= high:
            outVal = pix
    elif x == 0 and y == row-1:
        if matrix[y-1, x] >= high:
            outVal = pix
        if matrix[y-1, x+1] >= high:
            outVal = pix
        if matrix[y, x+1] >= high:
            outVal = pix
    elif x == col-1 and y == 0:
        if matrix[y, x-1] >= high:
            outVal = pix
        if matrix[y+1, x] >= high:
            outVal = pix
        if matrix[y+1, x-1] >= high:
            outVal = pix
    elif x == col-1 and y == row-1:
        if matrix[y-1, x-1] >= high:
            outVal = pix
        if matrix[y-1, x] >= high:
            outVal = pix
        if matrix[y, x-1] >= high:
            outVal = pix
    return outVal


# this function is supposed to do a horizontal 1-dimension convoltion of a
# matrix with a kernel and returns a new matrix
def horiConvolve(matrix, kernel):
    kernLen = kernel.shape[0]
    out = []
    col = 0
    row = 0
    with np.nditer(matrix,
                   flags=['multi_index'], op_flags=['readwrite']) as loop:
        for x in loop:
            col = loop.multi_index[1]
            row = loop.multi_index[0]
            if((col > int(kernLen/2)) and
               (col < matrix.shape[1] - int(kernLen/2))):
                subMat = matrix[row, (col - int(kernLen/2)):
                                (col + int(kernLen/2)+1)]
                if(np.mean(kernel) > 0):
                    out.append(
                        int(abs(np.mean(subMat*kernel)*(1/np.mean(kernel)))))
                else:
                    out.append(
                        int(abs(np.mean(subMat*kernel))))
            else:
                left = max(0, col - int(kernLen/2))
                right = min(matrix.shape[1]-1, col + int(kernLen/2))
                subMat = matrix[row, left:right+1]
                kLeft = max(0, col - left)
                kRight = min(kernLen, right-int(kernLen/2)+1)
                subKern = kernel[kLeft:kRight]
                if(np.mean(kernel) > 0):
                    out.append(
                        int(abs(np.mean(subMat*subKern)*(1/np.mean(subKern)))))
                else:
                    out.append(int(abs(np.mean(subMat*subKern))))
        out = np.array(out).reshape(matrix.shape)
        return out

# a two-dimension convolution function takes in a matrix and kernel as a np
# array and returns a new modified array


def twoDConvolve(matrix, kernel):
    matCol = matrix.shape[1]
    matRow = matrix.shape[0]
    kRow = kernel.shape[0]
    kCol = kernel.shape[1]
    kLen = int(kRow/2)
    out = []
    with np.nditer(matrix, flags=['multi_index'],
                   op_flags=['readwrite']) as loop:
        for x in loop:
            posX = loop.multi_index[1]
            posY = loop.multi_index[0]
            if((posX >= kLen and posY >= kLen) and (posX < (matCol - kLen) and
                                                    posY < (matRow - kLen))):
                subMat = getSub(matrix, posY, posX, kRow)
                if np.mean(kernel) > 0:
                    outVal = np.mean((subMat*kernel))*(1/(np.mean(kernel)))
                else:
                    outVal = np.mean((subMat*kernel))
                out.append(abs(outVal))
            else:
                mLeft = max(0, posX - kLen)
                mRight = min(matCol, posX+kLen)  # +1 because up to not inc
                mUp = max(0, posY - kLen)
                mDown = min(matRow, posY+kLen)  # +1 because up to not inc
                kLeft = max(0, posX-mLeft)
                kRight = min(kCol, mRight-posX+1)
                kUp = max(0, posY-mUp)
                kDown = min(kRow, mDown-posY+1)
                subMat = matrix[mUp:mDown, mLeft:mRight]
                subKern = kernel[kUp:kDown, kLeft:kRight]
                if np.mean(subKern) > 0:
                    outVal = np.mean(subMat*subKern)*(1/np.mean(subKern))
                else:
                    outVal = np.mean(subMat*subKern)
                out.append(abs(outVal))  # should this be absVal?
    out = np.array(out).reshape(matrix.shape)
    return out

# this is a helper function that makes a One Dimensional kernel with sigma
# value of 1.5 as required for assigment 3. It uses the genGaussOneDK which has
# a variable


def oneDKernel(maxPix, nVal):
    return genGaussOneDK(1.5, maxPix, nVal)


# this is a helper function to make a square 2d kernel with a sigma value of
# 1.5 as required for assigment 3. It uses the genGaussTwoDK to retrun a kernel
# as a np array


def twoDKernel(maxPix, nVal):
    return genGaussTwoDK(1.5, nVal, nVal, maxPix)


# generates a 2 dimension Keneral with a gaussian distribiton. The function
# requires a siga value, desired rows, desired colums, and the maxPix(el)
# value to normalize the distribution. Returns a numpy array


def genGaussTwoDK(sigma, row, col, maxPix):
    # rows represent our x vaule, and col represents our y value
    kernel = []
    AVALUE = 1/(math.tau*(sigma**2))
    for x in range(-int(col/2), int(col/2)+1):
        for y in range(-int(row/2), int(row/2)+1):
            if x == 0 and y == 0:
                kernel.append(AVALUE*maxPix)
            else:
                exponent = -((x**2)+(y**2))/(2*(sigma**2))
                kernel.append(AVALUE*maxPix*math.e**exponent)
    kernel = np.array(kernel)
    kernel.shape = (row, col)

    kernel = (1/(np.sum(kernel)-1))*kernel  # normalize the kernel

    return kernel


genGaussTwoDK(1.5, 3, 3, 256)

# generates a 1 Dimension gaussian distrubtion based weighted kernel
# takes in a sigma value, a maxpixel to weight results to sum to 1, and
# an nValue which represents how many pixels to the left and right of the pixel
# to be adveraged.


def genGaussOneDK(sigma, maxPix, nValue):
    kernel = []
    AVALUE = 1/math.sqrt(math.tau*(sigma**2))
    for x in reversed(range(-1*nValue, nValue+1)):
        if(x == 0):
            kernel.append(AVALUE*maxPix)
        else:
            exponent = (-x**2)/(2*(sigma**2))
            value = AVALUE*maxPix*np.exp(exponent)  # np.exp uses eulers number
            kernel.append(value)
    kernel = np.array(kernel)
    kernel = (1/(np.sum(kernel)-1))*kernel  # normalizes the kernel
    return kernel


#  returns the average pixel value divided by max pixel to account for
#  comparing images with different max values, then turn it into a precentage
#  usually needs a helper method to make the code less messy


def getBrightness(matrix, maxPixel):
    return (matrix.mean(axis=None, dtype=float)/maxPixel)*100


# this takes in the file (hopefully either a pgm pbm or ppm) and turns it into
# a numpy array. The function returns as a list with [0] np.array, [1] max
# pixel value, [2] rows, [3] col, [4] the fileName. The code also cleans the
# file of all comments because they would make a mess


def makeMatrix(fileName):
    imageIn = open(fileName)  # opens said file
    imgMat = re.findall(r'\b\d+\b', imageIn.read())  # find all the numbers`
    col = int(imgMat[0])  # 1st numbers should now be the colum followed by row
    row = int(imgMat[1])
    maxPix = int(imgMat[2])  # grabbing the max pixel
    del imgMat[0:3]  # we no longer need them in the matrix
    imgMat = np.array(imgMat, dtype=int)
    imgMat = np.reshape(imgMat, [row, col])
    imageIn.close()
    return imgMat, maxPix, row, col, fileName


# makeImage takes in a file name, a string to modify the name, the max pixel
# value of the matrix, and the np.array of the matrix. It then writes the file
# withe the appropriate header based on the file type and size of np.array
def makeImage(name, nameMod, maxPixel, matrix):
    if nameMod != "":
        out = open('{0}_{1}{2}'.format(name[0:-4], nameMod, name[-4:]), 'w')
    else:
        out = open('{0}{1}'.format(name[:-4], name[-4:]), 'w')
    if(name[-4:] == '.pbm'):
        out.write('P1 \n')
    if(name[-4:] == '.pgm'):
        out.write('P2 \n')
    if(name[-4:] == '.ppm'):
        out.write('P3 \n')
    res = matrix.shape
    out.write('{0} {1} \n{2} \n'.format(res[1], res[0], maxPixel))
    np.savetxt(out, matrix, fmt='%.0f', delimiter=' ', newline=' \n',
               encoding=str)
    out.close()


# takes in (name of file, a max pixel value, and a np.array)filps the image
# from lest to right then writes the file using the above makeImage function
def makeFlipLR(name, maxPixel, matrix):
    flipMatrix = np.fliplr(matrix)
    makeImage(name, 'lrFlip', maxPixel, flipMatrix)


# takes in a (name of file, max pixel value, and a np.array) flips the image up
# and down then writes the file using the above makeImage fucntion
def makeFlipUD(name, maxPixel, matrix):
    flipMatrix = np.flipud(matrix)
    makeImage(name, 'udFlip', maxPixel, flipMatrix)


# takes in (name of file, a max pixel value, and a np.array) inverts the
# imageby subtracting the max pixel and then taking the absolute value of the
# resulting matrix(array)
def makeInvert(name, maxPixel, matrix):
    invertMatrix = np.absolute((matrix-maxPixel))
    makeImage(name, 'invert', maxPixel, invertMatrix)


# takes in (name of file, max pixel value, the value amount to be changed, and
# the np.array) element wise the value is added to the array and then
# elementwise the resulting matrix is checked for values over the max pixel and
# replaced
def makeBright(name, maxPixel, value, matrix):
    brightMatrix = np.add(matrix, value)
    brightMatrix = np.minimum(brightMatrix, maxPixel)
    makeImage(name, 'brighterby{}'.format(value), maxPixel, brightMatrix)


# takes in (name of file, max pixel value, the value amount to be changed, and
# the np.array) element wise the value is subtracted from the array and then
# elementwise the resulting matrix is checked for values over under 0 and
# replaced
def makeDark(name, maxPixel, value, matrix):
    darkMatrix = np.subtract(matrix, value)
    darkMatrix = np.maximum(darkMatrix, 0)
    makeImage(name, 'darkerby{}'.format(value), maxPixel, darkMatrix)


# Makes a new directory if one does not exsists and returns false otherwise if
# the directory exsists it returns true and prints the fact that the directory
# exists
def makeDir(name):
    exists = False
    if not os.path.exists(name):
        os.mkdir(name)
        print("A new directory was created for: ", name, "\n")
    else:
        print("Directiory: ", name, ", already exsists",
              "no need to create another\n")
        exists = True
    return exists
