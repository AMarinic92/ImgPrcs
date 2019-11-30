import numpy as np
import math
import os

def uncannyEdge(matrix, maxPix, low, high):
    numrows, numcols = matrix.shape
    gv_array = gradient_vector(matrix)
    vl_array = np.clip(vector_length(gv_array,numrows,numcols), 0, maxPix) # Compute vector lengths
    et_array = np.clip(edge_thinning(gv_array,vl_array), 0, maxPix) # Apply edge-thinning
    final_array = np.clip(noise_reduction(et_array, low, high), 0, maxPix) # Reduce noise
    return final_array

def gradient_vector(array):
    '''
    This function creates of matrix of shape r, c, 2, storing the gradient vectors components of each pixel.
    The pixels are convolved with a central difference approximation kernel to find the direction with the
    greatest rate of change, and records these vector components in the same position as the pixel.
    '''
    # Get the dimensions of the pixel array
    numrows, numcols = array.shape

    # Create an empty matrix to contain the gradient vectors
    gv_array = np.zeros((numrows, numcols, 2))

    # The central difference kernel is linear of length 3, with fixed values [-1/2, 0, 1/2]
    # The central difference kernel sums to 0 (and we won't divide with it)
    # Asserting the properties of this kernel, I can hardcode values and ranges in the algorithm
    central_diff_kernel = np.array([-1/2, 0, 1/2])
    slice1 = np.array([-1/2, 0])
    slice2 = np.array([0, 1/2])

    # EDGE CASES
    for x in range(1, numrows-1): # Left and Right edges
        gv_array[x, 0, 0] = np.sum(slice2 * array[x,0:2])
        gv_array[x, 0, 1] = np.sum(central_diff_kernel * array[x-1:x+2,0])
        gv_array[x, numcols-1, 0] = np.sum(slice1 * array[x,numcols-2:numcols])
        gv_array[x, numcols-1, 1] = np.sum(central_diff_kernel * array[x-1:x+2,0])

    for y in range(1, numcols-1): # Top and Bottom edges
        gv_array[0, y, 0] = np.sum(central_diff_kernel * array[:0,y-1:y+2])
        gv_array[0, y, 1] = np.sum(slice2 * array[0:2,y])
        gv_array[numrows-1, y, 0] = np.sum(central_diff_kernel * array[:0,y-1:y+2])
        gv_array[numrows-1, y, 1] = np.sum(slice1 * array[numrows-2:numrows,y])

    # CORNER CASES
    gv_array[0, 0, 0] = np.sum(slice2 * array[0,0:2])
    gv_array[0, 0, 1] = np.sum(slice2 * array[0:2,0])
    gv_array[numrows-1, 0, 0] = np.sum(slice2 * array[numrows-1,0:2])
    gv_array[numrows-1, 0, 1] = np.sum(slice1 * array[numrows-2:numrows,0])
    gv_array[0, numcols-1, 0] = np.sum(slice1 * array[0,numcols-2:numcols])
    gv_array[0, numcols-1, 1] = np.sum(slice2 * array[0:2,numcols-1])
    gv_array[numrows-1, numcols-1, 0] = np.sum(slice1 * array[numrows-2:numrows,numcols-1])
    gv_array[numrows-1, numcols-1, 1] = np.sum(slice1 * array[numrows-1,numcols-2:numcols])

    # NON-EDGE CASES
    for x in range(1, numrows-1):
        for y in range(1, numcols-1):
            h_set = array[x-1:x+2, y]
            gv_array[x, y, 0] = np.sum(central_diff_kernel * h_set) # Delta x component
            v_set = array[x, y-1:y+2]
            gv_array[x, y, 1] = np.sum(central_diff_kernel * v_set) # Delta y component

    return gv_array


def vector_length(gv_array, numrows, numcols):
    '''
    This function takes a gradient vector array of shape r, c, 2 (and the numrows and numcols for quick referencing)
    and returns an array with the magnitudes of each vector as its pixel value.
    '''
    # Create an empty matrix of the same dimension as the original
    vl_array = np.zeros((numrows, numcols))

    # Compute magnitude of vector components
    for x in range(numrows):
        for y in range(numcols):
            vl_array[x, y] = math.hypot(gv_array[x, y, 0], gv_array[x, y, 1]) * 2

    return vl_array


def edge_thinning(gv_array, array):
    '''
    This function takes a gradient vector array of shape r, c, 2 with its corresponding r x c matrix, which
    the function will modify. The function applies an edge-thinning procedure to pixels that compose the edge
    by suppressing all except the brightest pixels. This procedure can be applied in the directions of x, y and
    along the slope of 1 and -1.
    '''
    # Create an empty matrix of the same dimension as the original
    numrows, numcols = array.shape

    for x in range(0, numrows):
        for y in range(0, numcols):
            angle = 0

            # Truncate if dividing by 0
            denom = gv_array[x, y, 0]
            if(denom == 0):
                angle = 90
            else:
                angle = math.degrees(math.atan(gv_array[x, y, 1] / denom))

            if(abs(angle) < 22.5): # Check x: left and right
                if(x-1 > -1 and x+1 < numrows):
                    if(array[x-1, y] > array[x, y] or array[x+1, y] > array[x, y]):
                        array[x, y] = 0
                elif(x-1 < 0): # Check right only
                    if(array[x+1, y] > array[x, y]):
                        array[x, y] = 0
                else: # Check left only
                    if(array[x-1, y] > array[x, y]):
                        array[x, y] = 0

            elif(abs(angle) > 67.5): # Check y: top and bottom
                if(y-1 > -1 and y+1 < numcols):
                    if(array[x, y-1] > array[x, y] or array[x, y+1] > array[x, y]):
                        array[x, y] = 0
                elif(y-1 < 0): # Check bottom only
                    if(array[x, y+1] > array[x, y]):
                        array[x, y] = 0
                else: # Check top only
                    if(array[x, y-1] > array[x, y]):
                        array[x, y] = 0

            elif(angle > 0): # Check slope 1: top-right and bottom-left
                if(x-1 > -1 and x+1 < numrows and y-1 > -1 and y+1 < numcols):
                    if(array[x+1, y+1] > array[x, y] or array[x-1, y-1] > array[x, y]):
                        array[x, y] = 0
                elif(x-1 < 0 and y-1 < 0): # Check bottom-right only
                    if(array[x+1, y+1] > array[x, y]):
                        array[x, y] = 0
                else: # Check top-left only
                    if(array[x-1, y-1] > array[x, y]):
                        array[x, y] = 0

            else: # Check slope -1: top-left and bottom-right
                if(x-1 > -1 and x+1 < numrows and y-1 > -1 and y+1 < numcols):
                    if(array[x-1, y+1] > array[x, y] or array[x+1, y-1] > array[x, y]):
                        array[x, y] = 0
                elif(x+1 >= numrows and y-1 < 0): # Check bottom-left only
                    if(array[x-1, y+1] > array[x, y]):
                        array[x, y] = 0
                else: # Check top-right only
                    if(array[x+1, y-1] > array[x, y]):
                        array[x, y] = 0


    return array


def noise_reduction(array, low, high):
    '''
    This function cleans an image with edge detection procedures applied by reducing extraneous
    noise from fainter edges. The first phase suppresses all pixels that fall under a certain brightness,
    and the second removes fainter edges that are too weak to stand alone.
    '''
    # Define thresholds
    maxpixel = np.amax(array)
    low_threshold = maxpixel * (low / 100)
    high_threshold = maxpixel * (high / 100)

    numrows, numcols = array.shape

    # EDGE CASES
    for x in range(1, numrows-1): # Left edge
        if(array[x, 0] < low_threshold): # Phase 1: Suppress pixels falling under the thresholds
            array[x, 0] = 0
        elif(array[x, 0] < high_threshold): # Phase 2: Suppress edges too weak to stand alone
            if(array[x+1,0] < high_threshold and array[x-1,0] < high_threshold and array[x,1] < high_threshold):
                array[x, 0] = 0

        if(array[x, numcols-1] < low_threshold): # Right edge
            array[x, numcols-1] = 0
        elif(array[x, numcols-1] < high_threshold):
            if(array[x+1,0] < high_threshold and array[x-1,0] < high_threshold and array[x,numcols-2] < high_threshold):
                array[x, numcols-1] = 0

    for y in range(1, numcols-1): # Top edge
        if(array[0, y] < low_threshold):
            array[0, y] = 0
        elif(array[0, y] < high_threshold):
            if(array[0,y+1] < high_threshold and array[0,y-1] < high_threshold and array[1,y] < high_threshold):
                array[0, y] = 0

        if(array[numrows-1, y] < low_threshold): # Bottom edge
            array[numrows-1, y] = 0
        elif(array[numrows-1, y] < high_threshold):
            if(array[numrows-1,y+1] < high_threshold and array[numrows-1,y-1] < high_threshold and array[numrows-2,y] < high_threshold):
                array[numrows-1, y] = 0

    # CORNER CASES
    if(array[0, 0] < low_threshold):
        array[0,0] = 0
    elif(array[0, 0] < high_threshold):
        if(array[0,1] < high_threshold and array[1,0] < high_threshold):
            array[0,0] = 0

    if(array[numrows-1, 0] < low_threshold):
        array[numrows-1,0] = 0
    elif(array[numrows-1, 0] < high_threshold):
        if(array[numrows-1,1] < high_threshold and array[numrows-2,0] < high_threshold):
            array[numrows-1,0] = 0

    if(array[0, numcols-1] < low_threshold):
        array[0,numcols-1] = 0
    elif(array[0, numcols-1] < high_threshold):
        if(array[0,numcols-2] < high_threshold and array[1,numcols-1] < high_threshold):
            array[0,numcols-1] = 0

    if(array[numrows-1,numcols-1] < low_threshold):
        array[numrows-1,numcols-1] = 0
    elif(array[numrows-1,numcols-1] < high_threshold):
        if(array[numrows-1,numcols-2] < high_threshold and array[numrows-2,numcols-1] < high_threshold):
            array[numrows-1,numcols-1] = 0

    # NON-EDGE CASES
    for x in range(1, numrows-1):
        for y in range(1, numcols-1):
            if(array[x, y] < low_threshold):
                array[x, y] = 0
            elif(array[x, y] < high_threshold):
                if(array[x+1, y] < high_threshold and array[x, y+1] < high_threshold
                   and array[x-1, y] < high_threshold and array[x, y-1] < high_threshold
                   and array[x+1, y+1] < high_threshold and array[x-1, y-1] < high_threshold
                   and array[x+1, y-1] < high_threshold and array[x-1, y+1] < high_threshold):
                    array[x, y] = 0

    return array


def convolve2D(array, kernel):
    # Decide whether to rescale the kernel after truncating
    if np.amax(kernel) > 0:
        renormalize = True
    else:
        renormalize = False

    # Assumes that the dimensions of the kernel have the form (2k+1) x (2k+1)
    kernel_size = kernel.shape[0]
    margin = int((kernel_size-1)/2)

    # Get the dimensions of the pixel array
    numrows, numcols = array.shape

    # The result has the same dimensions as the pixel array
    conv_array = np.zeros((numrows, numcols))

    # Iterate over the pixel array to perform the convolution
    for ai in range(numrows):
        # First indices for the slice of the pixel array
        pu=max(ai-margin,0)
        pd=min(ai+margin+1,numrows)
        # First indices for the slice of the kernel
        ku=max(0,margin-ai)
        kd=kernel_size-max(ai+margin+1-numrows,0)

        for aj in range(numcols):
            # Second indices for the slice of the pixel array
            pl=max(aj-margin,0)
            pr=min(aj+margin+1,numcols)
            # Second indices for the slice of the kernel
            kl=max(0,margin-aj)
            kr=kernel_size-max(aj+margin+1-numcols,0)

            if kl == 0 and ku == 0 and kr == kernel_size and kd == kernel_size:
                truncated = False
            else:
                truncated = True

            # Find the sum of the truncated kernel, if neeed
            if not truncated or not renormalize:
                kernel_sum = 1
            else:
                kernel_sum = np.sum(kernel[ku:kd,kl:kr])

            # Multiply the slice of the array by the slice of the kernel
            # component-wise, take the sum, renormalize if appropriate
            conv_array[ai,aj] = np.sum(array[pu:pd,pl:pr]*kernel[ku:kd,kl:kr])/kernel_sum

    return conv_array


def readimage(longname):
    # dummy values that are returned if the file cannot be opened
    filetype = 'none'
    maxpixel = 0
    array = np.array([0])

    try:
        image_file = open(longname)
    except:
        print("Failed to open the file named " + longname + ".")

    # list to hold all integer entries in the file
    longlist = []

    firstword = True
    for line in image_file:
        words = line.split();
        wi = 0
        comment = False
        while not comment and wi < len(words):
            word = words[wi]
            wi += 1
            if not word.startswith('#') and not firstword:
                # an entry that is not part of a comment and is not
                # the first word is an integer
                longlist.append(int(word))
            elif word.startswith('#'):
                # this is a comment
                # drop the rest of the line
                comment = True
            elif firstword:
                # the first word that is not a comment is the file type
                filetype = word
                firstword = False

    image_file.close()

    if filetype == 'P2':
        numcols = longlist[0] # number of columns in the image
        numrows = longlist[1] # number of rows in the image
        maxpixel = longlist[2] # maximum pixel value

        array = np.reshape(np.array(longlist[3:]), (numrows, numcols))

    elif filetype != 'none':
        # for the moment, only P2 files are supported
        print(filetype + " is not a recognized file type.")

    return filetype, maxpixel, array


def writeimage(longname, filetype, maxpixel, array):
    try:
        image_file = open(longname,'w')
    except:
        print("Failed to write to the file named " + longname + ".")
        return

    if filetype == "P2":
        # obtain the dimensions of the image from the shape of the array
        numrows, numcols = array.shape

        # write the file header
        image_file.write(filetype+'\n')
        image_file.write("{0} {1}\n".format(numcols,numrows))
        image_file.write("{}\n".format(int(maxpixel)))

        for i in range(numrows):
            for j in range(numcols):
                image_file.write("{} ".format(int(round(array[i,j]))))
            image_file.write('\n')

    else:
        print("This was not a P2 file.  No result was printed.")

    image_file.close()
    return


def mirror_lr(array):
    # reflect image from left to right
    return np.fliplr(array)


def mirror_ud(array):
    # reflect image from top to bottom
    return np.flipud(array)


def invert_array(array, maxpixel):
    # invert black and white
    return maxpixel - array


def brighten_or_darken(array, maxpixel, shift):
    # add shift to each pixel value, where shift can be a positive or negative integer
    array = array + shift

    # fix anywhere we exceeded maxpixel or got a negative number
    array = np.clip(array, 0, maxpixel)
    print(array)
    return array


def get_brightness(array, maxpixel, numcols, numrows):
	#  returns the percentage brightness of an image
	#  we choose this formula to accurately compare brightness of images with differing max pixel values
	percent = ((np.sum(array - maxpixel/2)) / (numrows * numcols) + maxpixel/2 ) / maxpixel * 100
	return percent


def calcD(testArray, maxpixel, libDirectory):
        # Import library of faces and add to a list
        N = []
        for filename in os.scandir(libDirectory):
            filetype, maxpixel, array = readimage(filename)
            N.append(np.ndarray.flatten(array).tolist())

        # Transform list into a matrix
        M = np.array(N)
        numrows, numcols = M.shape


        u = np.zeros((1, numcols))
        for y in range(numcols):
            u[0,y] = M[:,y].mean()
        L = M.copy()
        for x in range(numrows):
            L[x] = L[x] - u

        LT = L.copy().T
        L = np.matmul(L, L.T)

        # Derive the eigenvalues and eigenvectors
        l, v = np.linalg.eig(L)

        # Sort arrays in descending order
        for n in range(1, l.shape[0]):
            current = l[n]
            pos = n
            while pos > 0 and l[pos - 1] < current:
                l[pos] = l[pos - 1]
                v[:, pos] = v[:, pos - 1]
                pos -= 1
            l[pos] = current

        # Trim lower 5% of both sets
        max_total = np.sum(l) * 0.95
        total = 0
        k = 0
        # with k: (total + l[k] <= max_total)
        while(total < max_total): #until before k
            total += l[k]
            k += 1

        if k == 0:
            k = 1

        l = l[0:k]
        v = v[0:k]

        # Multiply LT to every eigenvector in v to get y
        y = np.zeros((LT.shape[0], k))
        for j in range(k):
            y[:,j] = np.matmul(LT, v[j])
            y[j] = np.linalg.norm(y[j], 1, keepdims = True)

        # For every row in L, get the dot product of that whole row with every eigenface in y
        W = y.copy()
        for c in range(LT.shape[1]):
            for j in range(k):
                W[j] = LT[:,c]@y[:,j]

        # Derive the weight of the test image
        t = np.ndarray.flatten(testArray.copy())
        v = t - u
        w = np.zeros((1, k))
        d_vector = W.copy()

        for j in range(k):
            w[0,j] = v@y[:,j]

        # Derive the distance
        for j in range(W.shape[0]):
            d_vector[j] = abs(W[j] - w)

        d = np.amin(d_vector)
        print("Our value for d is {0}.".format(d))

        return d



def calcD_all(testSlice, libDirectory):
    # Gets the distance of a slice of the original test array with each library image per se
    # This version of calcD will not need to derive weights for an "average ball"
    ballFound = False
    for filename in os.scandir(libDirectory):
        filetype, maxpixel, array = readimage(filename)
        if maxpixel < 255:
            array = auto_brighten(array, maxpixel)

        d = calcD_single(testSlice, array)
        print("The distance of slice with {0} is {1}.".format(filename, d))

        if d < 15:
            ballFound = True

    if ballFound:
        print("A ball had been detected in this slice")

    return ballFound



def calcD_single(testSlice, basisImage):
    # Helper for CalcD_all
    numrows, numcols = testSlice.shape
    diffArray = abs(testSlice - basisImage)
    d = np.sum(diffArray) / (numrows*numcols)
    return d



def auto_brighten(array, maxpixel):
    max = np.amax(array)
    min = np.amin(array)

    if max != min:
        numrows, numcols = array.shape
        for r in range(numrows):
            for c in range(numcols):
                ratio_upper = max - array[r,c]
                ratio_lower = array[r,c] - min

                # After getting the original ratio of the pixel's distance from the max and min pixel value
                # we can rescale the pixel relative to a new max and min pixel value
                # in this function, we are scaling the image to have its darkest value as 0
                # and its brightest value as 255
                step = 255 / (ratio_upper + ratio_lower)
                array[r,c] = math.ceil(step * ratio_lower)
    else:
        ratio = 255 / maxpixel
        array.fill(max * ratio)

    return array
