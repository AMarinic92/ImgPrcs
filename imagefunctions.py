import numpy as np
import math


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

