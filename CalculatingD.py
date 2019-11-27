import math
import numpy as np
import imagefunctions as imf
import sys

HOWMANYFILES = 300

def main():

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        print("Please provide the test file in the form image.pgm. ")
        return

    filetype, maxpixel, testArray = imf.readimage(filename)
    
    if filetype == 'P2':

        # Array to test
        basename = filename[:filename.rfind('.')]

        # Import library of faces and add to a list
        N = []
        for n in range(1, HOWMANYFILES+1):
            filename = '{0}.pgm'.format(n)
            filetype, maxpixel, array = imf.readimage(filename)
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
        while(total <= max_total): #until before k
            total += l[k]
            k += 1

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

        for j in range(W.shape[0]): 
            d_vector[j] = abs(W[j] - w)

        # Deciding cut-offs
        d = np.amin(d_vector)
        print("Our value for d is {0}.".format(d))

        dlow = 2000
        dhigh = 8000

        if(d > dhigh):
            print('High. Not a face.')
        elif(d < dlow):
            print('Low. Must be a picture already in the library.')
        else:
            print('Is a face. Add to library.')
        
    else:
        print("There was a problem with the input file.  No results printed.")
    
    return   



if __name__ == "__main__":
    main()
