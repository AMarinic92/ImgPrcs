import imgLib as il
import numpy as np

print("Pick a pgm file to make a reference library for")
file_name = input()
matList = il.makeMatrix(file_name)
mat = matList[0]
row = np.shape(mat)[1]
col = np.shape(mat)[0]
maxRadius = min(row, col)  # #maximum radius to fit the picture
dec = max(1, int(maxRadius/10))
for radius in range(dec, maxRadius, dec):
    out = il.makeCircle(row, col, radius, matList[1],
                        int(col/2), int(row/2), black=True)
    il.makeImage("circle.pgm", radius, 255, out)
