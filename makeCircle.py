import imgLib as il
import numpy as np
import os

print("Pick a pgm file to make a reference library for")
file_name = input()
print("Experiment Directory to be created if one does not exist")
prj_dir = file_name[:-4] + "_Ex"  # name of the project directory
il.makeDir(prj_dir)
matList = il.makeMatrix(file_name)
mat = matList[0]
row = np.shape(mat)[0]
col = np.shape(mat)[1]
maxRadius = int(min(row, col)/2)  # #maximum radius to fit the picture
dec = max(1, int(maxRadius/25))
os.chdir(prj_dir)
for radius in range(dec, maxRadius, dec):
    out = il.makeCircle(row, col, radius, matList[1],
                        int(col/2), int(row/2), black=False)
    il.makeImage("circle.pgm", radius, 255, out)
