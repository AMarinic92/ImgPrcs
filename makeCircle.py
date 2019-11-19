import math
import numpy as np
import imgLib as il


def makeCircle(matRow, matCol, radius, maxPix, xCord, yCord, black=True):
    mat = np.zeros((matRow, matCol))
    midX = xCord
    midY = yCord
    if xCord < int(radius/2) or xCord > matCol-int(radius/2):
        midX = int(matCol/2)
    if yCord < int(radius/2) or yCord > matRow-int(radius/2):
        midY = int(matRow/2)
    """
    circledata is a 3-tuple containing the radius of the circle and the x- and
    y- coordinates of the center of the circle.  All of these values should be
    integers.
    By default, the pixels making up the circle are set to 0.  If black is
    set to False, the pixels making how to test if I can connect to serverup
    the circle are set to MAXPIXEL.
    """
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
    return mat


out = makeCircle(150, 150, 25, 255, 26, 75, black=False)
il.makeImage("circle.pgm", "", 255, out)
