import imgLib as il
import sys
import time

start = time.process_time()
file_name = sys.argv[1]
info = il.makeMatrix(file_name)
mat = info[0]
maxPix = info[1]
low = int(input('Low Threshold Value: '))
high = int(input('High High Threshold Value: '))
print(high, low)
il.makeImage(file_name, "Uncanny_{0}_{1}".format(low, high),
             maxPix, il.uncannyEdge(mat, maxPix, low, high))
end = time.process_time()
print('Process took {} seconds'.format(end-start))
