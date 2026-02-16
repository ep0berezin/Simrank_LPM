import numpy as np
import csv
import pandas as pd
import sys
import time
import unidecode
import matplotlib.pyplot as plt
from memory_profiler import profile

#WARNING: GetMatrix only for non-directed graphs.
def GetMatrixFb(path, showfig, delimeter = ' '):
	n = 4039
	A = np.zeros((n,n))
	file = open(path,'r')
	for line in file:
		string = line.split(delimeter)
		node1, node2 = int(string[0]), int(string[1])
		A[node1,node2] = 1
	A = (A+A.T) #Omit this for directed graphs!
	if showfig:
		plt.figure()
		plt.imshow(A, cmap = 'binary')
		plt.show()
	print("Edges: ", sum(sum(A)))
	return A

def norm1_ColumnNormalize(M): #may be optimized! L1 col norms can be easily obtained by sum(A).
	col_1_norms = np.sum(np.abs(M), axis = 0)
	col_1_norms[col_1_norms == 0] = 1 #Avoid div by 0
	print("Columns 1-norms:")
	print(col_1_norms)
	normalized = M/col_1_norms
	print("Column 1-normalized matrix:")
	print (normalized)
	return normalized

def ObtainMatrix(path = "data/facebook_combined.txt", showfig=0):
	A = norm1_ColumnNormalize(GetMatrixFb(path=path, showfig=showfig))
	return A
