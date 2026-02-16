import numpy as np
import csv
import pandas as pd
import sys
import time
import unidecode
import matplotlib.pyplot as plt
from memory_profiler import profile

#WARNING: GetMatrix only for non-directed graphs.
def GetMatrixRnd():
	n=1000
	l = 0
	h = 10
	density = 0.1
	A = np.random.randint(low=l, high=h, size=(n,n)) #demo matrix
	for i in range(A.shape[0]):
		for j in range(A.shape[1]):
			if (A[i,j]>=(h-l)*density):
				A[i,j] = 0
			else:
				A[i,j] = 1
	print("Random adjacency matrix:")
	print(A)
	plt.figure()
	plt.imshow(A, cmap = 'binary')
	plt.show()
	return A

def GetOnes():
	n=100
	A = np.ones((n,n))
	print("Random matrix:")
	print(A)
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

def ObtainMatrix():
	A = norm1_ColumnNormalize(GetMatrixRnd())
	return A
