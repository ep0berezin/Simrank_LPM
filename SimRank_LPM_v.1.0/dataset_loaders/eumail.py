import numpy as np
import csv
import pandas as pd
import sys
import time
import unidecode
import matplotlib.pyplot as plt
from memory_profiler import profile

#WARNING: GetMatrix only for non-directed graphs.
def GetMatrix(path, showfig, delimeter = ' '):
	n = 1005
	A = np.zeros((n,n))
	#data = pd.read_csv(path, delimeter = ',')

	file = open(path,'r')
	for line in file:
		string = line.split(delimeter)
		if (string[0]!=""):
			#print(string)
			node1, node2 = int(string[0].strip()), int(string[1].strip())
			if (node1!=node2):
				A[node1,node2] = 1.0
	#A = (A+A.T) #Omit this for directed graphs!
	#print("Edges: ", sum(sum(A)))
	if showfig:
		plt.figure()
		plt.imshow(A, cmap = 'binary')
		plt.show()
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

def ObtainMatrix(path = "data/email-Eu-core.txt", showfig=0):
	A = norm1_ColumnNormalize(GetMatrix(path=path, showfig=showfig))
	return A
