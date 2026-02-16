import numpy as np
import csv
import pandas as pd
import sys
import time
import unidecode
import matplotlib.pyplot as plt 
from matplotlib.ticker import FixedLocator, FixedFormatter

def IsIn(j, M): #j = '12', nbs = '112 123'
	res = 0
	M = unidecode.unidecode(M)
	M_lst = M.split(' ')
	#print(M_lst)
	M_int = [int(a) for a in M_lst]
	for a in M_int:
		if (a == int(j)):
			res = 1
	return res

def GetMatrixMetro(path, showfig):
	N = 303
	data = pd.read_csv(path)
	M_ind = data['Station_index']
	M_name = data['English_name']
	M_cpy = M_ind.copy()
	M_nbs = data['Line_Neighbors']
	M_trs = data['Transfers']
	M_trs = M_trs.fillna('0') #filling NaNs with '0'
	G = np.zeros((N,N))

	for i in M_ind:
		for j in M_cpy:
			if ( IsIn(str(j) , M_nbs[i-1]) or IsIn(str(j) , M_trs[i-1]) ): #Making connection if connected or transfers
				G[i-1][j-1] = 1 #i,j or j,i?
	
	print("Trace-check before nullifiyng:")
	print(np.trace(G))
	
	for i in range(len(G)):
		if (G[i,i] > 0):
			G[i,i] = 0 #Nullifying self-transfers: self-transfers must be omitted.
	
	print("Trace-check after nullifiyng:")
	print(np.trace(G))
	if showfig:
		plt.figure()
		plt.imshow(G, cmap = 'binary')
		plt.show()
	return G

def norm1_ColumnNormalize(M): #may be optimized! L1 col norms can be easily obtained by sum(A).
	col_1_norms = np.sum(np.abs(M), axis = 0)
	col_1_norms[col_1_norms == 0] = 1 #Avoid div by 0
	print("Columns 1-norms:")
	print(col_1_norms)
	normalized = M/col_1_norms
	print("Column 1-normalized matrix:")
	print (normalized)
	return normalized

def ObtainMatrix(path="data/metro_stations.csv", showfig=0):
	A = norm1_ColumnNormalize(GetMatrixMetro(path=path, showfig=showfig))
	return A
