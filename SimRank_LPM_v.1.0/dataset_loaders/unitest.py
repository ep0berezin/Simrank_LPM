import numpy as np
import csv
import pandas as pd
import sys
import time
import unidecode
import matplotlib.pyplot as plt 

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

def GetMatrix(path = "data/test_set_in.csv"): #Test case from original Jeh, Widom article.
	N=5
	data = pd.read_csv(path)
	M_ind = data['id']
	M_name = data['Name']
	M_conn = data['Connections']
	G = np.zeros((N,N))
	M_ind_copy = M_ind.copy()
	for i in M_ind:
		for j in M_ind_copy:
			if IsIn(str(j), M_conn[i-1]): #Making connection if connected
				G[j-1][i-1] = 1
	print("Trace-check:")
	print(np.trace(G))
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

def ObtainMatrix():
	A = norm1_ColumnNormalize(GetMatrix())
	return A

