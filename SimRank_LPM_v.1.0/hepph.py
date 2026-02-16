import numpy as np
import scipy.sparse as scsp
import pandas as pd
import sys
import matplotlib.pyplot as plt

def getidlist(path, delimeter='\t'):
	print(f"Getting nodedict...")
	file = open(path, 'r')
	numnodes = int( file.readline().split(delimeter)[0].strip() )
	nodelist = []
	for line in file:
		string =  line.split(delimeter)
		#print(string)
		node1, node2 = int(string[0].strip()), int(string[1].strip())
		#print(f"node1 id = {node1}, node2 id = {node2}")
		if node1 not in nodelist:
			nodelist.append(node1)
		if node2 not in nodelist:
			nodelist.append(node2)
	file.close()
	
	print(f"numodes = {numnodes}, nodelist length = {len(nodelist)}")
	truenodelist = list(range(numnodes))
	nodedict = dict(zip(nodelist,truenodelist))
	print(f"Nodedict done.")
	return nodedict, numnodes

#WARNING: GetMatrix only for non-directed graphs.
def GetMatrix(path, showfig, delimeter = '\t', savecoo=True):
	nodedict, n = getidlist(path)
	print(f"Construcitng adjacency matrix in coo...")
	file = open(path, 'r')
	file.readline()
	row = []
	col = []
	for line in file:
		string =  line.split(delimeter)
		node1, node2 = int(string[0].strip()), int(string[1].strip())
		row.append(nodedict[node1])
		col.append(nodedict[node2])
	file.close()
	data = [1.]*len(row)
	A_coo = scsp.coo_matrix((data, (row,col)), shape=(n,n))
	if savecoo:
		np.save("data/hepph_data.npy", np.array(data))
		np.save("data/hepph_row.npy", np.array(row))
		np.save("data/hepph_col.npy", np.array(col))
	print(f"Constructing completed. nnz = {A_coo.nnz}.")
	return A_coo

def LoadMatrix(n):
	return scsp.coo_matrix((np.load("data/hepph_data.npy"), (np.load("data/hepph_row.npy"), np.load("data/hepph_col.npy"))), shape=(n,n))

def norm1_ColumnNormalize(M): #may be optimized! L1 col norms can be easily obtained by sum(A).
	col_1_norms = np.sum(np.abs(M), axis = 0)
	col_1_norms[col_1_norms == 0] = 1 #Avoid div by 0
	print("Columns 1-norms:")
	print(col_1_norms)
	normalized = M/col_1_norms
	print("Column 1-normalized matrix:")
	print (normalized)
	return normalized

def ObtainMatrix(path = "data/Cit-HepPh-prepared.txt", showfig=1):
	#A = norm1_ColumnNormalize(GetMatrix(path=path, showfig=showfig))
	A = norm1_ColumnNormalize(LoadMatrix(n=34546))
	return A.tocsr()
