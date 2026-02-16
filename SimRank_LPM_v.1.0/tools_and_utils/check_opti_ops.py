import numpy as np
import scipy as sp
import scipy.sparse as scsp
import metro as metro

def edmmp(X,Y,Z): #X (m x p), Y (p x m), Z (m x n) ; Effective Diagonal Matrix-Matrix Product : get diag(XY)Z without forming full (m x m) XY.
	Z_copy = Z.copy()
	for k in range(X.shape[0]):
		dotp = X[k,:]@Y[:,k]
		Z_copy[k,:] = Z[k,:]*dotp
	return Z_copy

def edmmp__(X,Y,Z): #X (m x p), Y (p x m), Z (m x n) ; Effective Diagonal Matrix-Matrix Product : get diag(XY)Z without forming full (m x m) XY.
	Z_copy = Z.copy()
	Y_copy = Y.copy()
	X_copy = X.copy()
	
	if scsp.issparse(X_copy):
		X_copy = X_copy.tocsr()
	if scsp.issparse(Y_copy):
		Y_copy = Y_copy.tocsr()
	if scsp.issparse(Z_copy):
		Z_copy = Z_copy.tocsr()

	print(f"Z_copy = {type(Z_copy)}, X_copy = {type(X_copy)}, Y = {type(Y_copy)}")
	if not scsp.issparse(Z_copy):
		#print("Z = np.ndarray")
		for k in range(X.shape[0]):
			if scsp.issparse(X_copy):
				vecl = X_copy[k,:].toarray()
			else:
				vecl = X_copy[k,:]
			if scsp.issparse(Y_copy):
				vecr = Y_copy[:,k].toarray()
			else:
				vecr = Y_copy[:,k] 
			dotp = vecl@vecr
			Z_copy[k,:] = Z_copy[k,:]*dotp
	if scsp.issparse(Z_copy):
		#print("Z = csr_matrix")
		for k in range(X.shape[0]):
			if scsp.issparse(X_copy):
				vecl = X_copy[k,:].toarray()
			else:
				vecl = X_copy[k,:]
			if scsp.issparse(Y_copy):
				vecr = Y_copy[:,k].toarray()
			else:
				vecr = Y_copy[:,k]
			dotp = vecl@vecr
			Z_copy.data[Z_copy.indptr[k]:Z_copy.indptr[k+1]] *= dotp
	return Z_copy


def F_conj_F_UUTU(A, c, U):

	T1_real = ( off(U@U.T) - c*off(A.T@off(U@U.T)@A) )@U 
	T2_real = ( off(A@off(U@U.T)@A.T) )@U 
	T3_real = ( off(A@off(A.T@off(U@U.T)@A)@A.T) )@U
	T3_vykl = ( A@A.T@U@U.T@A@A.T  # ok
	- A@A.T@diag(U@U.T)@A@A.T
	- A@diag(A.T@U@U.T@A)@A.T
	+ A@diag(A.T@diag(U@U.T)@A)@A.T
	- diag(A@A.T@U@U.T@A@A.T)
	+ diag(A@A.T@diag(U@U.T)@A@A.T)
	+ diag(A@diag(A.T@U@U.T@A)@A.T)
	- diag(A@diag(A.T@diag(U@U.T)@A)@A.T) )@U
	
	T1 =( U@(U.T@U)  #ok
	- edmmp(U,U.T,U) 
	- c*(A.T@U)@((U.T@A)@U) 
	+ c*A.T@edmmp(U,U.T,A)@U 
	+ c*edmmp((A.T@U),(U.T@A),U) 
	- c*edmmp(A.T, edmmp(U,U.T,A),U) )
	T2 = ( (A@U)@(U.T@A.T)@U  #ok
	- A@edmmp(U,U.T,A.T)@U 
	- edmmp((A@U),(U.T@A.T),U) 
	+ edmmp(A, edmmp(U, U.T, A.T),U) )
	
	q1 = (A@A.T@U)@((U.T@A@A.T)@U) #ok
	q2 = (A@A.T)@edmmp(U, U.T, A)@(A.T@U)
	q3 = A@edmmp(A.T@U,U.T@A, A.T@U)
	q4 = A@edmmp(A.T, edmmp(U,U.T,A),A.T@U)
	q5 = edmmp(A@A.T@U, U.T@A@A.T,U)
	#q6 = edmmp(A@A.T, edmmp(U,U.T,A@A.T),U)
	q6 = edmmp(A,A.T@edmmp(U,U.T,A)@A.T,U)
	q7 = edmmp(A, edmmp(A.T@U,U.T@A,A.T),U)
	q8 = edmmp(A, edmmp(A.T, edmmp(U,U.T,A),A.T),U)
	T3 = q1-q2-q3+q4-q5+q6+q7-q8


	print(f"Checks:")
	print(f"T1: {np.linalg.norm(T1_real - T1)}")
	print(f"T2: {np.linalg.norm(T2_real - T2)}")
	print(f"T3: {np.linalg.norm(T3_real - T3)}")
	print(f"T3_vykl: {np.linalg.norm(T3_real - T3_vykl)}")
	res = T1-c*T2+c**2*T3
	return res

def F_conj_F_XYZ(A, c, X, Y, Z):

	T1_real = ( off(X@Y) - c*off(A.T@off(X@Y)@A) )@Z
	T2_real = ( off(A@off(X@Y)@A.T) )@Z
	T3_real = ( off(A@off(A.T@off(X@Y)@A)@A.T) )@Z
	T3_vykl = ( A@A.T@X@Y@A@A.T  # ok
	- A@A.T@diag(X@Y)@A@A.T
	- A@diag(A.T@X@Y@A)@A.T
	+ A@diag(A.T@diag(X@Y)@A)@A.T
	- diag(A@A.T@X@Y@A@A.T)
	+ diag(A@A.T@diag(X@Y)@A@A.T)
	+ diag(A@diag(A.T@X@Y@A)@A.T)
	- diag(A@diag(A.T@diag(X@Y)@A)@A.T) )@Z
	
	T1 =( X@(Y@Z)  #ok
	- edmmp(X,Y,Z) 
	- c*(A.T@X)@((Y@A)@Z) 
	+ c*A.T@edmmp(X,Y,A)@Z 
	+ c*edmmp((A.T@X),(Y@A),Z) 
	- c*edmmp(A.T, edmmp(X,Y,A),Z) )
	T2 = ( (A@X)@(Y@A.T)@Z  #ok
	- A@edmmp(X,Y,A.T)@Z
	- edmmp((A@X),(Y@A.T),Z) 
	+ edmmp(A, edmmp(X, Y, A.T),Z) )
	
	q1 = (A@A.T@X)@((Y@A@A.T)@Z) #ok
	q2 = (A@A.T)@edmmp(X, Y, A)@(A.T@Z)
	q3 = A@edmmp(A.T@X,Y@A, A.T@Z)
	q4 = A@edmmp(A.T, edmmp(X,Y,A),A.T@Z)
	q5 = edmmp(A@A.T@X, Y@A@A.T,Z)
	#q6 = edmmp(A@A.T, edmmp(U,U.T,A@A.T),U)
	q6 = edmmp(A,A.T@edmmp(X,Y,A)@A.T,Z)
	q7 = edmmp(A, edmmp(A.T@X,Y@A,A.T),Z)
	q8 = edmmp(A, edmmp(A.T, edmmp(X,Y,A),A.T),Z)
	T3 = q1-q2-q3+q4-q5+q6+q7-q8


	print(f"Checks:")
	print(f"T1: {np.linalg.norm(T1_real - T1)}")
	print(f"T2: {np.linalg.norm(T2_real - T2)}")
	print(f"T3: {np.linalg.norm(T3_real - T3)}")
	print(f"T3_vykl: {np.linalg.norm(T3_real - T3_vykl)}")
	res = T1-c*T2+c**2*T3
	return res

def F_conj_F_XYZ_optimized(A, c, X, Y, Z):

	T1_real = ( off(X@Y) - c*off(A.T@off(X@Y)@A) )@Z
	T2_real = ( off(A@off(X@Y)@A.T) )@Z
	T3_real = ( off(A@off(A.T@off(X@Y)@A)@A.T) )@Z
	T3_vykl = ( A@A.T@X@Y@A@A.T  # ok
	- A@A.T@diag(X@Y)@A@A.T
	- A@diag(A.T@X@Y@A)@A.T
	+ A@diag(A.T@diag(X@Y)@A)@A.T
	- diag(A@A.T@X@Y@A@A.T)
	+ diag(A@A.T@diag(X@Y)@A@A.T)
	+ diag(A@diag(A.T@X@Y@A)@A.T)
	- diag(A@diag(A.T@diag(X@Y)@A)@A.T) )@Z
	
	ATX = A.T@X
	ATZ = A.T@Z
	YA = Y@A
	YAT = Y@A.T
	AX = A@X
	
	T1 =( X@(Y@Z) 
	- edmmp(X,Y,Z) 
	- c*(ATX)@((YA)@Z) 
	+ c*A.T@edmmp(X,Y,A)@Z 
	+ c*edmmp((ATX),(YA),Z) 
	- c*edmmp(A.T, edmmp(X,Y,A),Z) )

	T2 = ( (AX)@(YAT)@Z 
	- A@edmmp(X,Y,A.T)@Z
	- edmmp((AX),(YAT),Z) 
	+ edmmp(A, edmmp(X, Y, A.T),Z) )
	
	q1 = A@(ATX)@(YA)@(ATZ)
	q2 = A@(A.T@(edmmp(X,Y,A)@(ATZ)))
	q3 = A@edmmp(ATX,YA, ATZ)
	q4 = A@edmmp(A.T, edmmp(X,Y,A),ATZ)
	q5 = edmmp(A@(ATX),(YA)@A.T,Z)
	q6 = edmmp(A,A.T@edmmp(X,Y,A)@A.T,Z)
	q7 = edmmp(A, edmmp(ATX,YA,A.T),Z)
	q8 = edmmp(A, edmmp(A.T, edmmp(X,Y,A),A.T),Z)
	T3 = q1-q2-q3+q4-q5+q6+q7-q8


	print(f"Checks:")
	print(f"T1: {np.linalg.norm(T1_real - T1)}")
	print(f"T2: {np.linalg.norm(T2_real - T2)}")
	print(f"T3: {np.linalg.norm(T3_real - T3)}")
	print(f"T3_vykl: {np.linalg.norm(T3_real - T3_vykl)}")
	res = T1-c*T2+c**2*T3
	return res

def L(A, c, X, Y, Z):

	T1_real = ( off(X@Y) - c*off(A.T@off(X@Y)@A) )@Z
	T2_real = ( off(A@off(X@Y)@A.T) )@Z
	T3_real = ( off(A@off(A.T@off(X@Y)@A)@A.T) )@Z
	T3_vykl = ( A@A.T@X@Y@A@A.T  # ok
	- A@A.T@diag(X@Y)@A@A.T
	- A@diag(A.T@X@Y@A)@A.T
	+ A@diag(A.T@diag(X@Y)@A)@A.T
	- diag(A@A.T@X@Y@A@A.T)
	+ diag(A@A.T@diag(X@Y)@A@A.T)
	+ diag(A@diag(A.T@X@Y@A)@A.T)
	- diag(A@diag(A.T@diag(X@Y)@A)@A.T) )@Z
	
	ATX = A.T@X
	ATZ = A.T@Z
	YA = Y@A
	YAT = Y@A.T
	AX = A@X
	
	T1 =( X@(Y@Z) 
	- edmmp(X,Y,Z) 
	- c*(ATX)@((YA)@Z) 
	+ c*A.T@edmmp(X,Y,A)@Z 
	+ c*edmmp((ATX),(YA),Z) 
	- c*edmmp(A.T, edmmp(X,Y,A),Z) )

	T2 = ( (AX)@(YAT)@Z 
	- A@edmmp(X,Y,A.T)@Z
	- edmmp((AX),(YAT),Z) 
	+ edmmp(A, edmmp(X, Y, A.T),Z) )
	
	q1 = A@(ATX)@(YA)@(ATZ)
	q2 = A@(A.T@(edmmp(X,Y,A)@(ATZ)))
	q3 = A@edmmp(ATX,YA, ATZ)
	q4 = A@edmmp(A.T, edmmp(X,Y,A),ATZ)
	q5 = edmmp(A@(ATX),(YA)@A.T,Z)
	q6 = edmmp(A,A.T@edmmp(X,Y,A)@A.T,Z)
	q7 = edmmp(A, edmmp(ATX,YA,A.T),Z)
	q8 = edmmp(A, edmmp(A.T, edmmp(X,Y,A),A.T),Z)
	T3 = q1-q2-q3+q4-q5+q6+q7-q8


	print(f"Checks:")
	print(f"T1: {np.linalg.norm(T1_real - T1)}")
	print(f"T2: {np.linalg.norm(T2_real - T2)}")
	print(f"T3: {np.linalg.norm(T3_real - T3)}")
	print(f"T3_vykl: {np.linalg.norm(T3_real - T3_vykl)}")
	res = T1-c*T2+c**2*T3
	return res

def off(X):
	return X - np.diag(np.diag(X))

def diag(X):
	return np.diag(np.diag(X))

def F(A,c,X):
	return off(X) - c*off(A.T@off(X)@A)

def F_conj(A,c,X):
	return off(X) - c*off(A@off(X)@A.T)

def F_UUTU_explicit(A,c,U):
	return F_conj(A, c, F(A, c, U@U.T))@U

def F_XYZ_explicit(A,c,X,Y,Z):
	return F_conj(A, c, F(A, c, X@Y))@Z

def razloj(A, c, U):
	return off(U@U.T)-c*off(A.T@off(U@U.T)@A) - c*off(A@off(U@U.T)@A.T) + c**2*off(A@off(A.T@off(U@U.T)@A)@A.T)

def compare_1 (A, c, U):
	optimized = F_conj_F_UUTU(A,c,U)
	explicit = F_UUTU_explicit(A,c,U)
	print(f"UU.T case || explicit - optimized || = {np.linalg.norm(explicit-optimized)}")
def compare_2 (A, c, X,Y,Z):
	optimized = F_conj_F_XYZ_optimized(A, c, X, Y, Z)
	explicit = F_XYZ_explicit(A,c,X, Y, Z)
	print(f"XYZ case || explicit - optimized || = {np.linalg.norm(explicit-optimized)}")

def compare_3 (A, c, U):
	B = c*off(A.T@A)
	n = B.shape[0]
	#grad_opti = 4.*( L(A,c, U,U.T,U) - L(A,c, np.eye(n), B, U) )
	#grad_expl = 4.*F_conj(A, c, F(A, c, U@U.T)-B)@U
	term2 = F_conj(A, c, B)@U
	grad_opti = 4.*( L(A,c, U,U.T,U) ) - 4.*term2
	grad_expl = 4.*F_conj(A, c, F(A, c, U@U.T)-B)@U
	
	print(f"gradient case || explicit - optimized || = {np.linalg.norm(grad_expl-grad_opti)}")
	
if __name__ == "__main__":
	n = 303
	r = 200
	A = metro.ObtainMatrix()
	#A = scsp.csr_matrix(metro.ObtainMatrix())
	c=0.8
	U = np.random.randn(n,r)
	X = np.random.randn(n,r)
	Y = np.random.randn(r,n)
	Z = np.random.randn(n,r)
	print("Check edmmp:")
	print(f"|| diag(XY)@Z - edmmp(X,Y,Z) || = {np.linalg.norm( np.diag(np.diag(X@Y))@Z - edmmp(X,Y,Z) )}")
	print("Check razlojenie:")
	print(f"|| explicit - razloj || = {np.linalg.norm(F_UUTU_explicit(A, c, U) - razloj(A, c, U)@U )}")
	#print(f"Info about A@A.T : nnz(A) = nnz(A.T) = {A.nnz} nnz(A@A.T) = {(A@A.T).nnz}")
	
	compare_1(A, c, U)
	compare_2(A, c, X, Y, Z)
	compare_3(A, c, U)
	
