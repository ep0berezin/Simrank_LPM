import numpy as np
import matplotlib.pyplot as plt 
from scipy.sparse import csr_matrix
import scipy.sparse as scsp
import sys
import time

class iterations_data:
	def __init__(self, valtypes):
		self.values = { key : [] for key in valtypes }
		self.elapsed = 0.
	def saveval(self, val, valtype, printout = True):
		if printout : print(f"{valtype}: {val}")
		self.values[valtype].append(val)
		return val

#---Fixed Point---

def FixedPointIter(LinOp, tau, x_0, b, k_max, printout, eps = 1e-13):
	N = x_0.shape[0] #N = n^2
	s = x_0
	iterdata_vals_types = ["iteration", "relative_residual"]
	iterdata = iterations_data(iterdata_vals_types)
	save_residual = lambda val : iterdata.saveval(val, "relative_residual", printout)
	save_iteration = lambda iteration : iterdata.saveval(iteration, "iteration", printout)
	st = time.time()
	for k in range(k_max):
		s_prev = s
		s = (b-LinOp(s))*tau+s
		r_norm2 = np.linalg.norm(s-s_prev, ord = 2)
		relres = r_norm2/np.linalg.norm(b, ord = 2)
		save_iteration(k)
		save_residual(relres)
		if (relres  < eps):
			break
	et = time.time()
	iterdata.elapsed = et - st
	if printout: print(f"Average iteration time: {iterdata.elapsed/iterdata.values['iteration'][-1]} s")
	if printout: print(f"Elapsed: {iterdata.elapsed} s")
	return s, iterdata
#---


#---GMRES(m)---

def LSq(beta, H):
	m = H.shape[1]
	e1 = np.zeros((m+1,1))
	e1[0,0] = 1.0 #generating e1 vector
	b = e1*beta
	#y = np.linalg.inv(H.T@H)@H.T@b
	y = np.linalg.pinv(H)@b
	return y

def Arnoldi(V_list, h_list, m_start, m_Krylov, LinOp, eps_zero = 1e-15, printout = False):
	for j in range(m_start, m_Krylov):
		#print(f"Building Arnoldi: V[:,{j}]")
		st_1 = time.time()
		v_j = V_list[j]
		w_j = LinOp(v_j).reshape(-1, order='F') #
		if printout: print("Evaluate Av_j time:", time.time()-st_1)
		Av_j_norm2 = np.linalg.norm(w_j, ord = 2)
		st_2 = time.time()
		for i in range(j+1):
			v_i = V_list[i]
			h_ij = v_i@w_j  
			h_list[i][j] = h_ij
			w_j -= h_ij*v_i
		if printout: print("MGS for v_{j+1} time:", time.time()-st_2)
		w_j_norm2 = np.linalg.norm(w_j, ord = 2)
		h_list[j+1][j] = w_j_norm2
		if (w_j_norm2 <= eps_zero):
			return j
		V_list[j+1] = w_j*(1/w_j_norm2)
	return m_Krylov

def GMRES_m(LinOp, m_Krylov, x_0, b, k_max, eps = 1e-13, printout = False):
	N = x_0.shape[0] #N = n^2
	r = b - LinOp(x_0)
	r_norm2 = np.linalg.norm(r, ord = 2)
	relres = r_norm2/np.linalg.norm(b, ord = 2)
	iterdata_vals_types = ["iteration", "relative_residual"]
	iterdata = iterations_data(iterdata_vals_types)
	save_residual = lambda val : iterdata.saveval(val, "relative_residual", printout)
	save_iteration = lambda iteration : iterdata.saveval(iteration, "iteration", printout)
	save_iteration(0)
	save_residual(relres)
	st = time.time()
	x = x_0
	break_outer = False
	for k in range(k_max):
		if (break_outer):
			break
		st_restart = time.time()
		r = b - LinOp(x)
		r_norm2 = np.linalg.norm(r, ord = 2)
		beta = r_norm2
		V_list = [np.zeros(N)] #Stores columns of V matrix
		V_list[0] = r.reshape(-1, order='F')/beta
		H_list = [np.zeros(m_Krylov)] #Stores rows of Hessenberg matrix
		for m in range(1,m_Krylov+1):
			st_iter = time.time()
			V_list.append(np.zeros(N)) #Reserving space for vector (column of V) v_{j+1}
			H_list.append(np.zeros(m_Krylov)) #Reserving space for row of H h_{j+1}
			st_arnoldi = time.time()
			m_res = Arnoldi(V_list, H_list, (m-1), m,  LinOp)
			if printout: print("Arnoldi time:", time.time()-st_arnoldi)
			V = (np.array(V_list[:m_res])).T #Slicing V_list[:m] because v_{m+1} is not needed for projection step.
			H = (np.array(H_list))[:,:m_res] #Slicing because everything right to m'th column is placeholding zeros.
			st_lsq = time.time()
			y = LSq(beta, H)
			if printout: print("LSq time:", time.time()-st_lsq)
			st_proj = time.time()
			x = x_0 + V@y
			if printout: print("Projection step time:", time.time()-st_proj)
			r_norm2_inner = np.linalg.norm(b-LinOp(x), ord = 2)
			relres_inner = r_norm2_inner/np.linalg.norm(b, ord = 2)
			save_iteration(k*m_Krylov + m)
			save_residual(relres_inner) #rel residual
			if (relres_inner < eps):
				break_outer = True
				break
			et_iter = time.time()
			if printout: print(f"m = {m}; Absolute residual = :", r_norm2_inner, f"; Iteration time: {et_iter-st_iter} s")
		x_0 = x
		et_restart = time.time()
		if printout: print("Restart time:", et_restart - st_restart)
	et = time.time()
	iterdata.elapsed = et - st
	if printout: print(f"Average iteration time: {iterdata.elapsed/iterdata.values['iteration'][-1]} s")
	if printout: print(f"GMRES(m) time: {iterdata.elapsed} s")
	return x, iterdata
#---

#---GMRES SciPy ver---

def GMRES_scipy_callback(relres, iterdata, printout):
	iteration = len(iterdata.values["relative_residual"])
	iterdata.saveval(iteration, "iteration", printout)
	iterdata.saveval(relres, "relative_residual", printout) 
	return relres

def GMRES_scipy(LinOp, m_Krylov, x_0, b, k_max, eps = 1e-13, printout = False):
	N = x_0.shape[0] #N=n^2
	iterdata_vals_types = ["iteration", "relative_residual"]
	iterdata = iterations_data(iterdata_vals_types)
	save_residual = lambda val : GMRES_scipy_callback(val, iterdata, printout)
	r = b - LinOp(x_0) #initial residual
	GMRES_scipy_callback(np.linalg.norm(r, ord = 2)/np.linalg.norm(b, ord = 2), iterdata, printout)
	G = scsp.linalg.LinearOperator((N,N), matvec = LinOp)
	st = time.time()
	s, data = scsp.linalg.gmres(G, b, x0=x_0, atol=eps, restart=m_Krylov, maxiter=None, M=None, callback=save_residual, callback_type='legacy')
	et = time.time()
	iterdata.elapsed = et - st
	if printout: print(f"Average iteration time: {iterdata.elapsed/iterdata.values['iteration'][-1]} s")
	if printout: print("Elapsed:", iterdata.elapsed)
	if printout: print("Solution:", s)
	return s, iterdata
#---

#--- CGNR ---

def CGNR(LinOp, LinOp_conj, x_0, b, inner = lambda u,v : v.T@u , maxiter=10000, eps = 1e-13, printout = False):
	x = x_0
	r = b - LinOp(x)
	z = LinOp_conj(r)
	p = z
	for k in range(maxiter):
		w = LinOp(p)
		alpha = inner(z,z)/inner(w,w)
		x = x + alpha*p
		r = r - alpha*w
		r_norm2 = np.linalg.norm( r, ord = 'fro')
		z_kp1 = LinOp_conj(r)
		if printout: print(f"iteration {k} ; ||r||_2 = {r_norm2}") 
		if r_norm2 < eps:
			break
		beta = inner(z_kp1,z_kp1)/inner(z,z)
		p = z_kp1 + beta*p
		z = z_kp1
	return x
#---

#--- Alternating optimization solver (proposed by German Z. Alekhin, MSU CMC)---

class simrank_ops:
	def __init__(self, A, c):
		self.c = c
		self.A = A
		self.AT = A.T.tocsr()
		self.n = A.shape[1]
		self.ATA = A.T@A
		self.B = c*self.off(self.ATA)
	def off(self, X):
		Xcopy = X.copy()
		if scsp.issparse(Xcopy):
			Xcopy.setdiag(0.)
		else:
			np.fill_diagonal(Xcopy, 0.)
		return Xcopy
	def mat_inner(self, A,B):
		return np.trace(B.T@A)
	def AltMin_K_operator(self, Y, Ypinv, X, mdmp):
		ATY = self.A.T@Y # n x r
		XTA = X.T@self.A #r x n
		result = self.c * ( Ypinv @ ATY ) @ XTA - self.c * mdmp.DDD(Ypinv, ATY, XTA)
		return result

class diagmatmatprod:
	#method name corresponds with arguments types
	#D (dense) - np.ndarray
	#S (sparse csr) - scipy csr
	#St (sparse csr.T aka csc) - scipy csc
	def DDD(self, X, Y, Z):
		Z_copy = Z.copy()
		dotp = np.sum(X * Y.T, axis=1)
		return Z_copy * dotp.reshape((X.shape[0],1))
	def DDS(self, X, Y, Z):
		Z_copy = Z.copy()
		dotp = np.sum(X * Y.T, axis=1)
		return Z_copy.multiply(dotp.reshape((X.shape[0],1))).tocsr()
	def SSD(self, X, Y, Z): #works for both csc and csr 
		Z_copy = Z.copy()
		dotp = np.asarray( np.sum(X.multiply(Y.T), axis=1) ) #for some reason np.sum(csr*csr) return np.matrix. --> np.asarray()
		return Z_copy * dotp.reshape((X.shape[0],1))
	def SSS(self, X, Y, Z):
		Z_copy = Z.copy()
		dotp = np.asarray( np.sum(X.multiply(Y.T), axis=1) )
		return Z_copy.multiply(dotp.reshape((X.shape[0],1))).tocsr()
		
class matdiagmatprod:
	#method name corresponds with arguments types
	#D (dense) - np.ndarray
	#S (sparse csr) - scipy csr
	#St (sparse csr.T aka csc) - scipy csc
	def DDD(self, Z, X, Y):
		Z_copy = Z.copy()
		dotp = np.sum(X * Y.T, axis=1)
		return ( Z_copy.T * dotp.reshape((X.shape[0],1)) ).T
	def SDD(self, Z, X, Y):
		Z_copy = Z.copy()
		dotp = np.sum(X * Y.T, axis=1)
		return ( Z_copy.T.multiply(dotp.reshape((X.shape[0],1))) ).T.tocsr()
	def DSS(self, Z, X, Y): #works for both csc and csr 
		Z_copy = Z.copy()
		dotp = np.asarray( np.sum(X.multiply(Y.T), axis=1) ) #for some reason np.sum(csr*csr) return np.matrix. --> np.asarray()
		return ( Z_copy.T * dotp.reshape((X.shape[0],1)) ).T
	def SSS(self, Z, X, Y):
		Z_copy = Z.copy()
		dotp = np.asarray( np.sum(X.multiply(Y.T), axis=1) )
		return ( Z_copy.T.multiply(dotp.reshape((X.shape[0],1))) ).T.tocsr()

def LSFP(U, V, A, c, dir_maxit, printout): #Least Squares Fixed-Point iterations solver without n x n matrices.
	#NOTE: computation of pinv as (A.TA)^-1 A.T is faster but numerically unstable
	sops = simrank_ops(A, c)
	mdmp = matdiagmatprod()
	U_pinv = np.linalg.pinv(U)
	K_U = lambda X : sops.AltMin_K_operator(U, U_pinv, X, mdmp)
	for iter_v in range(dir_maxit): #U fixed
		if printout: print(f"V direction iteration {iter_v}")
		V = ( K_U(V) + U_pinv@sops.B ).T
	V_pinv = np.linalg.pinv(V)
	K_V = lambda X : sops.AltMin_K_operator(V, V_pinv, X, mdmp)
	for iter_u in range(dir_maxit):
		if printout: print(f"U direction iteration {iter_u}")
		U = ( K_V(U) +  V_pinv@sops.B.T ).T
	return U,V

def AltMin(A, c, r, solver, maxiter=100, dir_maxit=1, eps_fro = 1e-15, eps_cheb = 1e-15, compute_full_matrix= False, printout = False): #Main alternating optimiztion function.

	iterdata_vals_types = ["iteration", "AltMin_V_difference_Frobenius", "AltMin_U_difference_Frobenius"]
	if compute_full_matrix : 
		iterdata_vals_types.append("AltMin _U@V.T_difference_Frobenius")
	iterdata = iterations_data(iterdata_vals_types)
	
	save_iteration = lambda iteration : iterdata.saveval(iteration, "iteration", printout)
	save_diffV = lambda val : iterdata.saveval(val, "AltMin_V_difference_Frobenius", printout)
	save_diffU = lambda val : iterdata.saveval(val, "AltMin_U_difference_Frobenius", printout)
	if compute_full_matrix : 
		save_diffM = lambda val : iterdata.saveval(val, "AltMin _U@V.T_difference_Frobenius", printout)
	
	n = A.shape[1]
	np.random.seed(42)
	U = np.random.randn(n,r)
	V = np.random.randn(n,r)
	
	st = time.time()
	
	for k in range(maxiter):
		if printout: print(f"Alternating optimization iteration {k}")
		V_prev = V
		U_prev = U
		U, V = solver(U, V, A, c, dir_maxit, False)
		diffV = V - V_prev
		diffU = U - U_prev
		err_diffV_fro = (np.linalg.norm(diffV, ord = 'fro'))
		err_diffU_fro = (np.linalg.norm(diffU, ord = 'fro'))
		save_iteration(k)
		save_diffV(err_diffV_fro)
		save_diffU(err_diffU_fro)
		if printout : print(f"||V^(k+1) -V^(k)||_F at iter {k} = {err_diffV_fro}")
		if printout : print(f"||U^(k+1) -U^(k)||_F at iter {k} = {err_diffU_fro}")
		if  err_diffV_fro < eps_fro:
			if printout: print("Converged by V shift err Fro")
			break
		if  err_diffU_fro < eps_fro:
			if printout: print("Converged by U shift err Fro")
			break
			
		if compute_full_matrix :
			diffM = U@V.T - U_prev@V_prev.T
			err_diffM_fro = (np.linalg.norm(diffM, ord = 'fro'))
			err_diffM_cheb = np.max(np.abs(diffM))
			save_diffM(err_diffM_fro)
			if printout : print(f"||U^(k+1)@V^(k+1).T - U^(k)@V^(k).T||_C at iter {k} = {err_diffM_cheb}")
			
			if  err_diffM_fro < eps_fro:
				if printout: print("Converged by U@V.T err Fro")
				break
			if err_diffM_cheb < eps_cheb:
				if printout: print("Converged by U@V.T err Cheb")
				break

	iterdata.elapsed = time.time() - st
	print(f"Elapsed {iterdata.elapsed}")
	return U,V, iterdata
