import numpy as np
import scipy.sparse as scsp
import time
import solvers as slv

class simrank_ops:
	def __init__(self, A, c, r):
		self.c = c
		self.A = A
		self.n = A.shape[1]
		self.ATA = A.T@A
		self.B = c*self.off(self.ATA)
		self.r = r
	def off(self, X):
		Xcopy = X.copy()
		if scsp.issparse(Xcopy):
			Xcopy.setdiag(0.)
		else:
			np.fill_diagonal(Xcopy, 0.)
		return Xcopy
	def mat_inner(self, A, B):
		return np.trace(B.T@A)
	def f(self, X): #f = ||Phi(X)-B||_F^2
		return np.linalg.norm((self.Phi(X)-self.B), ord = 'fro')**2
	def Phi(self, X):
		return self.off(X) - self.c*self.off(self.A.T@self.off(X)@self.A)
	def Phi_conj(self, X):
		return self.off(X) - self.c*self.off(self.A@self.off(X)@self.A.T)
	def gradf(self, U): #grad(f)
		return 4.*self.Phi_conj(self.Phi(U@U.T)-self.B)@U
	def dgradf(self, U, dX): #differential of grad(f) at U from arg differential dX ; d(grad(f)(U)[dX] = (grad(f))'(U)[dX] = J(U)[dX]
		return 4.*( self.Phi_conj(self.Phi(dX@U.T+U@(dX).T))@U + self.Phi_conj(self.Phi(U@U.T)-self.B)@dX )
	def dgradf_vectorized(self, U, dX): #(vectorized for scipy iterative solvers) differential of grad(f) 
		return (4.*( self.Phi_conj(self.Phi(dX.reshape((self.n,self.r), order = 'F')@U.T+U@(dX.reshape((self.n,self.r), order = 'F')).T))@U + self.Phi_conj(self.Phi(U@U.T)-self.B)@dX.reshape((self.n,self.r), order = 'F') ) ).reshape((self.n*self.r,1), order = 'F')

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


class simrank_ops_optimized:
	def __init__(self, A, c, r):
		self.c = c
		self.A = A
		self.AT = (A.T).tocsr() #csr transpose = csc, but we need csr matrices.
		self.n = A.shape[1]
		self.ATA = A.T@A
		self.B = c*self.off(self.ATA)
		self.r = r
		self.termB = self.Phi_conj(self.B)
		self.dmmp = diagmatmatprod()
	def off(self, X):
		Xcopy = X.copy()
		if scsp.issparse(Xcopy):
			Xcopy.setdiag(0.)
		else:
			np.fill_diagonal(Xcopy, 0.)
		return Xcopy
	def mat_inner(self, A, B):
		return np.trace(B.T@A)
	def ffmp(self, X, Y, Z):  
		c = self.c
		dmmp = self.dmmp
		A = self.A
		AT = self.AT
		ATX = self.A.T@X
		AATX = A@(ATX)
		ATZ = self.A.T@Z
		YA = Y@self.A
		YAT = Y@self.A.T
		AX = self.A@X

		DDSXYA = dmmp.DDS(X,Y,A)
		
		T1 =( X@(Y@Z) 
		- dmmp.DDD(X,Y,Z)
		- c*( (ATX)@((YA)@Z)  
		- A.T@DDSXYA@Z
		- dmmp.DDD((ATX),(YA),Z)
		+ dmmp.SSD(AT,DDSXYA,Z) ) )

		T2 = ( (AX)@((YAT)@Z)
		- A@(dmmp.DDS(X,Y,AT)@Z)
		- dmmp.DDD((AX),(YAT),Z)
		+ dmmp.SSD(A, dmmp.DDS(X, Y, AT),Z) )
		
		T3 = ( AATX@((YA)@(ATZ))
		- A@(A.T@(DDSXYA@(ATZ)))
		- A@dmmp.DDD(ATX,YA, ATZ)
		+ A@dmmp.SSD(AT, DDSXYA,ATZ)
		- dmmp.DDD(AATX,(YA)@AT,Z)
		+ dmmp.SSD(A,AT@(DDSXYA@AT) + dmmp.DDS(ATX,YA,AT) - dmmp.SSS(AT, DDSXYA,AT),Z) )
		
		res = T1-c*T2+c**2*T3
		return res
	def f(self, X): #f = ||Phi(X)-B||_F^2
		return np.linalg.norm((self.Phi(X)-self.B), ord = 'fro')**2
	def Phi(self, X):
		return self.off(X) - self.c*self.off(self.A.T@self.off(X)@self.A)
	def Phi_conj(self, X):
		return self.off(X) - self.c*self.off(self.A@self.off(X)@self.A.T)
	def gradf(self, U): #grad(f)
		res = 4.*( self.ffmp(U,U.T,U) - self.termB@U )
		return res
	def dgradf_vectorized(self, U, dX):
		#differential of grad(f) at U from arg differential dX ; d(grad(f)(U)[dX] = (grad(f))'(U)[dX] = J(U)[dX] 
		#(vectorized for scipy iterative solvers) differential of grad(f) 
		res =  4.*( self.ffmp(dX.reshape((self.n,self.r), order = 'F'),U.T,U) 
		+ self.ffmp(U,dX.reshape((self.n,self.r), order = 'F').T,U) 
		+ self.ffmp(U,U.T,dX.reshape((self.n,self.r), order = 'F')) 
		- (self.termB@dX.reshape((self.n,self.r), order = 'F')) ).reshape((self.n*self.r,1), order = 'F')
		return res
		
class simrank_ops_another(simrank_ops): #with operators to solve with S = U@U.T
	def __init__(self, A, c, r):
		super().__init__(A, c, r)
		self.B = np.eye(self.n)
	def f(self, X): #f = ||Phi(X)-B||_F^2
		return np.linalg.norm((self.Phi(X)-self.B), ord = 'fro')**2
	def Phi(self, X):
		return X - self.c*self.off(self.A.T@X@self.A)
	def Phi_conj(self, X):
		return X - self.c*self.A@self.off(X)@self.A.T
#NOTE: atol --> to 28.11.2025. This COULD lead to different behaviour!
class GMRES_scipy:
	def __init__(self, matvec, b, x_0, m_Krylov, maxiter, eps=1e-10, printout=True):
		self.LinOp = scsp.linalg.LinearOperator((b.shape[0],b.shape[0]), matvec = matvec)
		self.b = b
		self.x_0 = x_0
		self.m = m_Krylov
		self.maxiter = maxiter
		self.eps = eps
		self.printout = printout
		self.cb = lambda res : print(f"GMRES relative residual = {res}")  if printout else None
	def __call__(self):
		u, _ = scsp.linalg.gmres(self.LinOp, self.b, x0=self.x_0, tol=self.eps, restart=self.m, maxiter=self.maxiter, M=None, callback=self.cb, callback_type='pr_norm') 
		return u

class CG_scipy:
	def __init__(self, matvec, b, x_0, m_Krylov, maxiter, eps=1e-10, printout=True):
		self.LinOp = scsp.linalg.LinearOperator((b.shape[0],b.shape[0]), matvec = matvec)
		self.b = b
		self.x_0 = x_0
		self.m = m_Krylov
		self.maxiter = maxiter
		self.eps = eps
		self.printout = printout
		self.cb = lambda res : print(f"GMRES relative residual = {res}")  if printout else None
	def __call__(self):
		u, _ = scsp.linalg.cg(self.LinOp, self.b, x0=self.x_0, tol=self.eps, maxiter=self.m, M=None, callback=self.cb) 
		return u

class GMRES_custom:
	def __init__(self, matvec, b, x_0, m_Krylov, maxiter, eps=1e-10, printout=True):
		self.matvec = matvec
		self.b = b
		self.x_0 = x_0
		self.m = m_Krylov
		self.maxiter = maxiter
		self.eps = eps
		self.printout = printout
		self.cb = lambda res : print(f"GMRES relative residual = {res}")
	def __call__(self):
		u, _ = slv.GMRES_m(LinOp=self.matvec, m_Krylov=self.m, x_0=self.x_0, b=self.b, k_max=self.maxiter, eps=self.eps, printout=self.printout)
		return u

#---Newton solver---

def inverse_matvec(LinOp, x, m_Krylov, restarts, solver, eps=1e-10, printout=True):
	if printout : print(f"Starting inverse matvec with : m_Krylov = {m_Krylov} ; restarts = {restarts}")
	st = time.time()
	n, r = x.shape
	solver = solver(LinOp, b=x.reshape((n*r,1), order = 'F'),  x_0=np.zeros((n*r,1)), m_Krylov=m_Krylov, maxiter=restarts, eps=eps, printout=printout)
	u = solver()
	if printout : print(f"Finished inverse matvec for {time.time()-st} s.")
	return u

def Newton(A, c, r, maxiter, gmres_restarts, m_Krylov, solver, stagstop = 1e-5, eps_f = 1e-15, eps_shift = 1e-15, optimize = False, compute_functional = True, printout = True):
	
	iterdata_vals_types = ["iteration", "U_shift"]
	if compute_functional : 
		iterdata_vals_types.append("f(U)_function")
	iterdata = slv.iterations_data(iterdata_vals_types)
	
	save_shift = lambda val : iterdata.saveval(val, "U_shift", printout)
	save_iteration = lambda iteration : iterdata.saveval(iteration, "iteration", printout)
	if compute_functional : 
		save_funcval = lambda val : iterdata.saveval(val, "f(U)_function", printout)
	
	n, _ = A.shape
	np.random.seed(42)
	U = np.random.randn(n,r)
	if optimize:
		sops = simrank_ops_optimized(A, c, r)
	else:
		sops = simrank_ops(A, c, r)
	if compute_functional : f_val_prev = sops.f(U@U.T)
	st = time.time()
	for k in range(maxiter):
		save_iteration(k)
		dgradf_vec = lambda X : sops.dgradf_vectorized(U, X)
		st = time.time()
		DeltaU = inverse_matvec(dgradf_vec, sops.gradf(U), m_Krylov, gmres_restarts, solver, printout=False).reshape((n,r), order = 'F')
		DeltaU_norm = np.linalg.norm(DeltaU, ord = "fro")
		save_shift(DeltaU_norm)
		U -= DeltaU
		if printout : print(f"Newton step time = {time.time()-st}")
		if DeltaU_norm < eps_shift :
			if printout : print(f"Stopped by Delta(U) value < {eps_shift})")
			break 
		if compute_functional:
			f_val = sops.f(U@U.T)
			save_funcval(f_val)
			if (np.abs(f_val - f_val_prev)) < stagstop:
				if printout : print(f"Stopped by f(U) stagnation (| f - f_prev | < {stagstop})")
				break
			if f_val < eps_f:
				if printout : print(f"Stopped by f(U) value < {eps_f})")
				break
			f_val_prev = f_val
	iterdata.elapsed = time.time()-st
	return U, iterdata
#---

#--- Gradient method with step splitting---

def getalpha(f, gradf, inner, x, tau, maxiter=100, printout = True): #step splitting
	if printout : print(f"Starting step splitting...")
	alpha = 1.
	kfin = 0
	relax = lambda alpha : f( (x + alpha*gradf(x))@(x + alpha*gradf(x)).T) <= ( f(x@x.T) - tau*alpha*inner(gradf(x), gradf(x)) )
	for k in range(maxiter):
		#print(f"alpha = {alpha}")
		if relax(alpha):
			kfin = k
			break
		alpha = alpha*0.5
	if printout : print(f"Finished with alpha = {alpha} at iter {kfin}")
	return alpha

def Gradmethod(A, c, r, maxiter, printout = True):
	iterdata_vals_types = ["iteration", "f(U)_function"]
	iterdata = slv.iterations_data(iterdata_vals_types)
	save_funcval = lambda val : iterdata.saveval(val, "f(U)_function", printout)
	save_iteration = lambda iteration : iterdata.saveval(iteration, "iteration", printout)
	n, _ = A.shape
	np.random.seed(42)
	U = np.random.randn(n,r)
	sops = simrank_ops(A,c,r)
	st = time.time()
	for k in range(maxiter):
		save_iteration(k)
		save_funcval(sops.f(U@U.T))
		alpha = getalpha(sops.f, sops.gradf, sops.mat_inner, U, tau=0.5)
		U = U - alpha*sops.gradf(U)
	iterdata.elapsed = time.time()-st
	return np.eye(n) + sops.off(U@U.T), iterdata
#---
