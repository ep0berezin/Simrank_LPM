import numpy as np
import solvers as slv
import time
import scipy.sparse as scsp
#import rsvd_solver as RSVDTEST

def off(X):
	Xcopy = X.copy()
	if scsp.issparse(Xcopy):
		Xcopy.setdiag(0.)
	else:
		np.fill_diagonal(Xcopy, 0.)
	return Xcopy

def RSVDIters(A, c, r, maxiter, eps, p=8):
	np.random.seed(42)
	n = A.shape[0]
	iterdata = slv.iterations_data()
	dmmp = slv.diagmatmatprod()
	mdmp = slv.matdiagmatprod()
	M_prev = c*off(A.T@A).toarray()
	Usvd, s, Vsvd = np.linalg.svd(M_prev, full_matrices = False)
	U = Usvd*np.sqrt(s)
	st = time.time()
	for k in range(maxiter):
		Omega = np.random.randn(n,r+p)
		ATU = A.T@U
		UTA = ATU.T
		MOmega = c*ATU@(UTA@Omega) - c*dmmp.DDD(ATU, UTA, Omega ) + c*A.T@(A@Omega) - c*dmmp.SSD(A.T, A, Omega)
		Q, R = np.linalg.qr(MOmega)
		QTM = Q.T@M_prev
		M = Q@QTM
		Usvd, s, Vsvd = np.linalg.svd(QTM, full_matrices=False)
		U = ((Q@Usvd)[:,:r]*np.sqrt(s[:r]))
		relres = np.linalg.norm((M - M_prev), ord='fro')/np.linalg.norm(M_prev, ord='fro')
		iterdata(relres)
		M_prev = M
	elapsed = time.time()-st
	solutiondata = [iterdata.iterations, iterdata.residuals, elapsed]
	
	return U, solutiondata

def REigenIters(A, c, r, maxiter, eps, p=8):
	n = A.shape[0]
	iterdata = slv.iterations_data()
	dmmp = slv.diagmatmatprod()
	mdmp = slv.matdiagmatprod()
	U = np.zeros((n,r))
	U_prev = U
	D = np.zeros((r,r))
	st = time.time()
	for k in range(maxiter):
		#S - I = UDU.T
		#UDU.T = cA.TUDU.TA - c diag (A.TUDU.TA) + cATA - c diag(ATA)
		Omega = np.random.randn(n,r+p)
		ATUD = A.T@(U@D)
		UTA = U.T@A
		
		FOmega = c*ATUD@(UTA@Omega) - c*dmmp.DDD(ATUD, UTA, Omega ) + c*A.T@(A@Omega) - c*dmmp.SSD(A.T, A, Omega)
		Q, R = np.linalg.qr(FOmega)
		QTF = c*(Q.T@ATUD)@UTA - c*mdmp.DDD(Q.T, ATUD, UTA) + c*(Q.T@A.T)@A - c*mdmp.DSS(Q.T, A.T, A) # (r + p) x n
		eigvals, V = np.linalg.eig(QTF@Q) 
		D = np.diag(eigvals[:r])
		U = (Q@V)[:,:r]
		relres = np.linalg.norm((U - U_prev ), ord='fro')/np.linalg.norm(U_prev, ord='fro')
		iterdata(relres)
		U_prev = U
	elapsed = time.time()-st
	solutiondata = [iterdata.iterations, iterdata.residuals, elapsed]
	
	return U, solutiondata

def RSVDIters_OLDVER(A, c, r, maxiter, eps, p=8):
	n = A.shape[0]
	iterdata = slv.iterations_data()
	dmmp = slv.diagmatmatprod()
	mdmp = slv.matdiagmatprod()
	U = np.zeros((n,r))
	st = time.time()
	for k in range(maxiter):
		Omega = np.random.randn(n,r+p)
		ATU = A.T@U
		UTA = ATU.T
		FOmega = c*ATU@(UTA@Omega) - c*dmmp.DDD(ATU, UTA, Omega ) + c*A.T@(A@Omega) - c*dmmp.SSD(A.T, A, Omega)
		Q, R = np.linalg.qr(FOmega)
		QTF = c*(Q.T@ATU)@UTA - c*mdmp.DDD(Q.T, ATU, UTA) + c*(Q.T@A.T)@A - c*mdmp.DSS(Q.T, A.T, A)
		Usvd, s, Vsvd = np.linalg.svd(QTF)
		U = ((Q@Usvd)*np.sqrt(s))[:,:r]
		relres = np.linalg.norm((U - U_prev ), ord='fro')/np.linalg.norm(U_prev, ord='fro')
		iterdata(residual)
		U_prev = U
	elapsed = time.time()-st
	solutiondata = [iterdata.iterations, iterdata.residuals, elapsed]
	
	return U, solutiondata
