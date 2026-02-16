import numpy as np
import scipy.sparse as scsp
import sys
import time
import matplotlib.pyplot as plt 
import datetime as dt
import pandas as pd
from matplotlib.ticker import FixedLocator, FixedFormatter
import solvers as slv
import opti_solvers as optslv
import pandas as pd
import rsvd_iters as rsvdit
import networkx as nx

dateformat = "%Y-%m-%d-%H-%M-%S" #for log and plots saving

class G_operator:
	def __init__(self, A, c):
		self.A = A
		self.n = self.A.shape[1]
		if (self.n!=self.A.shape[0]):
			print(f"Warning! Non-square adjacency matrix detected when constructing operator.")
		self.c = c
	def __call__(self, u):
		U = u.reshape((self.n, self.n), order = 'F')
		T_1 = self.A.T@U@self.A
		np.fill_diagonal(T_1, 0.0) #A.TUA - diag(A.TUA)
		G = U - self.c*T_1
		G = G.reshape((self.n**2,1), order = 'F')
		return G



def Solve(acc, m_Krylov, rank, k_iter_max, taskname, A, c, solver, optimize): #solvers = list of flags: ['SimpleIter, GMRES, MinRes'] (in any order)
	if (A.shape[0]!=A.shape[1]):
		print("Non-square matrix passed in argument. Stopped.")
		return 1
	print("Adjacency matrix:")
	print(A)
	
	tau=1.
	n = A.shape[0]
	I = np.eye(n)
	I_vec = np.eye(n).reshape((n**2,1), order = 'F')
	
	#epsilons in diag to avoid sparsity changes in A_csr
	if scsp.issparse(A):
		A.setdiag(1e-15)
	else:
		np.fill_diagonal(A, 1e-15)
	A_csr = scsp.csr_matrix(A)
	
	def LinSys_FixedPointIter():
		n = A_csr.shape[0]
		print(f"Starting FixedPointIter with {k_iter_max} iterations limit tau =  {tau} iter parameter ...")
		G = G_operator(A_csr, c)
		ts = time.time()
		s_fpi, solutiondata = slv.FixedPointIter(G, tau, np.zeros((n,n)).reshape((n**2,1), order = 'F'), I_vec, k_iter_max, printout, acc)
		ts = time.time() - ts
		S_fpi = s_fpi.reshape((n,n), order = 'F')
		return S_fpi, solutiondata, ts
	
	def LinSys_GMRES():
		print(f"Starting GMRES with {k_iter_max} iterations limit and {m_Krylov} max Krylov subspace dimensionality ...")
		G = G_operator(A_csr, c)
		ts = time.time()
		s_gmres, solutiondata = slv.GMRES_m(G, m_Krylov, np.zeros((n,n)).reshape((n**2,1), order = 'F'), I_vec, k_iter_max, acc, printout=True)
		ts = time.time() - ts
		S_gmres = s_gmres.reshape((n,n), order = 'F')
		return S_gmres, solutiondata, ts
		
	def LinSys_GMRES_scipy():
		print(f"Starting GMRES from SciPy with {k_iter_max} iterations limit and {m_Krylov} max Krylov subspace dimensionality ...")
		G = G_operator(A_csr, c)
		ts = time.time()
		s_gmres_scipy, solutiondata = slv.GMRES_scipy(G, m_Krylov, np.zeros((n,n)).reshape((n**2,1), order = 'F'), I_vec, k_iter_max, acc, printout=True)
		S_gmres_scipy = s_gmres_scipy.reshape((n,n), order = 'F')
		ts = time.time() - ts
		return S_gmres_scipy, solutiondata, ts
		
	def AltMin_LSFP():
		print(f"Starting Alternating optimization solver with rank={rank}, iterations limit {k_iter_max}  ...")
		ts = time.time()
		U, V, solutiondata = slv.AltMin(A_csr, c, rank, slv.LSFP, k_iter_max, dir_maxit=10, compute_full_matrix=False, printout = True)
		S_altmin = np.eye(n) + U@V.T
		ts = time.time() - ts
		return S_altmin, solutiondata, ts
	
	def QMin_Newton():
		print(f"Starting optimization using Newton solver with rank={rank}, iterations limit {k_iter_max} ...")
		#NOTE: best results are obtained if gmres_restarts=1 and m_Krylov 10...20 used.
		#Increasing m_Krylov (and generally total amount of iterations, i.e. gmres_restarts*m_Krylov) over 20 leads to worse results.
		ts = time.time()
		U, solutiondata = optslv.Newton(A_csr, c, rank, maxiter=k_iter_max, gmres_restarts=1, m_Krylov=m_Krylov, solver=optslv.GMRES_scipy, stagstop=1e-5, optimize=optimize)
		M = U@U.T
		np.fill_diagonal(M, 0.)
		S_qmin_newton = np.eye(n) + M
		ts = time.time() - ts
		return S_qmin_newton, solutiondata, ts
	
	def RSVDIters():
		print(f"Starting RSVD Iters with rank={rank}, iterations limit {k_iter_max} ...")
		ts = time.time()
		p=8
		U_rsvd, solutiondata =  rsvdit.RSVDIters(A_csr, c, rank, k_iter_max, acc)
		M = U_rsvd@U_rsvd.T
		M -= np.diag(np.diag(M))
		ts = time.time() - ts
		return M+I, solutiondata, ts
	def REigenIters():
		print(f"Starting REigen Iters Iters with rank={rank}, iterations limit {k_iter_max} ...")
		ts = time.time()
		p=8
		U_rsvd, solutiondata =  rsvdit.REigenIters(A_csr, c, rank, k_iter_max, acc)
		ts = time.time() - ts
		return U_rsvd@U_rsvd.T + I, solutiondata, ts
	def SimrankNX():
		print(f"Starting SimpleIter NX with {k_iter_max} iterations limit  ...")
		A_r = np.where(A_csr.toarray()>0, 1, 0)
		print("Restored adjacency matrix A_r:")
		print(A_r)
		Graph = nx.from_numpy_array(A_r, create_using=nx.MultiDiGraph())
		#plt.figure()
		#nx.draw(Graph)
		#plt.show()
		Graph.remove_edges_from(nx.selfloop_edges(Graph)) #remove loops
		ts = time.time()
		S_nx = nx.simrank_similarity(Graph, importance_factor=c, max_iterations = k_iter_max, tolerance = acc)
		ts = time.time() - ts
		print("Elapsed NX: ", ts)
		Snx_fin = np.zeros((n,n))
		for i in range(n):
			for j in range(n):
				Snx_fin[i,j] = S_nx[i][j]
		return Snx_fin, None, ts

	solvers_dict = {
	"FixedPointIter" : LinSys_FixedPointIter,
	"GMRES" : LinSys_GMRES,
	"GMRES_scipy" : LinSys_GMRES_scipy,
	"AltMin_LSFP" : AltMin_LSFP,
	"QMin_Newton" : QMin_Newton,
	"RSVDIters" : RSVDIters, 
	"REigenIters" : REigenIters,
	"SimrankNX" : SimrankNX
	}
	
	S, solutiondata, ts = solvers_dict[solver]()
	writelog(taskname, rank, c, solver, solutiondata)
	return S

def writelog(taskname, rank, c, solver, solutiondata):
	filename = f"results/log/log_{taskname}_rank_{rank}_c_{c}_solver_{solver}_{dt.datetime.now().strftime(dateformat)}.csv" 
	df = pd.DataFrame(solutiondata.values)
	df.to_csv(filename, index=False)
	print(f"Saved log to csv.")


