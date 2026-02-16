import numpy as np
import simrank_main as sm
import matplotlib.pyplot as plt

class results:
	def __init__(self, taskname, ranks, solver):
		self.ranks = ranks
		self.taskname = taskname
		self.err_Frob_frank = []
		self.err_Cheb_frank = []
		self.solver = solver
		self.solution_frank = []
	def getresults(self):
		for rank in ranks:
			args = {"acc":1e-5, "m_Krylov": 15, "rank": rank, "k_iter_max": 100, "taskname": self.taskname, "c": 0.8, "solvers":[self.solver, "Compare"],  "optimize" : False, "showfig": False}
			Sdict, err_Frob, err_Cheb = sm.launch(args)
			self.err_Frob_frank.append(err_Frob), self.err_Cheb_frank.append(err_Cheb), self.solution_frank.append(Sdict[list(Sdict.keys())[0]])

'''
def SVDtest(taskname, ranks):

	Newton_results = results(taskname, ranks, "Optimization_Newton")
	Newton_results.getresults()
	SVD_results = results(taskname, ranks, "SVDCompare")
	SVD_results.getresults()
	
	plt.figure()
	plt.grid()
	plt.plot(ranks, Newton_results.err_Cheb_frank, marker = '.')
	plt.plot(ranks, SVD_results.err_Cheb_frank, marker = '.')
	plt.legend(["Метод Ньютона","SVD"])
	plt.xlabel(r"Ранг аппроксимации $r$")
	plt.ylabel(r"Ошибка $||S-S_{approx}||_C$")
	#plt.title("Cheb from rank")
	
	plt.figure()
	plt.grid()
	plt.plot(ranks, Newton_results.err_Frob_frank, marker = '.')
	plt.plot(ranks, SVD_results.err_Frob_frank, marker = '.')
	plt.legend(["Метод Ньютона","SVD"])
	plt.xlabel(r"Ранг аппроксимации $r$")
	plt.ylabel(r"Ошибка, $||S-S_{approx}||_F$")
	#plt.title("Frob from rank")

def normtest(taskname, ranks):
	#ALS_results = results(taskname, ranks, "AltOpt")
	#ALS_results.getresults()
	#Newton_results = results(taskname, ranks, "Optimization_Newton")
	#Newton_results.getresults()
	RSVD_results = results(taskname, ranks, "RSVDIters")
	RSVD_results.getresults()

	plt.figure()
	plt.grid()
	#plt.plot(ranks, Newton_results.err_Cheb_frank, marker = '.')
	#plt.plot(ranks, ALS_results.err_Cheb_frank, marker = '.')
	plt.plot(ranks, RSVD_results.err_Cheb_frank, marker = '.')
	#plt.legend(["Метод Ньютона","Переменная оптимизация","Метод Оселедца"])
	plt.xlabel(r"Ранг аппроксимации $r$")
	plt.ylabel(r"Ошибка в норме Чебышева, $||S-S_{approx}||_C$")
	#plt.title("Cheb from rank")
	#plt.savefig(f"benchmark_{taskname}.pdf")
	
	plt.figure()
	plt.grid()
	#plt.plot(ranks, Newton_results.err_Cheb_frank, marker = '.')
	#plt.plot(ranks, ALS_results.err_Cheb_frank, marker = '.')
	plt.plot(ranks, RSVD_results.err_Frob_frank, marker = '.')
	#plt.legend(["Метод Ньютона","Переменная оптимизация","Метод Оселедца"])
	plt.xlabel(r"Ранг аппроксимации $r$")
	plt.ylabel(r"Ошибка в норме Фробениуса, $||S-S_{approx}||_F$")
	#plt.title("Cheb from rank")
	#plt.savefig(f"benchmark_{taskname}.pdf")
'''

class normtest():
	def __init__(self, solvers, k, ranks, taskname):
		self.solvers = solvers
		self.k = k
		self.ranks = ranks
		self.taskname = taskname
		self.S_etalon = np.load(f"data/S_etalon_{taskname}.npy")
		self.solver_results_dict = dict((key, []) for key in solvers)
		self.n = self.S_etalon.shape[0]
	def __call__(self):
		for solver in self.solvers:
			solver_results = results(self.taskname, self.ranks, solver)
			solver_results.getresults()
			self.solver_results_dict[solver] = solver_results
			#print(type(self.solver_results_dict[solver_results]))
		#print(self.solver_results_dict)
		self.plotCheb()
		self.plotFrob()
	def plotCheb(self):
		plt.figure()
		plt.grid()
		for solver in self.solvers:
			plt.plot(ranks, self.solver_results_dict[solver].err_Cheb_frank, marker = '.')
			plt.xlabel(r"Ранг аппроксимации $r$")
			plt.ylabel(r"Ошибка в норме Чебышева, $||S-S_{approx}||_C$")
		plt.legend(self.solvers)
	def plotFrob(self):
		plt.figure()
		plt.grid()
		for solver in self.solvers:
			plt.plot(ranks, self.solver_results_dict[solver].err_Frob_frank, marker = '.')
			plt.xlabel(r"Ранг аппроксимации $r$")
			plt.ylabel(r"Ошибка в норме Фробениуса, $||S-S_{approx}||_F$")
		plt.legend(self.solvers)

class toptest():
	def __init__(self, solvers, k, ranks, taskname):
		self.solvers = solvers
		self.k = k
		self.ranks = ranks
		self.taskname = taskname
		self.S_etalon = np.load(f"data/S_etalon_{taskname}.npy")
		self.n = self.S_etalon.shape[0]
		self.inds_intersec_arrs_dict = dict((key, []) for key in solvers)
		
	def __call__(self):
		for solver in self.solvers:
			solver_results = results(self.taskname, self.ranks, solver)
			solver_results.getresults()
			#S = self.S_etalon
			self.get_intersec(self.inds_intersec_arrs_dict[solver], solver_results)
			
		self.plot()
	def get_intersec(self, inds_intersec_lst, solver_results):
		for i in range(len(self.ranks)):
			S_approx = solver_results.solution_frank[i]
			I = topsim(self.k, self.S_etalon)
			I_approx = topsim(self.k, S_approx)
			inds_intersec = 0
			for j in range(self.n):
				inds_intersec += len( set(I[j]).intersection(set(I_approx[j])) )
			inds_intersec_lst.append(inds_intersec)
	def plot(self):
		plt.figure()
		plt.grid()
		for solver in self.solvers:
			plt.plot(ranks, np.array(self.inds_intersec_arrs_dict[solver])/(self.k*self.n), marker = '.')
			plt.xlabel(r"Ранг аппроксимации $r$")
			plt.ylabel(fr"Доля сохранившихся элементов из топ-{self.k} в строке")
		plt.legend(self.solvers)
	

def topsim(k, S):
	n = S.shape[0]
	inds = []
	for i in range(n):
		inds.append(np.argpartition(S[i,:], -k)[-k:])
	return inds

if __name__=="__main__":
	taskname = "metro"
	ranks = np.arange(10, 304, 10)
	#toptest(taskname, ranks, 20)
	normtest_inst = normtest(["AltOpt"], 5, ranks, taskname)
	normtest_inst()
	plt.show()
	
	

