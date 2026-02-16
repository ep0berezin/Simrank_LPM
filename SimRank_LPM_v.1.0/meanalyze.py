import numpy as np
import glob
#import simrank_main as sm
import matplotlib.pyplot as plt

class results:
	def __init__(self, taskname, ranks, solver):
		self.ranks = ranks
		self.taskname = taskname
		self.err_Frob_frank = []
		self.err_Cheb_frank = []
		self.solver = solver
		self.solution_frank = []
		self.S_etalon = np.load(f"data/S_etalon_{taskname}.npy")
	def getresults(self):
		for rank in ranks:
			args = {"acc":1e-5, "m_Krylov": 15, "rank": rank, "k_iter_max": 100, "taskname": self.taskname, "c": 0.8, "solvers":[self.solver, "Compare"],  "optimize" : False, "showfig": False}
			#Sdict, err_Frob, err_Cheb = sm.launch(args)
			pattern = f"results/solutions/S_{taskname}_rank_{rank}_c_{args['c']}_solver_{self.solver}_*.npy"
			matching = glob.glob(pattern)
			if matching: #here, if file was not found, old S is used
				filename = matching[0]
				S = np.load(filename)
				print(f"Loaded: {filename}")
				diff = self.S_etalon-S
				S = np.eye(2) #to zero memory
				err_Frob = np.linalg.norm(diff, ord='fro')
				err_Cheb = np.max(np.abs(diff))
				self.err_Frob_frank.append(err_Frob), self.err_Cheb_frank.append(err_Cheb), self.solution_frank.append(S)
			else:
				print(f"File not found for {pattern}")
				#zeroing maybe a kostyl, but it prevents misleading when not rank-consistent datas.
				self.err_Frob_frank.append(0.), self.err_Cheb_frank.append(0.), self.solution_frank.append( np.zeros((self.S_etalon.shape[0],self.S_etalon.shape[1])) )
				
				
class normtest():
	def __init__(self, solvers, k, ranks, taskname, linestyles_dict, legend):
		self.solvers = solvers
		self.k = k
		self.ranks = ranks
		self.taskname = taskname
		self.S_etalon = np.load(f"data/S_etalon_{taskname}.npy")
		self.solver_results_dict = dict((key, []) for key in solvers)
		self.n = self.S_etalon.shape[0]
		self.linestyle = linestyles_dict
		self.legend = legend
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
			plt.plot(ranks, self.solver_results_dict[solver].err_Cheb_frank, color='black', linestyle=self.linestyle[solver], marker = '.')
			plt.xlabel(r"Approximation rank $r$")
			plt.ylabel(r"Chebyshev norm error, $||S-S_{approx}||_C$")
		plt.legend(self.legend)
		plt.savefig(f"plotCheb_{self.taskname}.pdf")
		print("Cheb err plot saved.")

	def plotFrob(self):
		plt.figure()
		plt.grid()
		for solver in self.solvers:
			plt.plot(ranks, self.solver_results_dict[solver].err_Frob_frank, color='black', linestyle=self.linestyle[solver], marker = '.')
			plt.xlabel(r"Approximation rank $r$")
			plt.ylabel(r"Frobsnius norm error, $||S-S_{approx}||_F$")
		plt.legend(self.legend)
		plt.savefig(f"plotFrob_{self.taskname}.pdf")
		print("Frob err plot saved.")

class toptest():
	def __init__(self, solvers, k, ranks, taskname, linestyles_dict, legend):
		self.solvers = solvers
		self.k = k
		self.ranks = ranks
		self.taskname = taskname
		self.S_etalon = np.load(f"data/S_etalon_{taskname}.npy")
		self.n = self.S_etalon.shape[0]
		self.inds_intersec_arrs_dict = dict((key, []) for key in solvers)
		self.linestyle = linestyles_dict
		self.legend = legend
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
			plt.plot(ranks, np.array(self.inds_intersec_arrs_dict[solver])/(self.k*self.n), color='black', linestyle=self.linestyle[solver], marker = '.')
			plt.xlabel(r"Approximation rank $r$")
			plt.ylabel(fr"Percentage of elements from top-{self.k} remaind in row")
		#plt.legend(self.solvers)
		plt.legend(self.legend)
		plt.savefig(f"plot_top_{self.taskname}.pdf")
		print("toptest plot saved.")

def topsim(k, S):
	n = S.shape[0]
	inds = []
	for i in range(n):
		inds.append(np.argpartition(S[i,:], -k)[-k:])
	return inds

if __name__=="__main__":
	taskname = "eumail"
	ranks = np.arange(100,1100,100)
	toptest_inst = toptest(["AltOpt","Optimization_Newton"], 10, ranks, taskname, linestyles_dict={"AltOpt":'--', "Optimization_Newton":'-'}, legend=["AltOpt", "OptNewt"])
	toptest_inst()
	normtest_inst = normtest(["AltOpt","Optimization_Newton"], 10, ranks, taskname, linestyles_dict={"AltOpt":'--', "Optimization_Newton":'-'}, legend=["AltOpt", "OptNewt"])
	normtest_inst()

