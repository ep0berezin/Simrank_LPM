import numpy as np
import simrank_main as sm
import datetime as dt
import argparse

dateformat = "%Y-%m-%d-%H-%M-%S"
path = "results/solutions"


def getsolution(taskname, ranks, solver, maxiter):
	args = {
	"acc":1e-5,
	"m_Krylov": 15, 
	"rank": 200, 
	"k_iter_max": maxiter, 
	"taskname": taskname, 
	"c": 0.8, 
	"solver": solver,  
	"optimize" : True}
	#args = sm.load_args(sm.proc_args().argsfrom)
	for rank in ranks:
		args['rank'] = rank
		S = sm.launch(args)
		solvers = args['solvers']
		for solver in solvers:
			np.save(
			f"{path}/S_{args['taskname']}_rank_{rank}_c_{args['c']}_solver_{solver}_{dt.datetime.now().strftime(dateformat)}.npy",
			S[list(S.keys())[0]],)

if __name__ == "__main__":
	#Note: ranks:
	#ranks_metro = np.arange(10, 304, 10)
	#ranks_eumail = np.arange(100, 1006, 100)
	#ranks_fb = np.arange(100, 4097, 100)
	parser = argparse.ArgumentParser("collect_solutions.py")
	parser.add_argument("-tn", "--taskname", help="specify task name, for options see simrank_main")
	parser.add_argument("-sl", "--solver", help="specify solver, for options see simrank_main")
	parser.add_argument("-rk", "--ranks", help="Usage: --ranks start_rank,end_rank,rank_step (end rank is as end of open interval)")
	parser.add_argument("-it", "--maxiter", help = "Specify maxiter")
	clargs = parser.parse_args()
	solver = clargs.solver
	taskname = clargs.taskname
	ranks_ses = np.array(clargs.ranks.split(',')).astype(int)
	maxiter = int(clargs.maxiter)
	getsolution(taskname, np.arange(ranks_ses[0], ranks_ses[1], ranks_ses[2]), solver, maxiter)

