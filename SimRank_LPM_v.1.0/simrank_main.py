import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
print(f"Using threads: {os.environ.get('OMP_NUM_THREADS')}")
import importlib
import argparse
import solve_simrank as solvesim
import json

def launch(args):
	dataset = importlib.import_module(args["taskname"])
	A = dataset.ObtainMatrix()
	args.update({"A" : A})
	S = solvesim.Solve(**args)
	return S

def load_args(fname):
	f = open(f"args/{fname}")
	args = json.load(f)
	f.close()
	return args

def proc_args():
	parser = argparse.ArgumentParser("simrank_main.py")
	parser.add_argument("-af", "--argsfrom", help="specify .json file containing launch parameters; use 'py' to launch with parameters, specified in the simrank_main.py code.")
	clargs = parser.parse_args()
	return clargs

if __name__=="__main__":
	if len(sys.argv)==1: 
		print("Please use simrank_main.py -h to show help.")
		sys.exit(1)
	clargs = proc_args() 
	args_py = {"acc":1e-5, "m_Krylov": 15, "rank": 50, "k_iter_max": 100, "taskname": "metro", "c": 0.8, "solver": "QMin_Newton", "optimize" : True}
	if clargs.argsfrom != "py":
		launch(load_args(clargs.argsfrom))
	if clargs.argsfrom == "py":
		launch(args_py)
