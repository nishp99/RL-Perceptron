"""
script
"""
import numpy as np
import numpy.random as rnd
import scipy
import scipy.special
import math
from utils import surface_functions
from utils.surface_functions import *
import os
import submitit
import datetime
import sys

# sys.path.append('utils')

#start timestamp with unique identifier for name
run_timestamp = datetime.datetime.now().strftime('%Y%m-%d%H-%M%S')
#os.mkdir(with name of unique identifier)

#os.path.join(results, unique identifier)
results_path = os.path.join("utils", "results")
os.makedirs(results_path, exist_ok = True)

experiment_path = os.path.join(results_path, "n_or_more")
os.makedirs(experiment_path, exist_ok = True)

run_path = os.path.join(experiment_path, run_timestamp)
os.mkdir(run_path)

# from RL-Perceptron, utils, surface functions: generate teacher, and generate students

w_teacher = gen_teacher(400)
students = generate_students(w_teacher, 400, 20)
#s
# set range of values for learning rates 1 and 2, iterate through these values and the students

lr_1_s = np.array([i/20 for i in range(40)])
lr_2_s = np.array([i/20 for i in range(40)])

executor = submitit.AutoExecutor(folder="utils/results")

executor.update_parameters(timeout_min = 480, mem_gb = 4, gpus_per_node =0, cpus_per_task = 1, slurm_array_parallelism = 256 )

jobs = []
with executor.batch():
	for theta, w_student in students:
		job = executor.submit(n_or_more_neg, D = 400, teacher = w_teacher, rad = theta, student = w_student, T = 12, n = 7, lr_1_s = lr_1_s, lr_2_s = lr_2_s, steps = 10000, experiment_path = run_path)
		jobs.append(job)

