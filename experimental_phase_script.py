import numpy as np
import cupy as cp
import cupy.random as rnd
import scipy
import scipy.special
import math
from utils import experimental_functions
from utils.experimental_functions import *
import os
import submitit
import datetime

#cp.cuda.Device(0).use()

# sys.path.append('utils')

#start timestamp with unique identifier for name
run_timestamp = datetime.datetime.now().strftime('%Y%m-%d%H-%M%S')
#os.mkdir(with name of unique identifier)

#os.path.join(results, unique identifier)
results_path = os.path.join("utils", "results")
os.makedirs(results_path, exist_ok = True)

experiment_path = os.path.join(results_path, "n_or_more_exper")
os.makedirs(experiment_path, exist_ok = True)

run_path = os.path.join(experiment_path, run_timestamp)
os.mkdir(run_path)

# from RL-Perceptron, utils, surface functions: generate teacher, and generate students

w_teacher = gen_teacher(400)
vectors = generate_students(w_teacher, 400, 20)
students = vectors[31:32]

#s
# set range of values for learning rates 1 and 2, iterate through these values and the students

lr_1_s = cp.array([i/40 for i in range(80)])
lr_2_s = cp.array([i/40 for i in range(80)])

executor = submitit.AutoExecutor(folder="utils/results")

executor.update_parameters(timeout_min = 150, mem_gb = 4, gpus_per_node = 1, cpus_per_task = 0, slurm_array_parallelism = 256, slurm_partition = "gpu")

jobs = []
with executor.batch():
	for theta, w_student in students:
		job = executor.submit(n_or_more_neg_exp, D = 400, teacher = w_teacher, rad = theta, student = w_student, T = 12, n = 9, lr_1_s = lr_1_s, lr_2_s = lr_2_s, steps = 1600, experiment_path = run_path)
		jobs.append(job)
