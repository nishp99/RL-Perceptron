import cupy as cp
import numpy as np
import cupy.random as rnd
import scipy
import scipy.special
import math
from utils import experimental_functions
from utils.experimental_functions import *
from utils.surface_functions import *
import os
import submitit
import datetime

#start timestamp with unique identifier for name
run_timestamp = datetime.datetime.now().strftime('%Y%m-%d%H-%M%S')
#os.mkdir(with name of unique identifier)

#os.path.join(results, unique identifier)
results_path = os.path.join("utils", "new_results")
os.makedirs(results_path, exist_ok = True)

experiment_path = os.path.join(results_path, "phase")

os.makedirs(experiment_path, exist_ok = True)

run_path = os.path.join(experiment_path, run_timestamp)
os.mkdir(run_path)

# from RL-Perceptron, utils, surface functions: generate teacher, and generate students

w_teacher = gen_teacher(900)
vectors = generate_students(w_teacher, 900, 30, 1)
students = vectors[47:48]

#s
# set range of values for learning rates 1 and 2, iterate through these values and the students

lr_1_s = np.array([i/40 for i in range(80)])
lr_2_s = np.array([i/40 for i in range(80)])

executor_1 = submitit.AutoExecutor(folder="utils/new_results")

executor_1.update_parameters(timeout_min = 1500, mem_gb = 5, gpus_per_node = 1, cpus_per_task = 1, slurm_array_parallelism = 1, slurm_partition = "gpu")

jobs = []
with executor_1.batch():
	for theta, w_student in students:
		for i in range(10):
			job = executor_1.submit(n_or_more_neg_exp, D = 900, teacher = w_teacher, rad = theta, student = w_student, T = 12, n = 9, lr_1_s = lr_1_s, lr_2_s = lr_2_s, steps = 4800, experiment_path = run_path)
			jobs.append(job)

executor_2 = submitit.AutoExecutor(folder="utils/new_results")

executor_2.update_parameters(timeout_min = 1000, mem_gb = 4, gpus_per_node =0, cpus_per_task = 4, slurm_array_parallelism = 256)

jobs = []
with executor_2.batch():
	for theta, w_student in students:
		job = executor_2.submit(n_or_more_neg, D = 900, teacher = w_teacher, rad = theta, student = w_student, T = 12, n = 9, lr_1_s = lr_1_s, lr_2_s = lr_2_s, steps = 8000, experiment_path = run_path)
		jobs.append(job)
