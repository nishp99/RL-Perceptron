import cupy as cp
import numpy as np
import cupy.random as rnd
import scipy
import scipy.special
import math
from utils import exper_standard_funcs
from utils.exper_standard_funcs import *
from utils import standard_neg_funcs
from utils.standard_neg_funcs import *
#from utils import standard_funcs
#from utils.standard_funcs import *

import os
import submitit
import datetime

#start timestamp with unique identifier for name
run_timestamp = datetime.datetime.now().strftime('%Y%m-%d%H-%M%S')
#os.mkdir(with name of unique identifier)

#os.path.join(results, unique identifier)
results_path = os.path.join("utils", "truefinal")
os.makedirs(results_path, exist_ok = True)

experiment_path = os.path.join(results_path, "all_case_neg")
os.makedirs(experiment_path, exist_ok = True)

run_path = os.path.join(experiment_path, run_timestamp)
os.mkdir(run_path)

# from RL-Perceptron, utils, surface functions: generate teacher, and generate students


w_teacher = gen_teacher(900)
vectors = generate_students(w_teacher, 900, 30, 1)
student = vectors[47]
T_s = [11]
#T = 12
#n_s = [7,9,11]
lr_2s = [0, 0.005, 0.01, 0.02]

#s
executor_1 = submitit.AutoExecutor(folder="utils/truefinal/all_case_neg")

executor_1.update_parameters(timeout_min = 3000, mem_gb = 4, gpus_per_node = 1, cpus_per_task = 1, slurm_array_parallelism = 5, slurm_partition = "gpu")

jobs = []
with executor_1.batch():
	for T in T_s:
		for lr_2 in lr_2s:
			job = executor_1.submit(all_neg_exp, D = 900, teacher = w_teacher, rad = student[0], student = student[1], T = T, lr_1 = 1, lr_2 = lr_2 , steps = 8000, experiment_path = run_path)
			jobs.append(job)

executor_2 = submitit.AutoExecutor(folder="utils/truefinal/all_case_neg")

executor_2.update_parameters(timeout_min = 3000, mem_gb = 4, gpus_per_node = 0, cpus_per_task = 1, slurm_array_parallelism = 128)

jobs_2 = []
with executor_2.batch():
	for T in T_s:
		for lr_2 in lr_2s:
			job_2 = executor_2.submit(all_neg, D = 900, teacher = w_teacher, rad = student[0], student = student[1], T = T, lr_1 = 1, lr_2 = lr_2, steps = 8000, experiment_path = run_path)
			jobs_2.append(job_2)

