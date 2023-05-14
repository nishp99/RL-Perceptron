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

experiment_path = os.path.join(results_path, "partial")
os.makedirs(experiment_path, exist_ok = True)

run_path = os.path.join(experiment_path, run_timestamp)
os.mkdir(run_path)

# from RL-Perceptron, utils, surface functions: generate teacher, and generate students


w_teacher = gen_teacher(900)
vectors = generate_students(w_teacher, 900, 30, 1)
student = vectors[47]
#T_s = [5,9,12]
T = 11
lr_2s = [0.2]
n_s = [2,5,9]

#s
executor_1 = submitit.AutoExecutor(folder="utils/truefinal/partial")

executor_1.update_parameters(timeout_min = 3000, mem_gb = 4, gpus_per_node = 1, cpus_per_task = 1, slurm_array_parallelism = 4, slurm_partition = "gpu")

jobs = []
with executor_1.batch():
    for lr_2 in lr_2s:
        for n in n_s:
            job = executor_1.submit(partial_exp, D = 900, teacher = w_teacher, student = student[1], T = T, n = n, lr_1 = 1, lr_2 = lr_2 , steps = 8000, experiment_path = run_path)
            jobs.append(job)

executor_2 = submitit.AutoExecutor(folder="utils/truefinal/partial")

executor_2.update_parameters(timeout_min = 3000, mem_gb = 4, gpus_per_node = 0, cpus_per_task = 4, slurm_array_parallelism = 128)

jobs_2 = []
with executor_2.batch():
    for lr_2 in lr_2s:
        for n in n_s:
            job_2 = executor_2.submit(partial_ode, D = 900, teacher = w_teacher, student = student[1], T = T, n = n, lr_1 = 1, lr_2 = lr_2, steps = 8000, experiment_path = run_path)
            jobs_2.append(job_2)