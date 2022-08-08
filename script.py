"""
script
"""
import numpy as np
import numpy.random as rnd
import scipy
import scipy.special
import math
import surface functions
import os
import submitit
import datetime

#start timestamp with unique identifier for name
experiment = datetime.datetime.now().strftime('%Y%m-%d%H-%M%S')

#os.mkdir(with name of unique identifier)

#os.path.join(results, unique identifier)
path = os.path.join("utils/results/n or more", experiment)
os.mkdir(path)


# from RL-Perceptron, utils, surface functions: generate teacher, and generate students

w_teacher = gen_teacher(400)
students = generate_students(w_teacher, 400)

# set range of values for learning rates 1 and 2, iterate through these values and the students

lr_1_s = [i for i in range(1)][1:]
lr_2_s = [i/20 for i in range(1)][1:]

from cifar_curriculum import run_expt

executor = submitit.AutoExecutor(folder="log_exp")

executor.updateparameters(timeout_min = 20, mem_gb = 1, gpus_per_node =0, cpus_per_task = 1)


for i in lr_1_s:
	for j in lr_2_s:
		for theta, w_student in students[100:101]:
			job = executor.submit(n_or_more_neg, D = 400, teacher = w_teacher, rad = theta, student = w_student, T = 12, n = 9, lr_1 = i, lr_2 = j, steps = 20, experiment_path = experiment)


