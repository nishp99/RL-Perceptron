import numpy as np
import cupy as cp
import cupy_function
from cupy_function import *
import os
import submitit
import datetime

#cp.cuda.Device(0).use()
# sys.path.append('utils')

#start timestamp with unique identifier for name
run_timestamp = datetime.datetime.now().strftime('%Y%m-%d%H-%M%S')
#os.mkdir(with name of unique identifier)

experiment_path = os.path.join('utils', "test")
os.makedirs(experiment_path, exist_ok = True)

run_path = os.path.join(experiment_path, run_timestamp)
os.mkdir(run_path)

executor = submitit.AutoExecutor(folder="utils/test")

executor.update_parameters(timeout_min = 5, mem_gb = 1, gpus_per_node =1, cpus_per_task = 0, slurm_array_parallelism = 1, slurm_partition = "gpu")

job = executor.submit(tester)