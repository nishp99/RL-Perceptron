import cupy as cp
import numpy as np

def tester():
	cp.cuda.Device(0).use()
	x_cpu = np.array([1,2,3])
	y_cpu = np.array([4,5,6])
	z_cpu = x_cpu + y_cpu
	z_c = cp.get_array_module(z_cpu)

	x_gpu = cp.asarray(x_cpu)
	y_gpu = cp.asarray(y_cpu)
	z_gpu = x_gpu + y_gpu
	z_g = cp.get_array_module(z_gpu)

	print('type z_cpu')
	print(type(z_cpu))
	print('type z_gpu')
	print(type(z_gpu))
	print('number of recognised devices:')
	print(cp.cuda.runtime.getDeviceCount())
	print('CPU name:')
	print(z_c.__name__)
	print('GPU name:')
	print(z_g.__name__)
