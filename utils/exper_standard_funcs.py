import numpy as np
import cupy as cp
import cupy.random as rnd
import numpy.random as nprnd
import math
import copy
import os

"""
Returns the test accuracy for a teacher-student pair with the given overlaps.
"""
def p_T_correct(Q, R, T):
    return (1 - 1 / np.pi * cp.arccos(R / cp.sqrt(Q)))**T

"""
generates teacher
"""
def gen_teacher(D):
  teacher = nprnd.randn(D)
  teacher /= np.sqrt(teacher @ teacher/D)
  return teacher

"""
generate series of students from 0 to 180 degrees from teacher
"""
def generate_students(w_teacher, D, norm, step):
  w_student = -w_teacher + nprnd.randn(D)/(D/4)
  students = [np.copy(w_student)]

  #while w_student @ w_teacher/(20*np.linalg.norm(w_student)) < 0.9995:
  while w_student @ w_teacher/(np.sqrt(D)*np.linalg.norm(w_student)) < 0.995:
    mag = np.linalg.norm(w_student)
    z = w_student-w_teacher
    z -= (z @ w_student)*w_student/mag**2
    z /= np.linalg.norm(z)
    w_student -= step*z
    #w_student -= 3.78*z
    #w_student -= 13*z
    w_student /= np.linalg.norm(w_student)
    w_student *= norm
    students.append(w_student.copy())
  
  overlaps = [w_teacher @ student/np.linalg.norm(student)/np.sqrt(D) for student in students]
  angles = [np.round(np.arccos(overlap),2) for overlap in overlaps]

  result = [i for i in zip(angles, students)]
  return result



def n_or_more_neg_exp(D, teacher, rad, student, T, n, lr_1, lr_2, steps, experiment_path):
  cp.cuda.Device(0).use()
  teacher = cp.asarray(teacher)
  student = cp.asarray(student)

  teachers = cp.tile(cp.expand_dims(teacher, axis = 0), (20, 1))
  cp.cuda.Stream.null.synchronize()

  path = os.path.join(experiment_path, f'exp_{T}-{n}-{rad}-{lr_2}')
  os.mkdir(path)


  #create grid of learning_rates
  """x_1, y_1 = cp.meshgrid(lr_2_s, lr_1_s)
        cp.cuda.Stream.null.synchronize()
        L_s = cp.concatenate((cp.expand_dims(y_1,axis = 2), cp.expand_dims(x_1,axis = 2)), axis = 2)
        cp.cuda.Stream.null.synchronize()
      
        size_1 = lr_1_s.size
        size_2 = lr_2_s.size"""

  #initialize all students
  W = cp.tile(cp.expand_dims(student, axis = 0), (20, 1))
  cp.cuda.Stream.null.synchronize()

  #create dictionary of order parameters
  data = dict()
  """data['r_mean'] = cp.zeros(int(steps/8))
    cp.cuda.Stream.null.synchronize()
    data['r_std'] = cp.zeros(int(steps/8))
    cp.cuda.Stream.null.synchronize()
    data['q_mean'] = cp.zeros(int(steps/8))
    cp.cuda.Stream.null.synchronize()
    data['q_std'] = cp.zeros(int(steps/8))"""
  # added changes!!! to save all values of R/Q for all runs (20/2 times more data)
  data['R'] = cp.zeros((int(steps / 16)+1, 20))
  data['Q'] = cp.zeros((int(steps / 16)+1, 20))
  cp.cuda.Stream.null.synchronize()


  step = 0
  num_steps = steps * D
  dt = 1 / D

  while step < num_steps:
    if step % 16*D == 0:
      print(step)
      R = cp.sum(teachers * copy.deepcopy(W) , axis = 1)/D
      Q = cp.sum(copy.deepcopy(W)**2, axis = 1)/D

      """data['r_mean'][int(step/(8*D))] = cp.around(cp.mean(R),5)
            data['r_std'][int(step/(8*D))] = cp.around(cp.std(R),5)
            data['q_mean'][int(step/(8*D))] = cp.around(cp.mean(Q),5)
            data['q_std'][int(step/(8*D))] = cp.around(cp.std(Q),5)"""
      # added bit!!!!
      data['R'][int(step / (16 * D)), :] = cp.around(copy.deepcopy(R), 5)
      data['Q'][int(step / (16 * D)), :] = cp.around(copy.deepcopy(Q), 5)

    #sample T examples
    """xs = rnd.randn(T, D)
                cp.cuda.Stream.null.synchronize()
                X = cp.tile(cp.expand_dims(cp.expand_dims(xs, axis = 1), axis = 1), (1,size_1, size_2, 1))
                cp.cuda.Stream.null.synchronize()"""

    X = rnd.randn(T, 20, D)

    #predicted classification
    Y_pred = cp.sign(cp.sum(cp.expand_dims(cp.copy(W), axis = 0) * X, axis = 2))
    cp.cuda.Stream.null.synchronize()

    #actual classification
    Y = cp.sign(cp.sum(cp.expand_dims(cp.copy(teachers), axis = 0) * X, axis = 2))
    cp.cuda.Stream.null.synchronize()

    #create filter for rewards (1/0)
    reward = Y*Y_pred + 1
    cp.cuda.Stream.null.synchronize()
    reward = cp.sum(reward, axis = 0)
    cp.cuda.Stream.null.synchronize()
    reward = reward >= 2*n
    cp.cuda.Stream.null.synchronize()
    reward = reward.astype(int)
    cp.cuda.Stream.null.synchronize()
    reward = cp.expand_dims(reward, axis = 1)
    cp.cuda.Stream.null.synchronize()

    #update from mean of examples over episode
    hebbian_update = cp.mean(cp.expand_dims(Y_pred, axis = 2) * X, axis = 0)
    cp.cuda.Stream.null.synchronize()

    #update students
    W += ((lr_1 + lr_2) * reward - lr_2) * hebbian_update / cp.sqrt(D)
    
    #log order parameters

    step += 1

  #log final accuracy
  """R = cp.sum(cp.expand_dims(cp.expand_dims(cp.copy(teacher), axis = 0), axis = 0) * cp.copy(W), axis = 2)/D
        Q = cp.sum(cp.copy(W)**2, axis = 2)/D
        normalised_overlap = cp.divide(R,cp.sqrt(Q))
        theta = cp.arccos(normalised_overlap)
        P = (1- theta/np.pi)"""

  R = cp.sum(teachers * cp.copy(W), axis=1) / D
  Q = cp.sum(cp.copy(W) ** 2, axis=1) / D

  # added bit!!!!
  data['R'][int(steps / 16),:] = cp.around(cp.copy(R), 5)
  data['Q'][int(steps / 16),:] = cp.around(cp.copy(Q), 5)

  data['R'] = cp.asnumpy(data['R'])
  data['Q'] = cp.asnumpy(data['Q'])

  """data['p'] = cp.asnumpy(P)
        data['lr'] = cp.asnumpy(L_s)
        data['ang'] = rad"""

  file_path = os.path.join(path, 'dic.npy')
  np.save(file_path, data)

  


"""
input - dimension, teacher, student, episode length, threshold number for correctness, pos and neg learning rates, number of steps
output - dictionary of 
"""
def all_neg_exp(D, teacher, rad, student, T, lr_1, lr_2, steps, experiment_path):
  cp.cuda.Device(0).use()
  teacher = cp.asarray(teacher)
  student = cp.asarray(student)

  teachers = cp.tile(cp.expand_dims(teacher, axis = 0), (20, 1))
  cp.cuda.Stream.null.synchronize()

  W = cp.tile(cp.expand_dims(student, axis = 0), (20, 1))
  cp.cuda.Stream.null.synchronize()

  path = os.path.join(experiment_path, f'exp_{T}-{rad}-{lr_2}')
  os.mkdir(path)

  data = dict()
  """data['r_mean'] = cp.zeros(int(steps/8))
  cp.cuda.Stream.null.synchronize()
  data['r_std'] = cp.zeros(int(steps/8))
  cp.cuda.Stream.null.synchronize()
  data['q_mean'] = cp.zeros(int(steps/8))
  cp.cuda.Stream.null.synchronize()
  data['q_std'] = cp.zeros(int(steps/8))"""
  # added changes!!! to save all values of R/Q for all runs (20/2 times more data)
  data['R'] = cp.zeros((int(steps / 16)+1, 20))
  data['Q'] = cp.zeros((int(steps / 16)+1, 20))
  cp.cuda.Stream.null.synchronize()

  step = 0
  num_steps = steps * D
  dt = 1 / D

  while step < num_steps:
    if step % 16*D == 0:
      print(step)
      R = cp.sum(teachers * cp.copy(W) , axis = 1)/D
      Q = cp.sum(cp.copy(W)**2, axis = 1)/D

      """data['r_mean'][int(step/(8*D))] = cp.around(cp.mean(R),5)
      data['r_std'][int(step/(8*D))] = cp.around(cp.std(R),5)
      data['q_mean'][int(step/(8*D))] = cp.around(cp.mean(Q),5)
      data['q_std'][int(step/(8*D))] = cp.around(cp.std(Q),5)"""
      #added bit!!!!
      data['R'][int(step/(16*D))] = cp.around(R,5)
      data['Q'][int(step/(16*D))] = cp.around(Q,5)
    X = rnd.randn(T, 20, D)
    #predicted classification
    Y_pred = cp.sign(cp.sum(cp.expand_dims(cp.copy(W), axis = 0) * X, axis = 2))
    cp.cuda.Stream.null.synchronize()
    #actual classification
    Y = cp.sign(cp.sum(cp.expand_dims(cp.copy(teachers), axis = 0) * X, axis = 2))
    cp.cuda.Stream.null.synchronize()

    reward = cp.all(Y_pred == Y, axis = 0)
    cp.cuda.Stream.null.synchronize()
    reward = cp.expand_dims(reward, axis = 1)
    cp.cuda.Stream.null.synchronize()

    hebbian_update = cp.mean(cp.expand_dims(Y_pred, axis = 2) * X, axis = 0)

    W += ((lr_1 + lr_2) * reward - lr_2) * hebbian_update / cp.sqrt(D)

    step += 1

  #log final accuracy
  """R = cp.sum(cp.expand_dims(cp.expand_dims(cp.copy(teacher), axis = 0), axis = 0) * cp.copy(W), axis = 2)/D
        Q = cp.sum(cp.copy(W)**2, axis = 2)/D
        normalised_overlap = cp.divide(R,cp.sqrt(Q))
        theta = cp.arccos(normalised_overlap)
        P = (1- theta/np.pi)"""

  """data['r_mean'] = cp.asnumpy(data['r_mean'])
  data['r_std'] = cp.asnumpy(data['r_std'])
  data['q_mean'] = cp.asnumpy(data['q_mean'])
  data['q_std'] = cp.asnumpy(data['q_std'])"""

  R = cp.sum(teachers * cp.copy(W), axis=1) / D
  Q = cp.sum(cp.copy(W) ** 2, axis=1) / D

  """data['r_mean'][int(step/(8*D))] = cp.around(cp.mean(R),5)
  data['r_std'][int(step/(8*D))] = cp.around(cp.std(R),5)
  data['q_mean'][int(step/(8*D))] = cp.around(cp.mean(Q),5)
  data['q_std'][int(step/(8*D))] = cp.around(cp.std(Q),5)"""
  # added bit!!!!
  data['R'][-1] = cp.around(R, 5)
  data['Q'][-1] = cp.around(Q, 5)

  data['R'] = cp.asnumpy(data['R'])
  data['Q'] = cp.asnumpy(data['Q'])


  file_path = os.path.join(path, 'dic.npy')
  np.save(file_path, data)

  #save to path
