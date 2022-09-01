import numpy as np
import cupy as cp
import cupy.random as rnd
import numpy.random as nprnd
import math
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
def generate_students(w_teacher, D, norm):
  w_student = -w_teacher + nprnd.randn(D)/(D/4)
  students = [np.copy(w_student)]

  while w_student @ w_teacher/(20*np.linalg.norm(w_student)) < 0.9995:
  #while w_student @ w_teacher/(20*np.linalg.norm(w_student)) < 0.995:
    mag = np.linalg.norm(w_student)
    z = w_student-w_teacher
    z -= (z @ w_student)*w_student/mag**2
    z /= np.linalg.norm(z)
    #w_student -= z
    #w_student -= 0.24*z
    #w_student -= 3.78*z
    w_student -= 13*z
    w_student /= np.linalg.norm(w_student)
    w_student *= norm
    students.append(w_student.copy())
  
  overlaps = [w_teacher @ student/np.linalg.norm(student)/np.sqrt(D) for student in students]
  angles = [np.round(np.arccos(overlap),2) for overlap in overlaps]

  result = [i for i in zip(angles, students)]
  return result

"""
input - dimension, teacher, student, episode length, threshold number for correctness, pos and neg learning rates, number of steps
output - dictionary of 
"""
"""def n_or_more_neg(D, teacher, rad, student, T, n, lr_1, lr_2, steps,experiment_path):
  data = dict()
  
  R = teacher @ student / D
  Q = student @ student / D
  
  data['r'] = np.zeros(int(steps/4))
  data['q'] = np.zeros(int(steps/4))
  data['r'][0] = R
  data['q'][0] = Q
  #data['p'] = 0

  step = 0
  num_steps = steps * D
  dt = 1/D

  while step < num_steps:
    #compute quantities needed for updates
    normalised_overlap = R/ np.sqrt(Q)
    p_correct = (1 - 1 / np.pi * np.arccos(normalised_overlap))
  
    phi = (np.pi - np.arccos(normalised_overlap))/2
    theta = np.arccos(normalised_overlap)/2

    C_2 = np.sqrt(np.pi/2)*np.sin(theta)/theta
    C_1 = np.sqrt(np.pi/2)*np.sin(phi)/phi
    
    half_overlap = np.sqrt(1 + normalised_overlap)
    half_incorrect = np.sqrt(1 - normalised_overlap)

    a = 0
    b = 0

    c = 0
    d = 0
    e = 0


    for i in range(n,T+1):
      p_i = p_correct**i
      q_i = (1-p_correct)**(T-i)
      a += scipy.special.binom(T,i) * i * p_i * q_i
      b += scipy.special.binom(T,i) *(T-i) * p_i * q_i

      c += scipy.special.binom(T,i) * p_i * q_i
      d += scipy.special.binom(T,i) * i* (i-1) * p_i * q_i
      e += scipy.special.binom(T,i) * (T-i)* (T-i-1) * p_i * q_i

    #compute r,q updates
    dR = (lr_1 + lr_2)/(T*np.sqrt(D)) * (a*C_1*np.sqrt(D/2) * half_overlap - b*C_2*np.sqrt(D/2)*half_incorrect) - lr_2 *np.sqrt(2/np.pi)  * normalised_overlap
    
    dQ = (2 * (lr_1 + lr_2)/(T*np.sqrt(D)) * (a*C_1*np.sqrt(D*Q/2) * half_overlap + b*C_2*np.sqrt(D*Q/2)*half_incorrect) + 
          (lr_1**2 - lr_2**2)/(T**2 *D) * (c*T*D + d*C_1**2 + e* C_2**2)) - 2*lr_2 * np.sqrt(2*Q/np.pi) + lr_2**2/(T*D)*(D + (T-1)*2/np.pi)

    #update r, q
    R += dt * dR
    Q += dt * dQ

    if step % 4*D == 0:
      data['r'][int(step/(4*D))] = np.around(R,2)
      data['q'][int(step/(4*D))] = np.around(Q,2)
      
      p_correct = p_T_correct(Q,R,1)
                        P = 0
                        for i in range(n,T+1):
                          P += scipy.special.binom(T,i) * p_correct**i * (1-p_correct)**(T-i)
                        
                        data['p'][int(step/D) -1] = P

    step += 1

  p_correct = p_T_correct(Q,R,1)
  P = 0
  for i in range(n,T+1):
    P += scipy.special.binom(T,i) * p_correct**i * (1-p_correct)**(T-i)


  data['p'] = P
  path = os.path.join(experiment_path, f'{T}-{n}-{lr_1}-{lr_2}-{rad}')
  os.mkdir(path)
  file_path = os.path.join(path, 'dic.npy')
  np.save(file_path, data)

  #save to path
  """


def n_or_more_neg_exp(D, teacher, rad, student, T, n, lr_1_s, lr_2_s, steps, experiment_path):
  cp.cuda.Device(0).use()
  lr_1_s = cp.asarray(lr_1_s)
  lr_2_s = cp.asarray(lr_2_s)
  teacher = cp.asarray(teacher)
  student = cp.asarray(student)

  path = os.path.join(experiment_path, f'{T}-{n}-{rad}')
  os.mkdir(path)


  #create grid of learning_rates
  x_1, y_1 = cp.meshgrid(lr_2_s, lr_1_s)
  cp.cuda.Stream.null.synchronize()
  L_s = cp.concatenate((cp.expand_dims(y_1,axis = 2), cp.expand_dims(x_1,axis = 2)), axis = 2)
  cp.cuda.Stream.null.synchronize()

  size_1 = lr_1_s.size
  size_2 = lr_2_s.size

  #initialize all students
  W = cp.tile(cp.expand_dims(cp.expand_dims(student, axis = 0), axis = 0), (size_1, size_2, 1))
  cp.cuda.Stream.null.synchronize()

  #create dictionary of order parameters
  data = dict()
  data['r'] = cp.tile(cp.expand_dims(cp.zeros_like(L_s[:,:,0], dtype = float), axis =2), (1,1,int(steps/8)))
  cp.cuda.Stream.null.synchronize()
  data['q'] = cp.tile(cp.expand_dims(cp.zeros_like(L_s[:,:,0], dtype = float), axis =2), (1,1,int(steps/8)))
  cp.cuda.Stream.null.synchronize()


  step = 0
  num_steps = steps * D
  dt = 1 / D

  while step < num_steps:
    #sample T examples
    xs = rnd.randn(T, D)
    cp.cuda.Stream.null.synchronize()
    X = cp.tile(cp.expand_dims(cp.expand_dims(xs, axis = 1), axis = 1), (1,size_1, size_2, 1))
    cp.cuda.Stream.null.synchronize()

    #predicted classification
    Y_pred = cp.sign(cp.sum(cp.expand_dims(cp.copy(W), axis = 0) * X, axis = 3))
    cp.cuda.Stream.null.synchronize()

    #actual classification
    Y = cp.expand_dims(cp.expand_dims(cp.sign(cp.copy(teacher) @ xs.T), axis = 1), axis = 2)
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
    reward = cp.expand_dims(reward, axis = 2)
    cp.cuda.Stream.null.synchronize()

    #update from mean of examples over episode
    hebbian_update = cp.mean(cp.expand_dims(Y_pred, axis = 3) * X, axis = 0)
    cp.cuda.Stream.null.synchronize()

    #update students
    W += (cp.expand_dims(L_s[:,:,0] + L_s[:,:,1], axis = 2) * reward - cp.expand_dims(L_s[:,:,1], axis = 2)) * hebbian_update / cp.sqrt(D)
    
    #log order parameters      
    if step % 8*D == 0:
      print(step)
      data['r'][:,:,int(step/(8*D))] = cp.around(cp.sum(cp.expand_dims(cp.expand_dims(cp.copy(teacher), axis = 0), axis = 0) * cp.copy(W), axis = 2)/D, 5)
      data['q'][:,:,int(step/(8*D))] = cp.around(cp.sum(cp.copy(W)**2, axis = 2)/D, 5)
      
    step += 1

  #log final accuracy
  R = cp.sum(cp.expand_dims(cp.expand_dims(cp.copy(teacher), axis = 0), axis = 0) * cp.copy(W), axis = 2)/D
  Q = cp.sum(cp.copy(W)**2, axis = 2)/D
  normalised_overlap = cp.divide(R,cp.sqrt(Q))
  theta = cp.arccos(normalised_overlap)
  P = (1- theta/np.pi)

  data['r'] = cp.asnumpy(data['r'])
  data['q'] = cp.asnumpy(data['q'])

  data['p'] = cp.asnumpy(P)
  data['lr'] = cp.asnumpy(L_s)
  data['ang'] = rad

  file_path = os.path.join(path, 'dic.npy')
  np.save(file_path, data)

  


"""
input - dimension, teacher, student, episode length, threshold number for correctness, pos and neg learning rates, number of steps
output - dictionary of 
"""
def all_neg_exp(D, teacher, rad, student, T, lr_1_s, lr_2_s, steps, experiment_path):

  path = os.path.join(experiment_path, f'{T}-{rad}')
  os.mkdir(path)

  x_1, y_1 = np.meshgrid(lr_2_s, lr_1_s)
  L_s = np.concatenate((np.expand_dims(y_1,axis = 2), np.expand_dims(x_1,axis = 2)), axis = 2)

  size_1 = lr_1_s.size
  size_2 = lr_2_s.size

  W = np.tile(np.expand_dims(np.expand_dims(student, axis = 0), axis = 0), (lr_1_s.size, lr_2_s.size, 1))

  data = dict()
  data['r'] = np.tile(np.expand_dims(np.zeros_like(L_s[:,:,0], dtype = float), axis =2), (1,1,int(steps/8)))
  data['q'] = np.tile(np.expand_dims(np.zeros_like(L_s[:,:,0], dtype = float), axis =2), (1,1,int(steps/8)))

  step = 0
  num_steps = steps * D
  dt = 1 / D

  while step < num_steps:

    xs = rnd.randn(T, D)
    X = np.tile(np.expand_dims(np.expand_dims(xs, axis = 1), axis = 1), (1,size_1, size_2, 1))

    Y_pred = np.sign(np.sum(np.expand_dims(np.copy(W), axis = 0) * X, axis = 3))

    Y = np.expand_dims(np.expand_dims(np.sign(np.copy(teacher) @ xs.T), axis = 1), axis = 2)
    Y = np.tile(Y, (1, size_1, size_2))

    reward = np.all(Y_pred == Y, axis = 0)
    reward = np.expand_dims(reward, axis = 2)

    hebbian_update = np.mean(np.expand_dims(Y_pred, axis = 3) * X, axis = 0)

    W += (np.expand_dims(L_s[:,:,0] + L_s[:,:,1], axis = 2) * reward - np.expand_dims(L_s[:,:,1], axis = 2)) * hebbian_update / np.sqrt(D)
                                   
    if step % 8*D == 0:
      print(step)
      data['r'][:,:,int(step/(8*D))] = np.around(np.sum(np.expand_dims(np.expand_dims(np.copy(teacher), axis = 0), axis = 0) * np.copy(W), axis = 2)/D, 5)
      data['q'][:,:,int(step/(8*D))] = np.around(np.sum(np.copy(W)**2, axis = 2)/D, 5)
      
    step += 1

  R = np.sum(np.expand_dims(np.expand_dims(np.copy(teacher), axis = 0), axis = 0) * np.copy(W), axis = 2)/D
  Q = np.sum(np.copy(W)**2, axis = 2)/D
  normalised_overlap = np.divide(R,np.sqrt(Q))
  theta = np.arccos(normalised_overlap)
  P = (1- theta/np.pi)

  data['p'] = P
  data['lr'] = L_s
  data['ang'] = rad

  file_path = os.path.join(path, 'dic.npy')
  np.save(file_path, data)

  #save to path
