import numpy as np
import numpy.random as rnd
import scipy
import scipy.special
import math
import os

"""
Returns the test accuracy for a teacher-student pair with the given overlaps.
"""
def p_T_correct(Q, R, T):
    return (1 - 1 / np.pi * np.arccos(R / np.sqrt(Q)))**T

"""
generates teacher
"""
def gen_teacher(D):
  teacher = rnd.randn(D)
  teacher /= np.sqrt(teacher @ teacher/D)
  return teacher

"""
generate series of students from 0 to 180 degrees from teacher
"""
def generate_students(w_teacher, D, norm):
  w_student = -w_teacher + rnd.randn(D)/(D/4)
  students = [w_student.copy()]

  while w_student @ w_teacher/(20*np.linalg.norm(w_student)) < 0.9995:
  #while w_student @ w_teacher/(20*np.linalg.norm(w_student)) < 0.995:
    mag = np.linalg.norm(w_student)
    z = w_student-w_teacher
    z -= (z @ w_student)*w_student/mag**2
    z /= np.linalg.norm(z)
    #20
    #w_student -= z
    #5
    #w_student -= 0.24*z
    #320
    w_student -= z
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


def n_or_more(D, teacher, rad, student, T, n, lr_1, steps, experiment_path):

  path = os.path.join(experiment_path, f'{T}-{n}-{rad}')
  os.mkdir(path)

  R = teacher @ student /D
  Q = student @ student /D

  data = dict()
  data['r'] = np.zeros(int(steps/8))
  data['q'] = np.zeros(int(steps/8))

  step = 0
  num_steps = steps * D
  dt = 1 / D

  while step < num_steps:
    normalised_overlap = np.divide(np.copy(R),np.sqrt(np.copy(Q)))
    theta = np.arccos(normalised_overlap)
    p_correct = (1- theta/np.pi)
    phi = (np.pi - theta)/2

    C_2 = np.sqrt(np.pi/2)*np.divide(np.sin(theta/2),(theta/2))
    C_1 = np.sqrt(np.pi/2)*np.divide(np.sin(phi),phi)

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
    dR = lr_1/(T*np.sqrt(D)) * (a*C_1*np.sqrt(D/2) * half_overlap - b*C_2*np.sqrt(D/2)*half_incorrect)
    
    dQ = (2 * lr_1/(T*np.sqrt(D)) * (a*C_1*np.sqrt(D*Q/2) * half_overlap + b*C_2*np.sqrt(D*Q/2)*half_incorrect) + 
          lr_1**2/(T**2 *D) * (c*T*D + d*C_1**2 + e* C_2**2))

    #update r, q
    R += dt * dR
    Q += dt * dQ

    if step % 8*D == 0:
      print(step)
      data['r'][int(step/(8*D))] = np.around(np.copy(R),5)
      data['q'][int(step/(8*D))] = np.around(np.copy(Q),5)
      
    step += 1

  normalised_overlap = np.divide(np.copy(R),np.sqrt(np.copy(Q)))
  theta = np.arccos(normalised_overlap)
  P = (1- theta/np.pi)

  data['p'] = P
  data['lr'] = L_s
  data['ang'] = rad

  file_path = os.path.join(path, 'dic.npy')
  np.save(file_path, data)

  


"""
input - dimension, teacher, student, episode length, threshold number for correctness, pos and neg learning rates, number of steps
output - dictionary of 
"""
def all(D, teacher, rad, student, T, lr_1, steps, experiment_path):

  path = os.path.join(experiment_path, f'{T}-{rad}')
  os.mkdir(path)

  R = teacher @ student /D
  Q = student @ student /D

  data = dict()
  data['r'] = np.zeros(int(steps/8))
  data['q'] = np.zeros(int(steps/8))

  step = 0
  num_steps = steps * D
  dt = 1/D

  while step < num_steps:
    #compute quantities needed for updates
    normalised_overlap = np.divide(np.copy(R),np.sqrt(np.copy(Q)))
    theta = np.arccos(normalised_overlap)
    p_correct_all = (1 - 1 / np.pi * np.arccos(normalised_overlap))**T
  
    phi = (np.pi - theta)/2
    C_1 = np.sqrt(np.pi/2)*np.sin(phi)/phi
    
    half_overlap = np.sqrt(1 + normalised_overlap)

    #compute r,q updates
    dR = lr_1 * C_1/np.sqrt(2) * p_correct_all * half_overlap
        
    dQ = (lr_1**2*(D+ (T-1)*C_1**2)/(T*D) + lr_1*np.sqrt(2*Q)*half_overlap*C_1)*p_correct_all

    #update r, q
    R += dt * dR
    Q += dt * dQ

    if step % 8*D == 0:
      data['r'][int(step/(8*D))] = np.around(np.copy(R),5)
      data['q'][int(step/(8*D))] = np.around(np.copy(Q),5)

    step += 1


  normalised_overlap = np.divide(np.copy(R),np.sqrt(np.copy(Q)))
  theta = np.arccos(normalised_overlap)
  P = (1- theta/np.pi)
  
  data['p'] = P
  data['lr'] = L_s
  data['ang'] = rad

  file_path = os.path.join(path, 'dic.npy')
  np.save(file_path, data)

  #save to path
