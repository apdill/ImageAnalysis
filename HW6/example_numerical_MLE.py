# -*- coding: utf-8 -*-
# example_numerical_MLE.py
"""
Author:   Raghuveer Parthasarathy
Created on Tue Oct 25 22:11:12 2022
Last modified on Nov. 1, 2024

Description
-----------
Examples of Maximum Likelihood Estimation by numerically minimizing -log(p)

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as spoptimize


#%% Example 1

A = 5.0 # A for n = 0
N = 10  # number of measurements
A_array = np.arange(A, A+N)  # A values A0, A0+1, A0+2, ...

# Our "data"
x = np.random.poisson(A_array)

#%% Example 1, MLE!

# Here's the function we want to minimize:
def objfun(params, x):
    N = len(x)
    # Sum of -log(p) for each measurement
    An = params[0] + np.arange(0, N)
    nsum_logp = np.sum(An - x*np.log(An))
    return nsum_logp

# Guesses for initial parameter value: A
# let's use x[0] A
params0 = x[0];
# Bounds on the parameters: 0 to Infinity
bnds = ((0, None),)  # oddly, need the last comma to indicate that this is one item

results = spoptimize.minimize(objfun, params0, args = (x), bounds = bnds)
# Here is the best fit parameter value for A
best_A = results.x[0]
print(f'Best fit A: {best_A:.2f}')  

plt.figure()
plt.scatter(np.arange(len(x)), x, s=150, c='dodgerblue')
plt.plot(np.arange(len(x)),best_A + np.arange(len(x)), color='magenta', linewidth=4.0)
plt.xlabel('n', fontsize=16)
plt.ylabel('x', fontsize=16)
plt.title(f'True A: {A}, Est. A: {best_A:.2f}', fontsize=18)


#%% Example 2

def strangeFunction(A, B, n):
    # Inputs: parameters A, B, 
    #         array of integers n
    # returns array of "A[n]" values
    An = A + np.sqrt(A)*np.sin(np.sqrt(B*n*np.pi)) + np.sqrt(n)
    return An

A = 800.0 # A for n = 0
B = 0.07 # scale parameter
N = 100  # number of measurements
n_array = np.arange(N)
A_array = strangeFunction(A, B, n_array)  # A values

# Our "data"
x = np.random.poisson(A_array)

#%% Example 2, MLE!

# Guesses for initial parameter value: A
# let's use x[0] for A, 0.1 for B
params0 = np.array([x[0], 0.1]);    
# Bounds on the parameters: 0 to Infinity
bnds = ((0, None),(0, None))  

# Here's the function we want to minimize:
def objfun(params, x):
    N = len(x)
    n_array = np.arange(N)
    # Sum of -log(p) for each measurement
    An = strangeFunction(params[0], params[1], n_array)
    nsum_logp = np.sum(An - x*np.log(An))
    return nsum_logp
    
results = spoptimize.minimize(objfun, params0, args = (x), bounds = bnds)
# Here is the best fit parameter value for A
best_A = results.x[0]
best_B = results.x[1]
print(f'True A: {A}, Best fit A: {best_A:.2f}')  
print(f'True B: {B}, Best fit B: {best_B:.3f}') 

n_array = np.arange(N)
plt.figure()
plt.scatter(n_array, x, s=150, c='darkgray')
plt.plot(n_array,strangeFunction(best_A, best_B, n_array),
         color = 'dodgerblue', linewidth = 2.5, label = 'Best-fit params.')
plt.plot(n_array,strangeFunction(A, B, n_array),
         color = 'gold', linewidth = 2.5, label = 'True params.')
plt.xlabel('n')
plt.ylabel('x')
plt.title(f'True A: {A}, Est. A: {best_A:.2f}' + 
          f'True B: {B}, Est. B: {best_B:.4f}')
plt.legend()
