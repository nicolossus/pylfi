#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.integrate import odeint

'''
#from scipy solve_ivp docs:

def lotkavolterra(t, z, a, b, c, d):
    x, y = z
    return [a*x - b*x*y, -c*y + d*x*y]

sol = solve_ivp(lotkavolterra, [0, 15], [10, 5], args=(1.5, 1, 3, 1),
                dense_output=True)
t = np.linspace(0, 15, 300)
z = sol.sol(t)
import matplotlib.pyplot as plt
plt.plot(t, z.T)
plt.xlabel('t')
plt.legend(['x', 'y'], shadow=True)
plt.title('Lotka-Volterra System')
plt.show()
'''


class LotkaVolterra:

    def __init__(self):
        pass

    def __call__(self):
        return self.simulate()

    def simulate(self):
        pass

    def generate_data(self):
        pass

    def summary_statistics(self):
        pass


# Definition of parameters
a = 1.0
b = 0.1
c = 1.5
d = 0.75

# initial population of rabbits and foxes
X0 = [10.0, 5.0]
# size of data
size = 100
# time lapse
time = 15
t = np.linspace(0, time, size)


# Lotka-Volterra equation
def dX_dt(X, t, a, b, c, d):
    """ Return the growth rate of fox and rabbit populations. """
    return np.array([a * X[0] - b * X[0] * X[1], -c * X[1] + d * b * X[0] * X[1]])


# simulator function
def competition_model(a, b):
    return odeint(dX_dt, y0=X0, t=t, rtol=0.01, args=(a, b, c, d))


# function for generating noisy data to be used as observed data.
def add_noise(a, b, c, d):
    noise = np.random.normal(size=(size, 2))
    simulated = competition_model(a, b) + noise
    return simulated


# plotting observed data.
observed_true = competition_model(a, b)
observed = add_noise(a, b, c, d)
_, ax = plt.subplots(figsize=(8, 4))
ax.plot(observed_true[:, 0], label="prey true")
ax.plot(observed_true[:, 1], label="predator true")
ax.plot(observed[:, 0], "x", label="prey")
ax.plot(observed[:, 1], "x", label="predator")
ax.set_xlabel("time")
ax.set_ylabel("population")
ax.set_title("Observed data")
ax.legend()
plt.show()
