#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from pylfi.features import SpikingFeatures
from pylfi.inferences import RejectionABC
from pylfi.models import HodgkinHuxley
from pylfi.priors import Normal

T = 120.
dt = 0.025
t_stim_on = 10
t_stim_off = 110
duration = t_stim_off - t_stim_on
hh = HodgkinHuxley()


def set_plot_style():
    """Set plot style"""
    sns.set()
    sns.set_context("paper")
    sns.set_style("darkgrid", {"axes.facecolor": "0.96"})

    # Set fontsizes in figures
    params = {'legend.fontsize': 'large',
              'axes.labelsize': 'large',
              'axes.titlesize': 'large',
              'xtick.labelsize': 'large',
              'ytick.labelsize': 'large',
              'legend.fontsize': 'large',
              'legend.handlelength': 2}
    plt.rcParams.update(params)
    plt.rc('text', usetex=True)


def stimulus(t):
    return 10 if t_stim_on <= t <= t_stim_off else 0


def simulator(gbar_K, feature='spike_rate'):
    hh.gbar_K = gbar_K
    hh.solve(stimulus, T, dt)
    t = hh.t
    V = hh.V
    features = SpikingFeatures(V, t, duration, t_stim_on, threshold=0)
    sum_stat = getattr(features, feature)
    return sum_stat


set_plot_style()

prior = Normal(loc=36, scale=2, name=r'$\bar{g}_K$')
x = np.linspace(29, 43, 1000)
# prior.plot_prior(x)
#thetas = prior.rvs(size=100)
thetas = np.linspace(30, 45, 50)
sum_stats = np.array([simulator(theta) for theta in thetas])
print(sum_stats)
#index = np.where(sum_stats == 0.09)
plt.scatter(thetas, sum_stats)
#plt.scatter(sum_stats, thetas)
plt.show()
'''
theta = thetas[index]
hh.gbar_K = theta[0]
hh.solve(stimulus, T, dt)
plt.plot(hh.t, hh.V, label=f'{hh.gbar_K=}')
plt.legend()
plt.show()

hh.gbar_K = 32
hh.solve(stimulus, T, dt)
plt.plot(hh.t, hh.V, label=f'{hh.gbar_K=}')
plt.legend()
plt.show()

hh.gbar_K = 38
hh.solve(stimulus, T, dt)
plt.plot(hh.t, hh.V, label=f'{hh.gbar_K=}')
plt.legend()
plt.show()
'''
