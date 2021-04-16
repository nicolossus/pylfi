#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from pylfi.inferences import RejectionABC
from pylfi.plots import histplot, kdeplot, rugplot
from pylfi.priors import InvGamma, Normal, Uniform


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


set_plot_style()

# Task: infer variance parameter in zero-centered Gaussian model

# global variables
groundtruth = 2.0  # true variance
N = 10000          # number of observations
alpha = 60         # prior hyperparameter (inverse gamma distribution)
beta = 130         # prior hyperparameter (inverse gamma distribution)

# specify domain
(dmin, dmax) = (-5, 5)
x = np.arange(dmin, dmax, (dmax - dmin) / 100.)

# observed data
likelihood = Normal(loc=0, scale=np.sqrt(groundtruth),
                    name='observation', seed=42)
likelihood_pdf = likelihood.pdf(x)
obs_data = likelihood.rvs(size=N)

# true posterior
alphaprime = alpha + N / 2
betaprime = beta + 0.5 * np.sum(obs_data**2)
posterior = InvGamma(alphaprime, loc=0, scale=betaprime,
                     name='posterior', seed=42)
posterior_pdf = posterior.pdf(x)


# setup for inference
def summary_statistic(data):
    return np.var(data)


def simulator(theta, seed=42, N=10000):
    """Simulator model, returns summary statistic"""
    model = Normal(loc=0, scale=np.sqrt(theta), name='simulator', seed=seed)
    sim = model.rvs(size=N)
    sim_sumstat = summary_statistic(sim)
    return sim_sumstat


# observation
observation = groundtruth

# prior
theta = InvGamma(alpha, loc=0, scale=beta, seed=42,
                 name='theta', tex=r'$\theta$')
priors = [theta]

# initialize sampler
sampler = RejectionABC(observation, simulator, priors, distance='l2')

# inference config
num_simulations = 1000
epsilon = 0.5

# run inference
journal = sampler.sample(num_simulations, epsilon)


#samples_mu = journal.get_accepted_parameters["theta"]


'''
# plot stuff
def plot_observation(ax=None):
    kdeplot(x, likelihood_pdf, fill=True, label='Likelihood')
    rugplot(obs_data, label='Observation')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


# plot_observation()
# theta.plot_prior(x)
'''

'''
# BENCHMARK
n_experiments = 50
run_times = np.zeros(n_experiments)
for i in range(n_experiments):
    # start timer
    t0 = time.time()
    # function call
    journal = sampler.sample(num_simulations, epsilon)
    # end timer
    run_times[i] = time.time() - t0
run_time_avg = np.mean(run_times)
output = f"Average run time after {n_experiments} runs: {run_time_avg:.5f} secs\n"
print(output)
# Benchmarking simulation call loop
# Average run time after 50 runs: 0.47421 secs (list comprehension)
# Average run time after 50 runs: 0.47218 secs (Python loop)
'''
