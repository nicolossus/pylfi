#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from pylfi.inferences import RejectionABC
# from pylfi.plots import histplot, kdeplot, rugplot
from pylfi.priors import InvGamma, Normal, Uniform

'''
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
'''

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
likelihood = Normal(loc=0, scale=np.sqrt(groundtruth), name='observation')
likelihood_pdf = likelihood.pdf(x)
obs_data = likelihood.rvs(size=N, seed=42)

sigma_noise = 0.1
noise = np.random.RandomState().normal(0, sigma_noise, N)
# obs_data += noise

# true posterior
alphaprime = alpha + N / 2
betaprime = beta + 0.5 * np.sum(obs_data**2)
posterior = InvGamma(alphaprime, loc=0, scale=betaprime, name='posterior')
posterior_pdf = posterior.pdf(x)


# setup for inference
def summary_statistic(data):
    return np.var(data)


def simulator(theta, seed=42, N=10000):
    """Simulator model, returns summary statistic"""
    model = Normal(loc=0, scale=np.sqrt(theta), name='simulator')
    sim = model.rvs(size=N, seed=seed)
    sim_sumstat = summary_statistic(sim)
    return sim_sumstat


# observation
# observation = groundtruth
observation = summary_statistic(obs_data)

# prior
theta = InvGamma(alpha, loc=0, scale=beta, name='theta', tex=r'$\theta$')
priors = [theta]

# initialize sampler
sampler = RejectionABC(observation, simulator, priors,
                       distance='l2', seed=42)

# inference config
num_simulations = 3000
n_samples = 10
epsilon = 0.2

# run inference
#journal = sampler.sample(n_sims=num_simulations, epsilon=epsilon)
journal = sampler.sample(n_samples=n_samples, epsilon=epsilon)
print(f"{journal.get_acceptance_ratio=}")
print(journal.sampler_results)
print(journal.sampler_summary)
print(journal.sampler_stats)


'''
# Plot relationship between the unknown probability parameter theta and the
# simulated simulated summary statistic. Each point in the plot represents
# a particular value of proposal theta and the corresponding distance between
# the simulated summary statistic, s_sim, and the observed, s_obs.
samples = np.asarray(journal.get_accepted_parameters["theta"])
distances = np.asarray(journal.get_raw_distances)

plt.scatter(distances, samples, facecolors='none', edgecolor='C0')
plt.xlabel(r'$s_{\mathrm{sim}} - s_{\mathrm{obs}}$')
plt.ylabel(r'$\theta$')
plt.axvline(0, color='k', ls='--')
plt.plot(0, groundtruth, marker='x', markersize=10, color="k")
plt.show()
'''

'''
journal.histplot(bins='freedman',
                 figsize=(8, 4),
                 # rug=True,
                 plot_style='pylfi',
                 # point_estimate='mean',
                 # true_parameter_values=[groundtruth]
                 )
'''

# samples_mu = journal.get_accepted_parameters["theta"]


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
