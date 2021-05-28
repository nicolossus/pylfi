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
#from pylfi.plots import histplot, kdeplot, rugplot
from pylfi.priors import InvGamma, Normal, Uniform

rng = np.random.default_rng()
N = 1000

x = np.linspace(150, 200, 1000)

mu_true = 163
sigma_true = 15
true_parameter_values = [mu_true, sigma_true]
likelihood = stats.norm(loc=mu_true, scale=sigma_true)
likelihood = Normal(loc=mu_true, scale=sigma_true, name="likelihood")
data = likelihood.rvs(size=N, seed=42)

sigma_noise = 0.1
noise = rng.normal(0, sigma_noise, N)

# observation
# obs = np.mean(data + noise)
#obs = np.mean(data)
obs = np.array([np.mean(data), np.std(data)])


def gaussian_model(mu, sigma, n_samples=1000):
    """Simulator model, returns summary statistic"""
    sim = stats.norm(loc=mu, scale=sigma).rvs(size=n_samples)
    sumstat = np.array([np.mean(sim), np.std(sim)])
    #sumstat = np.mean(sim)
    return sumstat


# priors
mu = Normal(loc=165, scale=2, name="mu", tex="$\mu$")
sigma = Normal(loc=17, scale=4, name="sigma", tex="$\sigma$")
priors = [mu, sigma]


# initialize sampler
sampler = RejectionABC(obs, gaussian_model, priors, distance='l2', seed=42)

# inference config
num_simulations = 2000
n_samples = 10
epsilon = 1.0


# run inference
#journal = sampler.sample(n_sims=num_simulations, epsilon=epsilon, log=True)
journal = sampler.sample(n_samples=n_samples, epsilon=epsilon)

# journal
#mu_posterior = journal.get_accepted_parameters["mu"]
#sigma_posterior = journal.get_accepted_parameters["sigma"]
print(journal.get_acceptance_ratio)
print(journal.sampler_results)
print(journal.sampler_summary)
print(journal.sampler_stats)

'''
journal.histplot(bins='knuth',
                 figsize=(8, 4),
                 rug=True,
                 plot_style='pylfi',
                 # true_parameter_values=true_parameter_values,
                 )
'''
# az.plot_kde(mu_posterior)
#az.plot_posterior(mu_posterior, var_names=['mu'])
#az.plot_kde(mu_posterior, values2=sigma_posterior)
# plt.show()
