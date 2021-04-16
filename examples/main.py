import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from pylfi.inferences import RejectionABC
from pylfi.priors import Normal, Uniform

logging.basicConfig(level=logging.INFO)

rng = np.random.default_rng()
N = 150

mu_true = 163
sigma_true = 15
true_parameter_values = [mu_true, sigma_true]
likelihood = stats.norm(loc=mu_true, scale=sigma_true)
data = likelihood.rvs(size=N)

sigma_noise = 0.1
noise = rng.normal(0, sigma_noise, N)

# observation
# obs = np.mean(data + noise)
obs = np.mean(data)


def gaussian_model(mu, sigma, n_samples=150):
    """Simulator model, returns summary statistic"""
    sim = stats.norm(loc=mu, scale=sigma).rvs(size=n_samples)
    return np.mean(sim)


# priors
mu = Normal(165, 20, name="mu", tex="$\mu$")
sigma = Uniform(5, 30, name="sigma", tex="$\sigma$")
priors = [mu, sigma]

# initialize sampler
sampler = RejectionABC(obs, gaussian_model, priors, distance='l2')

# inference config
num_simulations = 1000
epsilon = 0.5

# run inference
journal = sampler.sample(num_simulations, epsilon)

# journal
samples_mu = journal.get_accepted_parameters["mu"]
samples_sigma = journal.get_accepted_parameters["sigma"]
