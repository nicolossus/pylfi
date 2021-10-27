#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
import pickle
import sys
from abc import ABCMeta, abstractmethod
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np
from pylfi.checks import check_distance_str
from pylfi.distances import euclidean
from pylfi.journal import Journal

'''
change implementation to something like this:

sbi takes any function as simulator. Thus, sbi also has the flexibility to use
simulators that utilize external packages, e.g., Brian (http://briansimulator.org/),
nest (https://www.nest-simulator.org/), or NEURON (https://neuron.yale.edu/neuron/).
External simulators do not even need to be Python-based as long as they store
simulation outputs in a format that can be read from Python. All that is necessary
is to wrap your external simulator of choice into a Python callable that takes a
parameter set and outputs a set of summary statistics we want to fit the parameters to

* simulator must return summary statistics
* then in init, only simulator is needed
* change distance metric to keyword and allow for custom callable
* remove n_simulator_samples_per_parameter from sample method
'''

'''
distance = self.distance.dist_max()

if distance < self.epsilon and self.logger:
    self.logger.warn("initial epsilon {:e} is larger than dist_max {:e}"
                     .format(float(self.epsilon), distance))
'''

'''
import logging
import time
import enlighten

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Setup progress bar
manager = enlighten.get_manager()
pbar = manager.counter(total=100, desc='Ticks', unit='ticks')

for i in range(1, 101):
    logger.info("Processing step %s" % i)
    time.sleep(.2)
    pbar.update()
'''


class ABCBase(metaclass=ABCMeta):
    def __init__(
        self,
        observation,
        simulator: Callable,
        priors,
        distance: Union[str, Callable] = "l2"
    ) -> None:
        """
        simulator : callable
            simulator model
        summary_calculator : callable, defualt None
            summary statistics calculator. If None, simulator should output
            sum stat
        distance : str
            Can be a custom function or one of l1, l2, mse
        distance_metric : callable
            discrepancy measure
        """
        self._obs = observation
        self._simulator = simulator
        self._priors = priors

        # Select distance function.
        if callable(distance):
            self._distance = distance
        elif isinstance(distance, str):
            check_distance_str(distance)
            self._distance = self._choose_distance(distance)
        else:
            raise TypeError()

        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def sample(self):
        """To be overwritten by sub-class: should implement sampling from
        inference scheme and return journal

        Examples
        --------
        If the desired distance maps to :math:`\mathbb{R}`, this method should return numpy.inf.
        Returns
        -------
        pylfi.journal
            Journal
        """

        raise NotImplementedError

    @staticmethod
    def _choose_distance(distance):
        """Return distance function for given distance type."""
        if distance == 'l1':
            return None
        elif distance == 'l2':
            return euclidean
        elif distance == 'mse':
            return None

    @staticmethod
    def run_lra():
        pass


class RejectionABC(ABCBase):

    def __init__(self, observation, simulator, priors, distance='l2'):
        """
        simulator : callable
            simulator model
        summary_calculator : callable, defualt None
            summary statistics calculator. If None, simulator should output
            sum stat
        distance : str
            Can be a custom function or one of l1, l2, mse
        distance_metric : callable
            discrepancy measure
        """

        # self._obs = observation
        # self._simulator = simulator  # model simulator function
        # self._priors = priors
        # self._distance = distance    # distance metric function
        super().__init__(
            observation=observation,
            simulator=simulator,
            priors=priors,
            distance=distance,
        )

    def __call__(self, num_simulations, epsilon, lra=False):
        journal = self.sample(num_simulations, epsilon, lra)
        return journal

    def sample(self, num_simulations, epsilon, lra=False):
        """
        add **kwargs for simulator call

        Pritchard et al. (1999) algorithm

        n_samples: integer
            Number of samples to generate

        lra bool, Whether to run linear regression adjustment as in Beaumont et al. 2002
        """

        _inference_scheme = "Rejection ABC"
        self.logger.info(f"Initializing {_inference_scheme} inference scheme.")
        n_sims = num_simulations

        journal = Journal()  # journal instance
        journal._start_journal()

        journal._add_config(self._simulator, _inference_scheme,
                            self._distance, num_simulations, epsilon)
        journal._add_parameter_names(self._priors)

        # draw thetas from priors
        thetas = np.array([prior.rvs(size=(n_sims,))
                          for prior in self._priors])
        # simulated
        sims = np.array([self._simulator(*thetas)
                        for thetas in np.stack(thetas, axis=-1)])
        # distances
        distances = np.array([self._distance(self._obs, sim) for sim in sims])
        # acceptance criterion
        is_accepted = distances < epsilon

        # accepted simulations
        num_accepted = is_accepted.sum().item()
        thetas_accepted = thetas[:, is_accepted]
        dist_accepted = distances[is_accepted]
        sims_accepted = sims[is_accepted]

        for i, thetas in enumerate(np.stack(thetas_accepted, axis=-1)):
            journal._add_accepted_parameters(thetas)
            journal._add_distance(dist_accepted[i])
            journal._add_sumstat(sims_accepted[i])

        journal._add_sampler_summary(n_sims, num_accepted)

        self.logger.info(f"Accepted {num_accepted} of {n_sims} simulations.")

        if lra:
            # self.logger.info("Running Linear regression adjustment.")
            # journal.do
            pass

        return journal

###
###


'''
class RejectionABC:

    def __init__(self, simulator, summary_calculator, distance_metric):
        """
        simulator : callable
            simulator model
        summary_calculator : callable, defualt None
            summary statistics calculator. If None, simulator should output
            sum stat
        distance_metric : callable
            discrepancy measure
        """

        self._simulator = simulator              # model simulator function
        self._summary_calc = summary_calculator  # summary calculator function
        self._distance_metric = distance_metric  # distance metric function

    def sample(self, observed_data, priors, n_posterior_samples, n_simulator_samples_per_parameter, epsilon):
        """
        add **kwargs for simulator call

        Pritchard et al. (1999) algorithm

        n_samples: integer
            Number of samples to generate
        """

        _inference_scheme = "Rejection ABC"
        N = n_simulator_samples_per_parameter

        obs_sumstat = self._summary_calc(
            observed_data)  # observed data summary statistic

        journal = Journal()  # journal instance
        journal._start_journal()

        journal._add_config(self._simulator, self._summary_calc, self._distance_metric,
                            _inference_scheme, n_posterior_samples, n_simulator_samples_per_parameter, epsilon)
        journal._add_parameter_names(priors)

        number_of_simulations = 0
        accepted_count = 0

        while accepted_count < n_posterior_samples:
            number_of_simulations += 1
            # draw thetas from priors
            thetas = [theta.rvs() for theta in priors]
            # simulated data given realizations of drawn thetas
            sim_data = self._simulator(*thetas, N)
            # summary stat of simulated data
            sim_sumstat = self._summary_calc(sim_data)
            # calculate distance
            distance = self._distance_metric(sim_sumstat, obs_sumstat)

            if distance <= epsilon:
                accepted_count += 1
                journal._add_accepted_parameters(thetas)
                journal._add_distance(distance)
                journal._add_sumstat(sim_sumstat)

        journal._add_sampler_summary(number_of_simulations, accepted_count)

        return journal


class MCMCABC:

    def __init__(self, simulator, summary_calculator, distance_metric):

        self._simulator = simulator              # model simulator function
        self._summary_calc = summary_calculator  # summary calculator function
        self._distance_metric = distance_metric  # distance metric function

    def sample(self):
        pass


class SMCABC:

    def __init__(self, simulator, summary_calculator, distance_metric):

        self._simulator = simulator              # model simulator function
        self._summary_calc = summary_calculator  # summary calculator function
        self._distance_metric = distance_metric  # distance metric function

    def sample(self):
        pass


class PMCABC:

    def __init__(self, simulator, summary_calculator, distance_metric):

        self._simulator = simulator              # model simulator function
        self._summary_calc = summary_calculator  # summary calculator function
        self._distance_metric = distance_metric  # distance metric function

    def sample(self):
        pass
'''

if __name__ == "__main__":
    import scipy.stats as stats
    from pylfi.distances import euclidean
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

    # simulator
    def gaussian_model(mu, sigma, n_samples=150):
        sim = stats.norm(loc=mu, scale=sigma).rvs(size=n_samples)
        return np.mean(sim)

    # priors
    mu = Normal(165, 20, name="mu", tex="$\mu$")
    sigma = Uniform(5, 30, name="sigma", tex="$\sigma$")
    priors = [mu, sigma]

    # initialize sampler
    sampler = RejectionABC(obs, gaussian_model, priors, euclidean)

    # inference config
    num_simulations = 1000
    epsilon = 0.5

    # run inference
    journal = sampler.sample(num_simulations, epsilon)

    # journal
    samples_mu = journal.get_accepted_parameters["mu"]
    samples_sigma = journal.get_accepted_parameters["sigma"]
    # print(journal.get_number_of_accepted_simulations)
