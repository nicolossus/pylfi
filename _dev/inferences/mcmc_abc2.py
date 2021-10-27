#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pylfi
import scipy.stats as stats
from pylfi.inferences import ABCBase
from pylfi.utils import (advance_PRNG_state, check_and_set_jobs,
                         distribute_workload, generate_seed_sequence,
                         setup_logger)


class MCMCABC(ABCBase):

    def __init__(
        self,
        observation,
        simulator,
        stat_calc,
        priors,
        distance='l2',
        seed=None
    ):

        super().__init__(
            observation=observation,
            simulator=simulator,
            statistics_calculator=stat_calc,
            priors=priors,
            distance_metric=distance,
            seed=seed
        )

    def sample(
        self,
        n_samples,
        epsilon=None,
        scaling=0.5,
        burn=100,
        tune=True,
        n_tune=500,
        tune_interval=50,
        n_jobs=-1,
        log=False
    ):
        _inference_scheme = "MCMC-ABC"
        self._epsilon = epsilon
        self._sigma = scaling
        self._rng = np.random.default_rng
        self._prior_logpdfs = [prior.logpdf for prior in self._priors]

        self._uniform_distr = stats.uniform(loc=0, scale=1)

        samples = self._sample(n_samples, self._seed)

        return samples

    def _sample(self, n_samples, seed):
        """Sample n_samples from posterior."""

        self._n_sims = 0
        self._n_iter = 0
        samples = []

        # initialize chain
        thetas_current = self._draw_initial_posterior_sample(seed)
        samples.append(thetas_current)

        # Pre-loop computations to better efficiency
        # create instance before loop to avoid some overhead
        unif_distr = pylfi.Prior('uniform', loc=0, scale=1, name='u')

        # Only needs to be re-computed if proposal is accepted
        log_prior_current = np.array([prior_logpdf(theta_current)
                                      for prior_logpdf, theta_current in
                                      zip(self._prior_logpdfs, thetas_current)]
                                     ).prod()

        # Metropolis-Hastings algorithm
        for _ in range(n_samples):
            # Advance PRNG state
            next_gen = advance_PRNG_state(seed, self._n_iter)

            # Gaussian proposal distribution (which is symmetric)
            proposal_distr = stats.norm(
                loc=thetas_current,
                scale=self._sigma,
            )

            # Draw proposal parameters (suggest new positions)
            thetas_proposal = [proposal_distr.rvs(
                random_state=self._rng(seed=next_gen))]

            # Compute Metropolis-Hastings ratio.
            # Since the proposal density is symmetric, the proposal density
            # ratio in MH acceptance probability cancel. Thus, we need only
            # to evaluate the prior ratio. In case of multiple parameters,
            # the joint prior logpdf is computed.
            log_prior_proposal = np.array([prior_logpdf(thetas_proposal)
                                           for prior_logpdf, thetas_proposal in
                                           zip(self._prior_logpdfs, thetas_proposal)]
                                          ).prod()

            r = np.exp(log_prior_proposal - log_prior_current)

            # Compute acceptance probability
            alpha = np.minimum(1., r)

            # Draw a uniform random number
            u = unif_distr.rvs(seed=next_gen)

            # Reject/accept step
            if u < alpha:
                sim = self._simulator(*thetas_proposal)
                sim_sumstat = self._stat_calc(sim)
                distance = self._distance_metric(
                    self._obs_sumstat, sim_sumstat)
                if distance <= self._epsilon:
                    thetas_current = thetas_proposal

                    # Re-compute current log-density for next iteration
                    log_prior_current = np.array([prior_logpdf(theta_current)
                                                  for prior_logpdf, theta_current in
                                                  zip(self._prior_logpdfs, thetas_current)]
                                                 ).prod()

            self._n_iter += 1
            # Update chain
            samples.append(thetas_current)

        return samples

    def _draw_initial_posterior_sample(self, seed):
        """Draw first posterior sample from prior via Rejection ABC algorithm."""
        sample = None
        n_sims = 0
        while sample is None:
            next_gen = advance_PRNG_state(seed, n_sims)
            thetas = [prior.rvs(seed=next_gen) for prior in self._priors]
            sim = self._simulator(*thetas)
            sim_sumstat = self._stat_calc(sim)
            n_sims += 1
            distance = self._distance_metric(self._obs_sumstat, sim_sumstat)
            if distance <= self._epsilon:
                sample = thetas

        return sample

    def _draw_proposal(self, scaling, next_gen):
        """Suggest new position(s)"""
        # Gaussian proposal distribution (which is symmetric)
        proposal_distr = stats.norm(
            loc=thetas_current,
            scale=scaling,
        )

        # Draw proposal parameters
        thetas_proposal = [proposal_distr.rvs(
            random_state=self._rng(seed=next_gen))]

        # In case of multiple parameters, the joint prior logpdf is computed
        log_prior_proposal = np.array([prior_logpdf(thetas_proposal)
                                       for prior_logpdf, thetas_proposal in
                                       zip(self._prior_logpdfs, thetas_proposal)]
                                      ).prod()

        return thetas_proposal, log_prior_proposal

    def _mh_step(self, log_prior_proposal, log_prior_current, next_gen):
        """
        Compute MH ratio and acceptance probability
        """
        # Since the proposal density is symmetric, the proposal density
        # ratio in MH acceptance probability cancel. Thus, we need only
        # to evaluate the prior ratio.
        r = np.exp(log_prior_proposal - log_prior_current)

        # Compute acceptance probability
        alpha = np.minimum(1., r)

        # Draw a uniform random number
        u = self._uniform_distr.rvs(random_state=self._rng(seed=next_gen))

        return u < alpha

    def _abc_step(self, thetas_proposal):
        pass

    def _mh_step(self, thetas_current, thetas_proposal, log_prior_current):
        sim = self._simulator(*thetas_proposal)
        sim_sumstat = self._stat_calc(sim)
        distance = self._distance_metric(
            self._obs_sumstat, sim_sumstat)
        # In MCMC-ABC, we also need to accept discrepancy between
        # observed and simulated summary statistics before the proposal
        # parameter(s) can be accepted
        if distance <= self._epsilon:
            thetas_current = thetas_proposal
            # Re-compute current log-density for next iteration
            log_prior_current = np.array([prior_logpdf(theta_current)
                                          for prior_logpdf, theta_current in
                                          zip(self._prior_logpdfs, thetas_current)]
                                         ).prod()

        return thetas_current, log_prior_current

    def _metropolis_hastings_step(self, initial_proposal, seed, scaling):

        # Pre-compute current (joint) prior logpdf. Only needs to be
        # re-computed in loop if a new proposal is accepted
        log_prior_current = np.array([prior_logpdf(theta_current)
                                      for prior_logpdf, theta_current in
                                      zip(self._prior_logpdfs, thetas_current)]
                                     ).prod()

        # Metropolis-Hastings algorithm
        for _ in range(n_samples):
            # Advance PRNG state
            next_gen = advance_PRNG_state(seed, self._n_iter)

            thetas_proposal, log_prior_proposal = self._draw_proposal(
                scaling, next_gen)

            acc_prob = self._accept_proposal(
                log_prior_proposal, log_prior_current, next_gen)

            # Reject/accept step
            if acc_prob:
                thetas_current, log_prior_current = self._mh_step(
                    thetas_current, thetas_proposal, log_prior_current)

        pass

    def _tune_sampler(self):
        # no need to store
        pass

    def _burn_in_sampler(self, initial_proposal):
        # no need to store
        pass

    def _metropolis_sampler(self):
        pass

    def _metropolis_hastings_sampler(self):
        pass

    '''
    def _draw_proposal(self, thetas_current, seed):
        """Suggest a new position"""

        # advance PRNG state
        next_gen = advance_PRNG_state(seed, self._n_sims)

        # Gaussian proposal distribution
        proposal_distr = stats.norm(
            loc=thetas_current,
            scale=self._sigma,
            random_state=self._rng(seed=next_gen)
        )

        # draw proposal parameters
        thetas_proposal = proposal_distr.rvs()

        return thetas_proposal

    def _metropolis_hastings_step(self):
        pass
    '''


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import seaborn as sns
    from arviz import autocorr

    # global variables
    groundtruth = 2.0  # true variance
    N = 1000           # number of observations

    # observed data
    likelihood = stats.norm(loc=0, scale=np.sqrt(groundtruth))
    obs_data = likelihood.rvs(size=N)

    def summary_calculator(data):
        return np.var(data)

    def simulator(theta, N=1000):
        """Simulator model, returns summary statistic"""
        model = stats.norm(loc=0, scale=np.sqrt(theta))
        sim = model.rvs(size=N)
        return sim

    # prior (conjugate)
    alpha = 60         # prior hyperparameter (inverse gamma distribution)
    beta = 130         # prior hyperparameter (inverse gamma distribution)
    theta = pylfi.Prior('invgamma', alpha, loc=0, scale=beta, name='theta')
    priors = [theta]

    # initialize sampler
    sampler = MCMCABC(obs_data, simulator, summary_calculator,
                      priors, distance='l2', seed=42)

    # inference config
    n_samples = 1000
    epsilon = 0.5

    # run inference
    journal = sampler.sample(n_samples,
                             epsilon=epsilon,
                             scaling=0.5,
                             n_jobs=-1,
                             log=True)

    samples = np.concatenate(journal, axis=0)
    print(samples)

    # print(autocorr(samples))

    lags = np.arange(1, 100)
    #fig, ax = plt.subplots()
    # ax.plot(autocorr(samples))

    sns.displot(samples, kind='kde')
    plt.show()

    '''
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))
    ax[0].plot(samples)
    ax[0].set(xlabel='Sample', ylabel='theta')
    ax[1].hist(samples, density=True, alpha=0.7)
    ax[1].axvline(np.mean(samples), ls='--', color='r')
    ax[1].set(xlabel='theta')
    plt.tight_layout()
    plt.show()
    '''
