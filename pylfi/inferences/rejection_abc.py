#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import time

import numpy as np
from pylfi.inferences import ABCBase
from pylfi.journal import Journal
from pylfi.utils import setup_logger
from tqdm.auto import tqdm


class RejectionABC(ABCBase):
    """Rejection ABC.
    """

    def __init__(self, observation, simulator, priors, distance='l2',
                 rng=np.random.RandomState, seed=None):
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
            rng=rng,
            seed=seed
        )

    def __call__(self, num_simulations, epsilon, lra=False):
        journal = self.sample(num_simulations, epsilon, lra)
        return journal

    def sample(self, n_sims=None, n_samples=None, epsilon=0.5, log=True):
        """
        add **kwargs for simulator call?

        Pritchard et al. (1999) algorithm

        n_samples: integer
            Number of samples to generate

        epsilon : {float, str}
            Default 'adaptive'

        Notes
        -----
        Specifying the 'n_simulations' is generally a faster computation than
        specifying 'n_samples', but with the trade-off that the number of
        posterior samples will be at the mercy of the configuration

        lra bool, Whether to run linear regression adjustment as in Beaumont et al. 2002
        """

        self._t0 = time.time()

        _inference_scheme = "Rejection ABC"

        self._log = log
        self._epsilon = epsilon

        if self._log:
            self.logger = setup_logger(self.__class__.__name__)
            self.logger.info(f"Initialize {_inference_scheme} sampler.")

        if n_sims is None and n_samples is None:
            msg = ("One of 'n_sims' or 'n_samples' must be specified.")
            raise ValueError(msg)
        if n_sims is not None and n_samples is not None:
            msg = ("Cannot specify both 'n_sims' and 'n_samples'.")
            raise ValueError(msg)

        # initialize journal
        self._journal = Journal()
        self._journal._start_journal(log, self._simulator, self._priors,
                                     _inference_scheme, self._distance, n_sims, epsilon)

        if n_sims is not None:
            if isinstance(n_sims, int):
                # call rejection loop
                self._sampler_n_sims(n_sims)
            else:
                msg = ("The number of simulations must be given as an integer.")
                raise TypeError(msg)

        if n_samples is not None:
            if isinstance(n_samples, int):
                # call rejection loop
                self._sampler_n_samples(n_samples)
            else:
                msg = ("The number of samples must be given as an integer.")
                raise TypeError(msg)

        return self._journal

    def _sampler_n_sims(self, n_sims):
        """Sampling loop for specified number of simulations"""

        # draw thetas from priors
        thetas = np.array([prior.rvs(
            size=(n_sims,), rng=self._rng, seed=self._seed) for prior in self._priors])

        # run simulator
        if self._log:
            self.logger.info(f"Run simulator with prior samples.")
            sims = []
            for i, theta in enumerate(tqdm(np.stack(thetas, axis=-1),
                                           desc="Simulation progress",
                                           position=0,
                                           leave=True,
                                           colour='green')):
                sim = self._simulator(*theta)
                sims.append(sim)
            sims = np.array(sims)
        else:
            sims = np.array([self._simulator(*thetas)
                             for thetas in np.stack(thetas, axis=-1)])

        # calculate distances
        distances = np.array([self._distance(self._obs, sim) for sim in sims])

        # acceptance criterion
        is_accepted = distances <= self._epsilon

        # accepted simulations
        n_accepted = is_accepted.sum().item()
        thetas_accepted = thetas[:, is_accepted]
        dist_accepted = distances[is_accepted]
        sims_accepted = sims[is_accepted]

        if self._log:
            self.logger.info(f"Accepted {n_accepted} of {n_sims} simulations.")
            self._journal._processing_msg()

        for i, thetas in enumerate(np.stack(thetas_accepted, axis=-1)):
            self._journal._add_accepted_parameters(thetas)
            self._journal._add_distance(dist_accepted[i])
            self._journal._add_rel_distance(sims_accepted[i] - self._obs)
            self._journal._add_threshold(self._epsilon)
            self._journal._add_sumstats(sims_accepted[i])

        '''Rework this'''
        # if num_accepted < ... : raise RuntimeError eller custom InferenceError
        t1 = time.time() - self._t0
        self._journal._process_inference(n_sims, n_accepted, t1)

        if self._log:
            self._journal._done_msg()

    def _sampler_n_samples(self, n_samples):
        """Sampling loop for specified number of posterior samples"""

        n_sims = 0
        n_accepted = 0

        if self._log:
            self.logger.info("Run sampler.")
            pbar = tqdm(total=n_samples,
                        desc="Sampling progress",
                        position=0,
                        leave=True,
                        colour='green')

        while n_accepted < n_samples:
            if self._seed is None:
                thetas = [prior.rvs(rng=self._rng) for prior in self._priors]
            else:
                thetas = [prior.rvs(rng=self._rng, seed=self._seed + n_sims)
                          for prior in self._priors]
            sim = self._simulator(*thetas)
            n_sims += 1
            distance = self._distance(self._obs, sim)
            if distance <= self._epsilon:
                if self._log:
                    pbar.update(1)
                n_accepted += 1
                self._journal._add_accepted_parameters(thetas)
                self._journal._add_distance(distance)
                self._journal._add_rel_distance(sim - self._obs)
                self._journal._add_threshold(self._epsilon)
                self._journal._add_sumstats(sim)

        if self._log:
            pbar.close()
            self.logger.info(f"Sampler ran {n_sims} simulations to "
                             + f"obtain {n_accepted} samples.")
            self._journal._processing_msg()

        t1 = time.time() - self._t0
        self._journal._process_inference(n_sims, n_accepted, t1)

        if self._log:
            self._journal._done_msg()
