#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import numpy as np
from pylfi.inferences import ABCBase
from pylfi.journal import Journal
from pylfi.utils import setup_logger
from tqdm import tqdm


class RejectionABC(ABCBase):
    """Rejection ABC.
    """

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

    def sample(self, num_simulations, epsilon, lra=False, log=True):
        """
        add **kwargs for simulator call

        Pritchard et al. (1999) algorithm

        n_samples: integer
            Number of samples to generate

        lra bool, Whether to run linear regression adjustment as in Beaumont et al. 2002
        """

        _inference_scheme = "Rejection ABC"

        if log:
            self.logger = setup_logger(self.__class__.__name__)
            self.logger.info(
                f"Initializing {_inference_scheme} inference scheme.")

        n_sims = num_simulations

        # initialize journal
        journal = Journal()
        journal._start_journal(log, self._simulator, self._priors,
                               _inference_scheme, self._distance, num_simulations, epsilon)

        '''
        journal._add_config(self._simulator, _inference_scheme,
                            self._distance, num_simulations, epsilon)
        journal._add_parameter_names(self._priors)
        '''
        # draw thetas from priors
        thetas = np.array([prior.rvs(size=(n_sims,))
                           for prior in self._priors])

        # run simulator
        if log:
            self.logger.info(f"Running simulator with prior samples.")
            sims = np.zeros(n_sims)
            for i, theta in enumerate(tqdm(np.stack(thetas, axis=-1),
                                           desc="Simulation progress",
                                           position=0,
                                           leave=True,
                                           colour='green')):
                sims[i] = self._simulator(*theta)
        else:
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
            journal._add_raw_distance(sims_accepted[i] - self._obs)
            journal._add_sumstat(sims_accepted[i])

        #journal._sampler_summary(n_sims, num_accepted)
        if log:
            self.logger.info("Sampling results written to journal.")
            self.logger.info(
                f"Accepted {num_accepted} of {n_sims} simulations.")

        # if num_accepted < ... : raise RuntimeError eller custom InferenceError
        journal._process_inference(n_sims, num_accepted, lra)

        return journal
