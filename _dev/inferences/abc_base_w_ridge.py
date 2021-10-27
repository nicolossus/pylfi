#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from abc import abstractmethod
from multiprocessing import Lock, RLock, freeze_support

import numpy as np
from pathos.pools import ProcessPool
from pylfi.utils import (advance_PRNG_state, check_and_set_jobs,
                         distribute_workload, generate_seed_sequence,
                         setup_logger)
from tqdm.auto import tqdm


class PilotStudyMissing(Exception):
    """Failed attempt at accessing pilot study.

    A call to the pilot_study method must be carried out first.
    """
    pass


class SamplingNotPerformed(Exception):
    """Failed attempt at accessing posterior samples.

    A call to the sample method must be carried out first.
    """
    pass


VALID_STAT_SCALES = ["sd", "mad"]
VALID_KERNELS = ["gaussian", "epkov"]


class ABCBase:
    """
    ABC base class.


    """

    def __init__(
        self,
        observation,
        simulator,
        stat_calc,
        priors,
        inference_scheme,
        log=True
    ):

        self._obs_data = observation
        self._stat_calc = stat_calc
        self._simulator = simulator
        self._priors = priors
        self._log = log

        if isinstance(self._obs_data, tuple):
            self._obs_sumstat = self._stat_calc(*self._obs_data)
        else:
            self._obs_sumstat = self._stat_calc(self._obs_data)

        self._inference_scheme = inference_scheme

        if self._log:
            self.logger = setup_logger(self.__class__.__name__)
            self.logger.info(f"Initialize {self._inference_scheme} sampler.")

        self._done_pilot_study = False
        self._done_sampling = False

    @abstractmethod
    def sample(self):
        """To be overwritten by sub-class: should implement sampling from
        inference scheme and (optionally) return journal.

        The sample method needs specific attribute names. See `RejABC` class
        for an example.

        Returns
        -------
        pylfi.journal
            Journal
        """

        raise NotImplementedError

    @abstractmethod
    def journal(self):
        """To be overwritten by sub-class: method to write and return journal.

        See `RejABC` class for an example.

        Returns
        -------
        pylfi.journal
            Journal
        """

        raise NotImplementedError

    def distance(self, s_sim, s_obs, weight=1., scale=1.):
        """Weighted Euclidean distance.

        Computes the weighted Euclidean distance between two 1-D arrays
        of summary statistics. The `weight` parameter can be set to weight
        the importance of a summary statistic, whereas the `scale` parameter
        can be used to scale particular summary statistics in order to avoid
        dominance.

        Parameters
        ----------
        s_sim : {:obj:`int`, :obj:`float`}, :term:`array_like`
            Simulated summary statistic(s).
        s_obs : {:obj:`int`, :obj:`float`}, :term:`array_like`
            Observed summary statistic(s).
        weight: {:obj:`int`, :obj:`float`}, :term:`ndarray`, optional
            Importance weight(s) of summary statistic(s). Default: `1.`.
        scale: {:obj:`int`, :obj:`float`}, :term:`ndarray`, optional
            Scale weight(s) of summary statistic(s). Default: `1.`.

        Returns
        -------
        distance : :obj:`float`
            The weighted Euclidean distance between simulated and observed
            summary statistics.
        """

        if isinstance(s_sim, (int, float)):
            s_sim = [s_sim]
        if isinstance(s_obs, (int, float)):
            s_obs = [s_obs]

        s_sim = np.asarray(s_sim, dtype=float)
        s_obs = np.asarray(s_obs, dtype=float)

        q = weight * (s_sim - s_obs) / scale
        dist = np.linalg.norm(q, ord=2)

        return dist

    def batches(self, n_samples, n_jobs, seed, force_equal=False):
        """Divide and conquer.

        Divide the number of samples, `n_samples, between` `n_jobs` workers.
        If the ABC algorithm uses Markov chains, then `n_jobs` is the number
        chains. If `n_jobs` exceeds the number of available CPUs found by
        `Pathos` (this might include hardware threads), `n_jobs` is set to the
        number found by `Pathos`.

        The method also creates a seed for each worker based on the input seed.
        This ensures reproducibility if wanted. If `seed=None`, the results
        will be stochastic between each run of the sampler.

        When using Markov chains, the `force_equal` keyword must be set to
        `True`, in order to enforce equal length of chains. The `n_samples`
        that results in equal chain lengths will be found internally and
        returned. The corrected `n_samples` should be used downstream if
        needed.

        Parameters
        ----------
        n_samples : :obj:`int`
            Number of (posterior) samples to draw.
        n_jobs : :obj:`int`
            Number of processes (workers). If `n_jobs=-1`, then `n_jobs` is
            set to half of the CPUs found by `Pathos` (we assume half of the
            CPUs are hardware threads only and ignore those).
        seed : :obj:`int`
            User-provided seed.
        force_equal : :obj:`bool`, optional
            If `True`, `n_samples` will be adjusted to enforce equal number
            of tasks for each worker. Default: `False`.

        Returns
        -------
        n_samples : :obj:`int`
            Possibly corrected number of samples (posterior draws).
        n_jobs : :obj:`int`
            Possibly corrected number of processes (workers).
        tasks : :obj:`int`
            The number of tasks for each parallel pool worker.
        seeds : :obj:`list`
            Initial states for parallel pool workers.
        """

        if self._log:
            n_jobs = check_and_set_jobs(n_jobs, self.logger)
        else:
            n_jobs = check_and_set_jobs(n_jobs)

        seeds = generate_seed_sequence(seed, n_jobs)

        if self._log:
            n_samples, tasks = distribute_workload(n_samples,
                                                   n_jobs,
                                                   force_equal=force_equal,
                                                   logger=self.logger
                                                   )
        else:
            n_samples, tasks = distribute_workload(n_samples,
                                                   n_jobs,
                                                   force_equal=force_equal
                                                   )

        return n_samples, n_jobs, tasks, seeds

    def pilot_study(
        self,
        n_sim=500,
        quantile=None,
        stat_scale=None,
        stat_weight=1.,
        n_jobs=-1,
        seed=None,
    ):
        """Perform pilot study.

        The pilot study runs the simulator `n_sim` times and sets the
        threshold parameter `epsilon` automatically as the p-quantile of
        simulated distances. For instance, the 0.5-quantile (the median)
        will give a threshold that accepts roughly (because the threshold
        will be an estimate) 50% of the simulations.

        The pilot study can also be used to provide an estimate of the `scale`
        parameter, used in the weighted Euclidean distance, from the prior
        predictive distribution by passing the `stat_scale` keyword as
        `sd` or `mad`.

        Parameters
        ----------
        n_sims : :obj:`int`, optional
            Number of simulator runs.
        quantile : :obj:`int`
            Quantile of the Euclidean distances.
        stat_scale : :obj:`str`, optional
            Summary statistics scale to estimate; can be set as either `'sd'`
            (standard deviation)  or `'mad'` (median absolute deviation).
            If `None`, scale is set to `1.`. Default: `None`.
        stat_weight : {:obj:`int`, :obj:`float`}, :term:`ndarray`, optional
                Importance weights of summary statistics. Default: `1.`.
        n_jobs : :obj:`int`, optional
            Number of processes (workers). If `n_jobs=-1`, then `n_jobs` is
            set to half of the CPUs found by `Pathos` (we assume half of the
            CPUs are hardware threads only and ignore those). Default: `-1`.
        seed : :obj:`int`
            User-provided seed. Will be used to generate seed for each
            worker. Default: `None`.
        """

        if quantile is None:
            msg = ("quantile must be passed. The pilot study sets the "
                   "accept/reject threshold as the provided p-quantile of the "
                   "distances.")
            raise ValueError(msg)

        if not 0 < quantile <= 1.0:
            msg = ("quantile must be a value in (0, 1].")
            raise ValueError(msg)

        if stat_scale is not None:
            if stat_scale not in VALID_STAT_SCALES:
                msg = ("scale can be set as either sd (standard deviation) or "
                       "mad (median absolute deviation). If None, it defaults "
                       "to 1.")
                raise ValueError(msg)

        if self._log:
            msg = f"Run pilot study to estimate:\n"
            msg += f"* epsilon as the {quantile}-quantile of the distances"

            if stat_scale is not None:
                msg += f"\n* summary statistics scale ({stat_scale.upper()}) "
                msg += f"from the prior predictive distribution"

            self.logger.info(msg)

        self._quantile = quantile
        _, n_jobs, tasks, seeds = self.batches(n_sim, n_jobs, seed)

        if self._log:
            tqdm.set_lock(RLock())  # for managing output contention
            # freeze_support()  # for Windows support
            initializer = tqdm.set_lock
            initargs = (Lock(),)

            # initializer=tqdm.set_lock, initargs=(Lock(),))
        else:
            initializer = None
            initargs = None

        with ProcessPool(n_jobs) as pool:

            results = pool.map(self._pilot_study,
                               tasks,
                               range(n_jobs),
                               seeds,
                               initializer=initializer,
                               initargs=initargs
                               )

        sum_stats = np.concatenate(results, axis=0)

        if stat_scale is None:
            self._stat_scale = 1.
        elif stat_scale == "sd":
            self._stat_scale = sum_stats.std(axis=0)
        elif stat_scale == "mad":
            self._stat_scale = np.median(np.absolute(
                sum_stats - np.median(sum_stats, axis=0)), axis=0)
        else:
            msg = ("scale can be set as either sd (standard deviation) or "
                   "mad (median absolute deviation). If None, defaults to 1.")
            raise ValueError(msg)

        distances = []
        for sum_stat in sum_stats:
            distance = self.distance(sum_stat,
                                     self._obs_sumstat,
                                     weight=stat_weight,
                                     scale=self._stat_scale
                                     )
            distances.append(distance)

        self._epsilon = np.quantile(np.array(distances), self._quantile)
        self._done_pilot_study = True

        if self._log:
            self.logger.info(f"epsilon = {self._epsilon}")
            self.logger.info(f"{stat_scale} = {self._stat_scale}")

    def _pilot_study(self, n_sims, position, seed):
        """Pilot study simulator calls and calculations."""

        if self._log:
            t_range = tqdm(range(n_sims),
                           desc=f"[Simulation progress] CPU {position+1}",
                           position=position,
                           leave=True,
                           colour='green')
        else:
            t_range = range(n_sims)

        sum_stats = []

        for i in t_range:
            next_gen = advance_PRNG_state(seed, i)
            thetas = [prior.rvs(seed=next_gen) for prior in self._priors]
            sim = self._simulator(*thetas)
            if isinstance(sim, tuple):
                sim_sumstat = self._stat_calc(*sim)
            else:
                sim_sumstat = self._stat_calc(sim)
            sum_stats.append(sim_sumstat)

        if self._log:
            t_range.clear()

        return sum_stats

    def reg_adjust(
        self,
        method="loclinear",
        transform=True,
        kernel='epkov',
        lmbda=1.,
        return_journal=False
    ):
        """Post-sampling regression adjustment.

        Regresses summary statistics on the obtained posterior samples, and
        corrects the posterior samples for the trend in the relationship.

        Implementation based on Beaumont et al. (2002) and Blum (2017).

        References
        ----------
        M. Beaumont, W. Zhang and D.Balding.
        "Approximate Bayesian Computation in Population Genetics"
        GENETICS December 1, 2002 vol. 162 no. 4 2025-2035

        M. Blum.
        "Regression approaches for approximate bayesian computation".
        arXiv preprint arXiv:1707.01254, 2017 - arxiv.org

        Parameters
        ----------
        method : :obj:`str`, optional
            The regression method to use:
                * `linear`: ordinary least squares regression;
                * `loclinar`: local linear regression (default);
                * `ridge`: Ridge regression;
                * `locridge`: local Ridge regression.
        kernel : :obj:`str`, optional
            The smoothing kernel function to use in local regression:
                * `gaussian`: Gaussian smoothing kernel;
                * `epkov`: Epanechnikov smoothing kernel.
        lmbda : :obj:`float`, optional
            Regularization strength in Ridge regression. Must be a positive
            float. Regularization improves the conditioning of the problem and
            reduces the variance of the estimates. Larger values specify
            stronger regularization.
        return_journal : :obj:`bool`, optional
            If `True`, journal is returned. Default: `False`.
        """

        if not self._done_sampling:
            msg = ("In order to perform regression adjustment, the "
                   "sample method must be run in advance.")
            raise SamplingNotPerformed(msg)

        if kernel not in VALID_KERNELS:
            msg = ("kernel must be passed as either 'gaussian' or 'epkov' "
                   "(Epanechnikov).")
            raise ValueError(msg)

        if not lmbda >= 0:
            msg = ("Regularization strength must be a positive float.")
            raise ValueError(msg)

        self._kernel = kernel
        self._lmbda = lmbda
        self._transform = transform

        if self._log:
            self.logger.info(f"Perform {method} regression adjustment.")

        # data (design matrix)
        X = copy.deepcopy(self._sum_stats)
        X /= self._stat_scale

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # augment with column of ones
        X = np.c_[np.ones(X.shape[0]), X]

        # target
        thetas = copy.deepcopy(self._original_samples)

        if self._transform:
            thetas = np.log(thetas)

        if method == "linear":
            self._lra(X, thetas)
        elif method == "loclinear":
            self._loclra(X, thetas)
        elif method == "ridge":
            self._ridge(X, thetas)
        elif method == "locridge":
            self._locridge(X, thetas)
        else:
            raise ValueError("Unrecognized regression method.")

        if return_journal:
            return self.journal()

    def _gaussian_kernel(self, d, h):
        """Gaussian smoothing kernel function"""
        return 1 / (h * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (d * d) / (h * h))

    def _epkov_kernel(self, d, h):
        """Epanechnikov smoothing kernel function"""
        return 0.75 / h * (1.0 - (d * d) / (h * h)) * (d < h)

    def _lra(self, X, y):
        """Linear regression adjustment"""

        # Compute coefficients (pinv = pseudo-inverse)
        coef = np.linalg.pinv(X.T @ X)  @ X.T @ y
        alpha = coef[0]
        beta = coef[1:]

        # Adjust posterior samples
        correction = ((self._sum_stats / self._stat_scale) -
                      (self._obs_sumstat / self._stat_scale)) @ beta

        if self._transform:
            self._samples = np.exp(np.log(self._original_samples) - correction)
        else:
            self._samples = self._original_samples - correction

    def _loclra(self, X, y):
        """Local linear regression adjustment"""

        # Compute weights
        if self._kernel == "gaussian":
            weights = np.array([self._gaussian_kernel(d, self._epsilon)
                                for d in self._distances])
        elif self._kernel == "epkov":
            weights = np.array([self._epkov_kernel(d, self._epsilon)
                                for d in self._distances])
        else:
            raise ValueError(f'Unrecognized kernel')

        # Weight matrix
        W = np.diag(weights)

        # Compute coefficients (pinv = pseudo-inverse)
        coef = np.linalg.pinv(X.T @ (W @ X)) @ X.T @ (W @ y)
        alpha = coef[0]
        beta = coef[1:]

        # Adjust posterior samples
        correction = ((self._sum_stats / self._stat_scale) -
                      (self._obs_sumstat / self._stat_scale)) @ beta

        if self._transform:
            self._samples = np.exp(np.log(self._original_samples) - correction)
        else:
            self._samples = self._original_samples - correction

    def _ridge(self, X, y):
        """Ridge regression adjustment"""

        # Compute coefficients (pinv = pseudo-inverse)
        xTx = X.T @ X
        lmb_eye = self._lmbda * np.identity(xTx.shape[0])
        coef = np.linalg.pinv(xTx + lmb_eye) @ X.T @ y
        alpha = coef[0]
        beta = coef[1:]

        # Adjust posterior samples
        correction = ((self._sum_stats / self._stat_scale) -
                      (self._obs_sumstat / self._stat_scale)) @ beta

        if self._transform:
            self._samples = np.exp(np.log(self._original_samples) - correction)
        else:
            self._samples = self._original_samples - correction

    def _locridge(self, X, y):
        """Local Ridge regression adjustment"""

        # Compute weights
        if self._kernel == "gaussian":
            weights = np.array([self._gaussian_kernel(d, self._epsilon)
                                for d in self._distances])
        elif self._kernel == "epkov":
            weights = np.array([self._epkov_kernel(d, self._epsilon)
                                for d in self._distances])
        else:
            raise ValueError(f'Unrecognized kernel')

        # Weight matrix
        W = np.diag(weights)

        # Compute coefficients (pinv = pseudo-inverse)
        xTWx = X.T @ (W @ X)
        lmb_eye = self._lmbda * np.identity(xTWx.shape[0])
        coef = np.linalg.pinv(xTWx + lmb_eye) @ X.T @ (W @ y)
        alpha = coef[0]
        beta = coef[1:]

        # Adjust posterior samples
        correction = ((self._sum_stats / self._stat_scale) -
                      (self._obs_sumstat / self._stat_scale)) @ beta

        if self._transform:
            self._samples = np.exp(np.log(self._original_samples) - correction)
        else:
            self._samples = self._original_samples - correction
