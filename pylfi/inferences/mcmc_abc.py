#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import multiprocessing
from multiprocessing import Lock, RLock

import numpy as np
import scipy.stats as stats
from pathos.pools import ProcessPool
from pylfi.inferences import ABCBase, PilotStudyMissing, SamplingNotPerformed
from pylfi.journal import Journal
from pylfi.utils import advance_PRNG_state, generate_seed_sequence
from tqdm.auto import tqdm


class MCMCABC(ABCBase):
    """Class implementing the MCMC ABC algorithm.
    """

    def __init__(
        self,
        observation,
        simulator,
        stat_calc,
        priors,
        log=True
    ):
        super().__init__(
            observation=observation,
            simulator=simulator,
            stat_calc=stat_calc,
            priors=priors,
            inference_scheme="MCMC ABC",
            log=log
        )

    def tune(
            self,
            prop_scale=0.5,
            epsilon=None,
            tune_iter=500,
            tune_interval=100,
            stat_weight=1.,
            stat_scale=1.,
            seed=None,
            use_pilot=False
    ):
        """Tune the proposal scale

        So how do we choose sd for the proposal distribution? There are some
        papers that suggest Metropolis-Hastings is most efficient when you accept
        23.4% of proposed samples, and it turns out that lowering step size
        increases the probability of accepting a proposal. PyMC3 will spend the
        first 500 steps increasing and decreasing the step size to try to find
        the best value of sd that will give you an acceptance rate of 23.4%
        (you can even set different acceptance rates).

        The problem is that if you change the step size while sampling, you lose
        the guarantees that your samples (asymptotically) come from the target
        distribution, so you should typically discard these. Also, there is
        typically a lot more adaptation going on in those first steps than just
        step_size.
        """

        if self._log:
            self.logger.info("Run MCMC tuner.")

        if use_pilot:
            if not self._done_pilot_study:
                msg = ("In order to use tuning from pilot study, the "
                       "pilot_study method must be run in advance.")
                raise PilotStudyMissing(msg)
        else:
            if epsilon is None:
                msg = ("epsilon must be passed.")
                raise ValueError(msg)
            self._epsilon = epsilon
            self._stat_scale = stat_scale

        self._n_samples = n_samples
        self._burn = burn
        self._stat_weight = stat_weight

        self._prop_scale = prop_scale

        if self._log:
            t_range = tqdm(range(n_iter),
                           desc=f"[Sampling progress] Chain {position+1}",
                           position=position,
                           leave=False,
                           colour='green')
        else:
            t_range = range(n_iter)

        seeds = generate_seed_sequence(seed, n_jobs)

        n_accepted = 0

        # Initialize chain
        thetas_current, _, _ = self._draw_initial_sample(seed)

        # Compute current logpdf
        # (only needs to be re-computed if proposal is accepted)
        logpdf_current = self._compute_logpdf(thetas_current)

        # Metropolis algorithm
        for i in t_range:
            # Advance PRNG state
            next_gen = advance_PRNG_state(seed, i)
            # Draw proposal
            thetas_proposal = self._draw_proposal(thetas_current, next_gen)
            # Compute proposal logpdf
            logpdf_proposal = self._compute_logpdf(thetas_proposal)
            # Compute acceptance probability
            alpha = self._acceptance_prob(logpdf_proposal, logpdf_current)
            # Draw a uniform random number
            u = self._draw_uniform(next_gen)
            # Metropolis reject/accept step
            if u < alpha:
                # Simulator call to generate simulated data
                sim = self._simulator(*thetas_proposal)
                # Calculate summary statistics of simulated data
                if isinstance(sim, tuple):
                    sim_sumstat = self._stat_calc(*sim)
                else:
                    sim_sumstat = self._stat_calc(sim)

                # Compute distance between summary statistics
                distance = self.distance(sim_sumstat,
                                         self._obs_sumstat,
                                         weight=self._stat_weight,
                                         scale=self._stat_scale
                                         )
                # ABC reject/accept step
                if distance <= self._epsilon:
                    thetas_current = thetas_proposal
                    # Increase accepted counter
                    n_accepted += 1
                    # Re-compute current logpdf
                    logpdf_current = self._compute_logpdf(thetas_current)

            if tune_now:
                pass

        #
        self._done_tuning = True

    def sample(
        self,
        n_samples,
        epsilon=None,
        prop_scale=0.5,
        burn=100,
        tune=True,
        tune_iter=500,
        tune_interval=100,
        stat_weight=1.,
        stat_scale=1.,
        use_pilot=False,
        chains=2,
        seed=None,
        return_journal=False
    ):
        """
        tune: bool
            Flag for tuning. Defaults to True.
        tune_interval: int
            The frequency of tuning. Defaults to 100 iterations.

        Due to multiprocessing, estimation time (iteration per loop, total
        time, etc.) could be unstable, but the progress bar works perfectly.

        A good choice for the number of jobs is the number of cores or processors on your computer.
        If your processor supports hyperthreading, you can select an even higher number of jobs.
        The number of jobs is set to the number of cores found in the system by default.

        There are some papers that suggest Metropolis-Hastings is most efficient
        when you accept 23.4% of proposed samples, and it turns out that lowering
        step size increases the probability of accepting a proposal. PyMC3 will
        spend the first 500 steps increasing and decreasing the step size to try
        to find the best value of sd that will give you an acceptance rate of
        23.4% (you can even set different acceptance rates).

        burn : either burn away or add to n_samples
        """

        if self._log:
            self.logger.info("Run MCMC sampler.")

        if use_pilot:
            if not self._done_pilot_study:
                msg = ("In order to use tuning from pilot study, the "
                       "pilot_study method must be run in advance.")
                raise PilotStudyMissing(msg)
        else:
            if epsilon is None:
                msg = ("epsilon must be passed.")
                raise ValueError(msg)
            self._epsilon = epsilon
            self._quantile = None
            self._stat_scale = stat_scale

        self._n_samples = n_samples
        self._burn = burn
        self._stat_weight = stat_weight

        # These are set in base instead
        #self._prior_logpdfs = [prior.logpdf for prior in self._priors]
        #self._rng = np.random.default_rng
        #self._uniform_distr = stats.uniform(loc=0, scale=1)

        # mcmc knobs
        self._prop_scale = prop_scale

        # force equal, n_samples
        n_samples, chains, tasks, seeds = self.batches(n_samples,
                                                       chains,
                                                       seed,
                                                       force_equal=True
                                                       )
        # n_samples + burn

        if self._log:
            # for managing output contention
            initializer = tqdm.set_lock(RLock(),)
            initargs = (tqdm.get_lock(),)
        else:
            initializer = None
            initargs = None

        with ProcessPool(chains) as pool:
            r0, r1, r2, r3 = zip(
                *pool.map(
                    self._sample,
                    tasks,
                    range(chains),
                    seeds,
                    initializer=initializer,
                    initargs=initargs
                )
            )

        #self._original_samples = np.stack(r0)
        self._original_samples = np.concatenate(r0, axis=0)
        self._samples = copy.deepcopy(self._original_samples)
        self._distances = np.concatenate(r1, axis=0)
        self._sum_stats = np.concatenate(r2, axis=0)
        self._n_accepted = np.sum(r3)

        self._done_sampling = True

        if return_journal:
            return self.journal()

    def _sample(self, n_samples, position, seed):
        """Sample n_samples from posterior."""

        n_iter = n_samples + self._burn - 1

        if self._log:
            t_range = tqdm(range(n_iter),
                           desc=f"[Sampling progress] Chain {position+1}",
                           position=position,
                           leave=False,
                           colour='green')
        else:
            t_range = range(n_iter)

        n_accepted = 0
        samples = []
        distances = []
        sum_stats = []

        # Initialize chain
        thetas_current, distance, sim_sumstat = self._draw_initial_sample(seed)
        samples.append(thetas_current)
        distances.append(distance)
        sum_stats.append(sim_sumstat)

        # Compute current logpdf
        # (only needs to be re-computed if proposal is accepted)
        logpdf_current = self._compute_logpdf(thetas_current)

        # Metropolis algorithm
        for i in t_range:
            # Advance PRNG state
            next_gen = advance_PRNG_state(seed, i)
            # Draw proposal
            thetas_proposal = self._draw_proposal(thetas_current, next_gen)
            # Compute proposal logpdf
            logpdf_proposal = self._compute_logpdf(thetas_proposal)
            # Compute acceptance probability
            alpha = self._acceptance_prob(logpdf_proposal, logpdf_current)
            # Draw a uniform random number
            u = self._draw_uniform(next_gen)
            # Metropolis reject/accept step
            if u < alpha:
                # Simulator call to generate simulated data
                sim = self._simulator(*thetas_proposal)
                # Calculate summary statistics of simulated data
                if isinstance(sim, tuple):
                    sim_sumstat = self._stat_calc(*sim)
                else:
                    sim_sumstat = self._stat_calc(sim)

                # Compute distance between summary statistics
                distance = self.distance(sim_sumstat,
                                         self._obs_sumstat,
                                         weight=self._stat_weight,
                                         scale=self._stat_scale
                                         )
                # ABC reject/accept step
                if distance <= self._epsilon:
                    thetas_current = thetas_proposal
                    # Increase accepted counter
                    n_accepted += 1
                    # Re-compute current logpdf
                    logpdf_current = self._compute_logpdf(thetas_current)

            # Update chain
            samples.append(thetas_current)
            distances.append(distance)
            sum_stats.append(sim_sumstat)

        if self._log:
            t_range.clear()

        # Remove burn-in samples
        samples = samples[self._burn:]
        distances = distances[self._burn:]
        sum_stats = sum_stats[self._burn:]

        return [samples, distances, sum_stats, n_accepted]

    def _acceptance_prob(self, logpdf_proposal, logpdf_current):
        """Compute Metropolis acceptance probability

        Since the proposal density is symmetric, the ratio of proposal
        densities in the Metropolis-Hastings algorithm cancels, and we are
        left with what is known as the Metropolis algorithm where we only
        need to evaluate the ratio of the prior densities.
        """
        # Compute Metropolis ratio
        r = np.exp(logpdf_proposal - logpdf_current)
        # Compute acceptance probability
        alpha = np.minimum(1., r)
        return alpha

    def _draw_initial_sample(self, seed):
        """Draw first posterior sample from prior via Rejection ABC algorithm."""
        sample = None
        n_sims = 0

        while sample is None:
            # Advance PRNG state
            next_gen = advance_PRNG_state(seed, n_sims)
            # Draw proposal parameters from priors
            thetas = [prior.rvs(seed=next_gen) for prior in self._priors]
            # Simulator call to generate simulated data
            sim = self._simulator(*thetas)
            # Calculate summary statistics of simulated data
            if isinstance(sim, tuple):
                sim_sumstat = self._stat_calc(*sim)
            else:
                sim_sumstat = self._stat_calc(sim)
            # Increase simulations counter
            n_sims += 1
            # Compute distance between summary statistics
            distance = self.distance(sim_sumstat,
                                     self._obs_sumstat,
                                     weight=self._stat_weight,
                                     scale=self._stat_scale
                                     )
            # ABC reject/accept step
            if distance <= self._epsilon:
                sample = thetas

        return sample, distance, sim_sumstat

    def _draw_proposal(self, thetas_current, next_gen):
        """Suggest new positions"""
        # Gaussian proposal distribution (which is symmetric)
        proposal_distr = stats.norm(
            loc=thetas_current,
            scale=self._prop_scale,
        )

        # Draw proposal parameters
        thetas_proposal = proposal_distr.rvs(
            random_state=self._rng(seed=next_gen)
        ).tolist()

        if not isinstance(thetas_proposal, list):
            thetas_proposal = [thetas_proposal]

        return thetas_proposal

    def _compute_logpdf(self, thetas):
        """
        Compute the joint prior log density for thetas.

        In case of multiple parameters, the joint prior logpdf is computed

        Note that where the proposal log density needs to be computed for each
        new proposal, the current log density only needs to be (re-)computed
        if a proposal is accepted.
        """
        '''
        logpdf = np.array(
            [prior_logpdf(theta)] for prior_logpdf, theta in
            zip(self._prior_logpdfs, thetas)
        ).prod()
        '''

        logpdf = np.array([prior_logpdf(theta)
                           for prior_logpdf, theta in
                           zip(self._prior_logpdfs, thetas)]
                          ).prod()

        return logpdf

    def _draw_uniform(self, next_gen):
        """Draw a uniform random number"""
        return self._uniform_distr.rvs(random_state=self._rng(seed=next_gen))

    def _tune_scale_table(self):
        """Proposal scale lookup table.

        Function retrieved from PyMC3 source code.

        Tunes the scaling parameter for the proposal distribution
        according to the acceptance rate over the last tune_interval:

        Rate    Variance adaptation
        ----    -------------------
        <0.001        x 0.1
        <0.05         x 0.5
        <0.2          x 0.9
        >0.5          x 1.1
        >0.75         x 2
        >0.95         x 10
        """
        if acc_rate < 0.001:
            # reduce by 90 percent
            return scale * 0.1
        elif acc_rate < 0.05:
            # reduce by 50 percent
            return scale * 0.5
        elif acc_rate < 0.2:
            # reduce by ten percent
            return scale * 0.9
        elif acc_rate > 0.95:
            # increase by factor of ten
            return scale * 10.0
        elif acc_rate > 0.75:
            # increase by double
            return scale * 2.0
        elif acc_rate > 0.5:
            # increase by ten percent
            return scale * 1.1

        return scale

    @property
    def prop_scale(self):
        try:
            return self._prop_scale
        except AttributeError:
            msg = ("stat_scale inaccessible. A call to a method where the"
                   "attribute is set must be carried out first.")
            raise MissingParameter(msg)

    def journal(self):
        """
        Create and return an instance of Journal class.

        Returns
        -------

        """
        if not self._done_sampling:
            msg = ("In order to access the journal, the "
                   "sample method must be run in advance.")
            raise SamplingNotPerformed(msg)

        if self._log:
            self.logger.info(f"Write results to journal.")

        accept_ratio = self._n_accepted / self._n_samples
        print(f"{accept_ratio=}")
        journal = Journal()
        journal._write_to_journal(
            inference_scheme=self._inference_scheme,
            observation=self._obs_data,
            simulator=self._simulator,
            stat_calc=self._stat_calc,
            priors=self._priors,
            n_samples=self._n_samples,
            n_chains=1,
            n_sims=1,
            samples=self._samples,
            accept_ratio=accept_ratio,
            epsilon=self._epsilon,
            quantile=self._quantile
        )

        return journal


if __name__ == "__main__":
    import arviz as az
    import matplotlib.pyplot as plt
    import pylfi
    import scipy.stats as stats
    import seaborn as sns

    N = 1000
    mu_true = 163
    sigma_true = 15
    true_parameter_values = [mu_true, sigma_true]
    # likelihood = stats.norm(loc=mu_true, scale=sigma_true)
    likelihood = pylfi.Prior('norm',
                             loc=mu_true,
                             scale=sigma_true,
                             name='likelihood'
                             )

    obs_data = likelihood.rvs(size=N, seed=30)

    # simulator model
    def gaussian_model(mu, sigma, seed=43, n_samples=1000):
        """Simulator model"""
        # sim = stats.norm(loc=mu, scale=sigma).rvs(size=n_samples)
        model = pylfi.Prior('norm', loc=mu, scale=sigma, name='model')
        sim = model.rvs(size=n_samples, seed=seed)

        return sim

    # summary stats
    def summary_calculator(data):
        """returns summary statistic(s)"""
        sumstat = np.array([np.mean(data), np.std(data)])
        # sumstat = np.mean(sim)
        return sumstat

    s_obs = summary_calculator(obs_data)
    # print(f"{s_obs=}")
    # priors
    mu = pylfi.Prior('norm', loc=165, scale=2, name='mu', tex=r'$\mu$')
    sigma = pylfi.Prior('norm', loc=17, scale=4,
                        name='sigma', tex=r'$\sigma$')
    # mu = pylfi.Prior('uniform', loc=160, scale=10, name='mu')
    # sigma = pylfi.Prior('uniform', loc=10, scale=10, name='sigma')
    priors = [mu, sigma]

    # initialize sampler
    sampler = MCMCABC(obs_data,
                      gaussian_model,
                      summary_calculator,
                      priors,
                      log=True
                      )

    sampler.pilot_study(3000,
                        quantile=0.1,
                        stat_scale="mad",
                        n_jobs=4,
                        seed=4
                        )

    journal = sampler.sample(4000,
                             use_pilot=True,
                             chains=4,
                             burn=1000,
                             seed=42,
                             return_journal=True
                             )

    df = journal.df
    print(df["mu"].mean())
    print(df["sigma"].mean())
    sns.jointplot(
        data=df,
        x="mu",
        y="sigma",
        kind="kde",
        fill=True
    )

    journal = sampler.reg_adjust(
        method='loclinear',
        transform=True,
        kernel='epkov',
        return_journal=True
    )

    df = journal.df
    print(df["mu"].mean())
    print(df["sigma"].mean())
    sns.jointplot(
        data=df,
        x="mu",
        y="sigma",
        kind="kde",
        fill=True
    )
    plt.show()
