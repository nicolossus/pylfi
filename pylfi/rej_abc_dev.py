import numpy as np

"""
Pathos, a python parallel processing library from caltech.

Unlike Python's default multiprocessing library, pathos provides a more flexible
parallel map which can apply almost any type of function --- including lambda
functions, nested functions, and class methods --- and can easily handle
functions with multiple arguments.

In author’s own words:

"Pathos is a framework for heterogenous computing.
It primarily provides the communication mechanisms for configuring
and launching parallel computations across heterogenous resources"

multiprocess.Pool is a fork of multiprocessing.Pool, with the only difference
being that multiprocessing uses pickle and multiprocess uses dill

The preferred interface is pathos.pools.ProcessPool

The pathos-wrapped pool is pathos.pools.ProcessPool (and the old interface
provides it at pathos.multiprocessing.Pool).

tqdm for pathos: https://pypi.org/project/p-tqdm/

"""


"""
from pathos.multiprocessing import ProcessingPool as Pool

	class myClass:
		def __init__(self):
			pass

		def square(self, x):
			return x*x

		def run(self, inList):
			pool = Pool().map
			result = pool(self.square, inList)
			return result

	if __name__== '__main__' :
		m = myClass()
		print m.run(range(10))
"""


class RejABC:

    def __init__(self, observation, simulator, priors, distance='l2',
                 rng=np.random.RandomState, seed=None):

        super().__init__(
            observation=observation,
            simulator=simulator,
            priors=priors,
            distance=distance,
            rng=rng,
            seed=seed
        )

        self._n_sims = 0

    def sample(self, n_samples, epsilon=None, n_jobs=None, log=True):
        """
        n_jobs: int, default: None
            The maximum number of concurrently running jobs, such as the
            number of Python worker processes when backend=”multiprocessing” or
            the size of the thread-pool when backend=”threading”. If -1 all
            CPUs are used. If 1 is given, no parallel computing code is used
            at all, which is useful for debugging. For n_jobs below -1,
            (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but
            one are used. None is a marker for ‘unset’ that will be interpreted
            as n_jobs=1 (sequential execution) unless the call is performed
            under a parallel_backend context manager that sets another value
            for n_jobs.
        """

        samples = self._sample(n_samples)
        return n_samples

    def _sample(self, n_samples):
        samples = []

        for _ in range(n_samples):
            samples.append(self._draw_posterior_sample())

        return samples

    def _draw_posterior_sample(self):
        sample = None
        while sample is None:
            thetas = [prior.rvs(rng=self._rng, seed=self._seed + self._n_sims)
                      for prior in self._priors]
            sim = self._simulator(*thetas)
            self._n_sims += 1
            distance = self._distance(self._obs, sim)
            if distance <= self._epsilon:
                sample = thetas
        return sample

##


class ApproximateBayesianComputation(Procedure):
    r""""""

    def __init__(self, simulator, prior, summary, acceptor):
        super(ApproximateBayesianComputation, self).__init__()
        # Main classical ABC properties.
        self.acceptor = acceptor
        self.prior = prior
        self.simulator = simulator
        self.summary = summary

    def _register_events(self):
        # TODO Implement.
        pass

    def _draw_posterior_sample(self, summary_observation):
        sample = None

        while sample is None:
            prior_sample = self.prior.sample()
            x = self.simulator(prior_sample)
            s = self.summary(x)
            if self.acceptor(s, summary_observation):
                sample = prior_sample.unsqueeze(0)

        return sample

    def sample(self, observation, num_samples=1):
        samples = []

        summary_observation = self.summary(observation)
        for _ in range(num_samples):
            samples.append(self._draw_posterior_sample(summary_observation))
        samples = torch.cat(samples, dim=0)

        return samples


"""
OLD

"""


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
