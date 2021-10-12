import numpy as np
import pathos as pa


def check_and_set_jobs(n_jobs, logger=None):
    """Check and set passed number of jobs.

    Checks whether passed `n_jobs` has correct type and value, raises if not.

    If n_jobs exceeds the number of available CPUs found by `Pathos` (this
    might include hardware threads), `n_jobs` is set to the number found by
    `Pathos`.

    If `n_jobs=-1`, then n_jobs is set to half of the CPUs found by `Pathos`
    (we assume half of the CPUs are only hardware threads and ignore those).

    Parameters
    ----------
    n_jobs : :obj:`int`
        Number of processes (workers) passed by user.
    logger : :obj:`logging.Logger`
        Logger object.

    Returns
    -------
    n_jobs : :obj:`int`
        Possibly corrected number of processes (workers).
    """
    if not isinstance(n_jobs, int):
        msg = ("n_jobs must be passed as int.")
        raise TypeError(msg)

    if n_jobs < -1 or n_jobs == 0:
        msg = ("With the exception of n_jobs=-1, negative n_jobs cannot be "
               "passed.")
        raise ValueError(msg)

    n_cpus = pa.helpers.cpu_count()

    if n_jobs > n_cpus:
        if logger is not None:
            logger.warn("n_jobs exceeds the number of CPUs in the system.")
            logger.warn("Reducing n_jobs to match number of CPUs in system.")
        n_jobs = n_cpus
    elif n_jobs == -1:
        n_jobs = n_cpus // 2

    return n_jobs


def distribute_workload(n_samples, n_jobs, force_equal=False, logger=None):
    """Distribute workload between workers.

    Parameters
    ----------
    n_samples : :obj:`int`
        Number of (posterior) samples to draw.
    n_jobs : :obj:`int`
        Number of processes (workers).

    Returns
    -------
    object : :obj:`list`
        List of workload for each worker in the pool.
    """
    if force_equal:
        n_samples_org = n_samples
        while n_samples % n_jobs != 0:
            n_samples += 1
        if n_samples_org != n_samples and logger is not None:
            msg = (f"Number of samples set to {n_samples} to enforce "
                   "equal length of chains.")
            logger.warn(msg)
    inputs = np.arange(n_samples)
    chunks = np.array_split(inputs, n_jobs)
    tasks = [len(chunk) for chunk in chunks]
    return n_samples, tasks


def generate_seed_sequence(user_seed=None, pool_size=None):
    """Process a user-provided seed and convert it into initial states for
    parallel pool workers.

    Parameters
    ----------
    user_seed : :obj:`int`
        User-provided seed. Default is None.
    pool_size : :obj:`int`
        The number of spawns that will be passed to child processes.

    Returns
    -------
    seeds : :obj:`list`
        Initial states for parallel pool workers.
    """
    seed_sequence = np.random.SeedSequence(user_seed)
    seeds = seed_sequence.spawn(pool_size)
    return seeds


def advance_PRNG_state(seed, delta):
    """Advance the underlying PRNG as-if delta draws have occurred.

    In the ABC samplers, the random values are simulated using a
    rejection-based method and so, typically, more than one value from the
    underlying PRNG is required to generate a single posterior draw.

    Advancing a PRNG updates the underlying PRNG state as if a number
    of delta calls to the underlying PRNG have been made.

    Parameters
    ----------
    seed : SeedSequence
        Initial state.
    delta : :obj:`int`
        Number of draws to advance the PRNG.

    Returns
    -------
    object : PCG64
        PRNG advanced delta steps.
    """
    return np.random.PCG64(seed).advance(delta)


if __name__ == "__main__":
    import pylfi

    seed = None
    n_samples = 10
    n_jobs = 3

    #theta = pylfi.Prior('uniform', loc=0, scale=1, name='theta')
    alpha = 60         # prior hyperparameter (inverse gamma distribution)
    beta = 130         # prior hyperparameter (inverse gamma distribution)
    theta = pylfi.Prior('invgamma', alpha, loc=0, scale=beta, name='theta')

    def sample(n_samples, seed):
        samples = []
        for i in range(n_samples):
            #next_gen = np.random.PCG64(seed).advance(i)
            next_gen = advance_PRNG_state(seed, i)
            rng = np.random.default_rng(next_gen)
            samples.append(rng.normal())
        return samples

    def sample2(n_samples, seed):
        samples = []
        for i in range(n_samples):
            next_gen = advance_PRNG_state(seed, i)
            sample = theta.rvs(seed=next_gen)
            samples.append(sample)
        return samples

    seeds = generate_seed_sequence(seed, n_jobs)
    n_samples_distr = distribute_workload(n_samples, n_jobs)

    with pa.pools.ProcessPool(n_jobs) as pool:
        samples = pool.map(sample2, n_samples_distr, seeds)

    samples = np.concatenate(samples, axis=0)
    print(samples)

    """
    [2.62027044 2.607115   2.540724   2.04424041 2.1559203  1.76635961
 2.2483583  1.81264695 2.34266364 1.81355353]


    """
