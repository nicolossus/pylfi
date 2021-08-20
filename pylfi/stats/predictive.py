def _predictive(
    rng_key,
    model,
    posterior_samples,
    batch_shape,
    return_sites=None,
    parallel=True,
    model_args=(),
    model_kwargs={},
):
    model = numpyro.handlers.mask(model, mask=False)

    def single_prediction(val):
        rng_key, samples = val
        model_trace = trace(seed(substitute(model, samples), rng_key)).get_trace(
            *model_args, **model_kwargs
        )
        if return_sites is not None:
            if return_sites == "":
                sites = {
                    k for k, site in model_trace.items() if site["type"] != "plate"
                }
            else:
                sites = return_sites
        else:
            sites = {
                k
                for k, site in model_trace.items()
                if (site["type"] == "sample" and k not in samples)
                or (site["type"] == "deterministic")
            }
        return {
            name: site["value"] for name, site in model_trace.items() if name in sites
        }

    num_samples = int(np.prod(batch_shape))
    if num_samples > 1:
        rng_key = random.split(rng_key, num_samples)
    rng_key = rng_key.reshape(batch_shape + (2,))
    chunk_size = num_samples if parallel else 1
    return soft_vmap(
        single_prediction, (rng_key, posterior_samples), len(
            batch_shape), chunk_size
    )


class Predictive(object):
    """
    This class is used to construct predictive distribution. The predictive distribution is obtained
    by running model conditioned on latent samples from `posterior_samples`.
    .. warning::
        The interface for the `Predictive` class is experimental, and
        might change in the future.
    :param model: Python callable containing Pyro primitives.
    :param dict posterior_samples: dictionary of samples from the posterior.
    :param callable guide: optional guide to get posterior samples of sites not present
        in `posterior_samples`.
    :param dict params: dictionary of values for param sites of model/guide.
    :param int num_samples: number of samples
    :param list return_sites: sites to return; by default only sample sites not present
        in `posterior_samples` are returned.
    :param bool parallel: whether to predict in parallel using JAX vectorized map :func:`jax.vmap`.
        Defaults to False.
    :param batch_ndims: the number of batch dimensions in posterior samples. Some usages:
        + set `batch_ndims=0` to get prediction for 1 single sample
        + set `batch_ndims=1` to get prediction for `posterior_samples`
          with shapes `(num_samples x ...)`
        + set `batch_ndims=2` to get prediction for `posterior_samples`
          with shapes `(num_chains x N x ...)`. Note that if `num_samples`
          argument is not None, its value should be equal to `num_chains x N`.
    :return: dict of samples from the predictive distribution.
    """

    def __init__(
        self,
        model,
        posterior_samples=None,
        guide=None,
        params=None,
        num_samples=None,
        return_sites=None,
        parallel=False,
        batch_ndims=1,
    ):
        if posterior_samples is None and num_samples is None:
            raise ValueError(
                "Either posterior_samples or num_samples must be specified."
            )

        posterior_samples = {} if posterior_samples is None else posterior_samples

        prototype_site = batch_shape = batch_size = None
        for name, sample in posterior_samples.items():
            if batch_shape is not None and sample.shape[:batch_ndims] != batch_shape:
                raise ValueError(
                    f"Batch shapes at site {name} and {prototype_site} "
                    f"should be the same, but got "
                    f"{sample.shape[:batch_ndims]} and {batch_shape}"
                )
            else:
                prototype_site = name
                batch_shape = sample.shape[:batch_ndims]
                batch_size = int(np.prod(batch_shape))
                if (num_samples is not None) and (num_samples != batch_size):
                    warnings.warn(
                        "Sample's batch dimension size {} is different from the "
                        "provided {} num_samples argument. Defaulting to {}.".format(
                            batch_size, num_samples, batch_size
                        ),
                        UserWarning,
                    )
                num_samples = batch_size

        if num_samples is None:
            raise ValueError(
                "No sample sites in posterior samples to infer `num_samples`."
            )

        if batch_shape is None:
            batch_shape = (1,) * (batch_ndims - 1) + (num_samples,)

        if return_sites is not None:
            assert isinstance(return_sites, (list, tuple, set))

        self.model = model
        self.posterior_samples = {} if posterior_samples is None else posterior_samples
        self.num_samples = num_samples
        self.guide = guide
        self.params = {} if params is None else params
        self.return_sites = return_sites
        self.parallel = parallel
        self.batch_ndims = batch_ndims
        self._batch_shape = batch_shape

    def __call__(self, rng_key, *args, **kwargs):
        """
        Returns dict of samples from the predictive distribution. By default, only sample sites not
        contained in `posterior_samples` are returned. This can be modified by changing the
        `return_sites` keyword argument of this :class:`Predictive` instance.
        :param jax.random.PRNGKey rng_key: random key to draw samples.
        :param args: model arguments.
        :param kwargs: model kwargs.
        """
        posterior_samples = self.posterior_samples
        if self.guide is not None:
            rng_key, guide_rng_key = random.split(rng_key)
            # use return_sites='' as a special signal to return all sites
            guide = substitute(self.guide, self.params)
            posterior_samples = _predictive(
                guide_rng_key,
                guide,
                posterior_samples,
                self._batch_shape,
                return_sites="",
                parallel=self.parallel,
                model_args=args,
                model_kwargs=kwargs,
            )
        model = substitute(self.model, self.params)
        return _predictive(
            rng_key,
            model,
            posterior_samples,
            self._batch_shape,
            return_sites=self.return_sites,
            parallel=self.parallel,
            model_args=args,
            model_kwargs=kwargs,
        )
