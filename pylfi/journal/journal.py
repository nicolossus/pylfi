#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import sys

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib import gridspec


class Journal:

    def __init__(self):
        # list of parameter names (the 'name' kw from Prior object)
        self.param_names = []
        # list of parameter LaTeX names (the 'tex' kw from Prior object)
        self.param_names_tex = []

        # for tallying the number of inferred parameters
        self._n_params = 0

        # sampler configuration and info
        self._info_df = {}

        # posterior samples dicts for data structures
        self._idata = {}
        self._idata_plot = {}
        self._df = {}
        self._df_plot = {}

        # bool used to limit access if journal has not been written to
        self._journal_written = False

    def _write_to_journal(
        self,
        inference_scheme,
        observation,
        simulator,
        stat_calc,
        priors,
        n_samples,
        chains,
        samples,
        accept_ratio,
        epsilon,
        quantile
    ):
        """
        Write to journal
        """

        self._observation = observation
        self._simulator = simulator
        self._stat_calc = statistics_calculator
        self._priors = priors

        # Extract parameter names and set up data structures
        for param in priors:

            self._param_names.append(param.name)
            self._idata[param.name] = None
            seldf._df[param.name] = None

            if param.tex is not None:
                self._param_names_tex.append(param.tex)
                self._idata_plot[param.tex] = None
                seldf._df_plot[param.tex] = None

            self._n_params += 1

        # configuration
        self._config["Inference scheme"] = inference_scheme
        self._config["Simulator model"] = self._simulator.__name__
        self._config["quantile"] = quantile
        self._config["epsilon"] = epsilon
        self._config["accept_ratio"] = accept_ratio

        for i, parameter_name in enumerate(self.parameter_names):
            self._sampler_results[parameter_name] = posterior_samples[:, i]
            self._posterior_samples[parameter_name] = posterior_samples[:, i]

        # Written to journal
        self._journal_written = True

    def _idata(self):
        for i, param_name in enumerate(parameter_names):
            idata_posterior[param_name] = (
                ["chain", "draw"], [posterior_samples[:, i]])

        idata_coords = {"chain": chains,
                        "draw": np.arange(n_samples, dtype=int)}

        # print(idata_posterior)
        # print(idata_coords)

        idata = xr.Dataset(idata_posterior, idata_coords)
        print(idata)
        # ppc - need to add simulator
        # idata_plot with tex names (see Prior class for extraction)
        ppc = pm.sample_posterior_predictive(..., keep_size=True)
        az.concat(idata, az.from_dict(posterior_predictive=ppc), inplace=True)

    def plot_priors(self):
        pass
