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
        self._param_names = []
        # list of parameter LaTeX names (the 'tex' kw from Prior object)
        self._param_names_tex = []

        # for tallying the number of inferred parameters
        self._n_params = 0

        # sampler configuration and info
        self._info_df = {}

        # posterior samples dicts for data structures
        self._idata = {}
        self._idata_plot = {}
        self._df = {}
        self._df_plot = {}

        # flags
        # Becomes True if df with tex names are constructed
        self._df_plot_exist = False
        # Becomes True if idata with tex names are constructed
        self._idata_plot_exist = False

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
        n_chains,
        n_sims,
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
        self._stat_calc = stat_calc
        self._priors = priors

        # Extract parameter names and set up data structures
        for param in priors:

            self._param_names.append(param.name)
            self._idata[param.name] = None
            self._df[param.name] = None

            if param.tex is not None:
                self._param_names_tex.append(param.tex)
                self._idata_plot[param.tex] = None
                self._df_plot[param.tex] = None

            self._n_params += 1

        if self._df_plot:
            self._df_plot_exist = True
        if self._idata_plot:
            self._idata_plot_exist = True

        # write to data structures
        self._write_config(inference_scheme, quantile, epsilon, accept_ratio)
        self._write_df(samples)
        #self._write_idata(samples, n_samples, n_chains)

        # Written to journal
        self._journal_written = True

    def _write_config(self, inference_scheme, quantile, epsilon, accept_ratio):
        self._info_df["Inference scheme"] = inference_scheme
        #self._info_df["Simulator model"] = self._simulator.__name__
        self._info_df["quantile"] = quantile
        self._info_df["epsilon"] = epsilon
        self._info_df["accept_ratio"] = accept_ratio

    def _write_df(self, samples):
        for i, param_name in enumerate(self._param_names):
            self._df[param_name] = samples[:, i]
            if self._df_plot_exist:
                self._df_plot[self._param_names_tex[i]] = samples[:, i]

        self._df = pd.DataFrame(self._df)
        if self._df_plot_exist:
            self._df_plot = pd.DataFrame(self._df_plot)

    def _write_idata(self, samples, n_samples, n_chains):
        if n_chains == 1:
            n_draws = n_samples
            for i, param_name in enumerate(self._param_names):
                self._idata[param_name] = (["chain", "draw"], samples[:, i])
                if self._idata_plot_exist:
                    self._idata_plot[self._param_names_tex[i]] = (
                        ["chain", "draw"], samples[:, i])
        else:
            n_draws = samples.shape[1]
            for i, param_name in enumerate(self._param_names):
                self._idata[param_name] = (["chain", "draw"], samples[:, :, i])
                if self._idata_plot_exist:
                    self._idata_plot[self._param_names_tex[i]] = (
                        ["chain", "draw"], samples[:, :, i])

        self._idata = az.convert_to_inference_data(
            self._idata,
            coords={
                "chain": np.arange(n_chains),
                "draws": np.arange(n_draws)
            },
            observed_data=self._observation
        )

        if self._idata_plot_exist:
            self._idata_plot = az.convert_to_inference_data(
                self._idata_plot,
                coords={
                    "chain": np.arange(n_chains),
                    "draws": np.arange(n_draws)
                },
                observed_data=self._observation
            )

    def _idata(self):
        for i, param_name in enumerate(parameter_names):
            idata_posterior[param_name] = (
                ["chain", "draw"], [posterior_samples[:, i]])
        coords = {"draw": np.arange(1, N + 1), "chain": np.arange(n_chains)}
        idata_coords = {"chain": chains,
                        "draw": np.arange(n_samples, dtype=int)}

        # print(idata_posterior)
        # print(idata_coords)

        idata = xr.Dataset(idata_posterior, idata_coords)
        print(idata)
        # ppc - need to add simulator
        # idata_plot with tex names (see Prior class for extraction)
        ppc = pm.sample_posterior_predictive(..., keep_size=True)
        # below can perhaps be used in method for plotting ppc
        az.concat(idata, az.from_dict(posterior_predictive=ppc), inplace=True)

        '''
        Stacking chains and draws is often useful when one doesn't care about
        which chain a draw is coming from. This is currently possible by doing
        idata.posterior.stack(sample=("chain", "draw"))
        '''

    def _check_journal_status(self):
        """Check if journal has been initiated by an inference scheme.

        Parameters
        ----------
        is_journal_started : bool
            ``True`` if the journal has been initiated by an inference scheme,
            ``False`` otherwise.

        Raises
        ------
        RuntimeError
            If journal has not been initiated by an inference scheme.
        """
        if not self._journal_written:
            msg = ("Journal unavailable; run an inference scheme first")
            raise RuntimeError(msg)

    def plot_priors(self):
        pass

    @property
    def idata(self):
        self._check_journal_status()
        return self._idata

    @property
    def df(self):
        self._check_journal_status()
        return self._df

    def save(self, filename):
        """
        Stores the journal to disk.

        Parameters
        ----------
        filename: string
            the location of the file to store the current object to.
        """

        with open(filename, 'wb') as output:
            pickle.dump(self, output, -1)

    def load(self, filename):
        with open(filename, 'rb') as input:
            journal = pickle.load(input)
        return journal
