#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from pylfi.densest import histogram
from pylfi.utils import setup_logger

from ._checks import *

#from ._plotting import add_densityplot, add_histplot, add_rugplot


class JournalInternal:
    """
    Internal journal handling
    """

    def __init__(self):
        # dict of accepted parameters; key=param_name, value=accepted_samples
        #self.accepted_parameters = {}
        # list of parameter names (the 'name' kw from Prior object)
        self.parameter_names = []
        # list of parameter LaTeX names (the 'tex' kw from Prior object)
        self.parameter_names_tex = []
        # list of labels (param names) for plots; uses 'name' if 'tex' is None
        self.labels = []
        # list for storing distances of accepted samples
        #self.distances = []
        #self.rel_distances = []
        # list for storing summary statistic values of accepted samples
        #self.sumstats = []
        # for tallying the number of inferred parameters
        self._n_parameters = 0

        # dict for storing inference configuration
        self.configuration = {}
        # dict for summarizing inference run
        self._sampler_summary = {}
        # dict for storing sampler results
        self._sampler_results = {}

        self._sampler_stats = {}

        # bool used to limit access if journal has not been written to
        self._journal_started = False

    def _start_journal(self, log, simulator, priors, inference_scheme, distance, n_simulations, epsilon):
        """Start the journal.

        To be called when an inference scheme is initialized in order to:
            * store configuration
            * tally the number of parameters that will be inferred
            * extract parameter names from prior objects
            * set key-value pairs in 'accepted_parameters' dict
            * set plot axis labels
        """
        # journal is started
        self._journal_started = True

        # logging on/off
        self._log = log
        if self._log:
            self.logger = setup_logger(self.__class__.__name__)
            self.logger.info("Create journal.")

        # store inference configuration
        self.configuration["Simulator model"] = simulator.__name__
        self.configuration["Inference scheme"] = inference_scheme
        self.configuration["Distance metric"] = distance.__name__
        self.configuration["Number of simulations"] = n_simulations
        self.configuration["Epsilon"] = epsilon

        # extract parameter names
        for parameter in priors:
            name = parameter.name
            tex = parameter.tex
            self.parameter_names.append(name)
            self._sampler_results[name] = []
            self._sampler_stats[name] = []
            self.parameter_names_tex.append(tex)
            self._n_parameters += 1
            if tex is None:
                self.labels.append(name)
            else:
                self.labels.append(tex)

        self._sampler_results["epsilon"] = []
        self._sampler_results["distance"] = []
        self._sampler_results["relative distance"] = []
        self._sampler_results["summary stats"] = []

    def _processing_msg(self):
        self.logger.info("Processing sampler results.")

    def _done_msg(self):
        self.logger.info("Ready for post-sampling processing and analysis.")
        self.logger.info("For details and tips, run 'journal.help'.")

    def _add_accepted_parameters(self, thetas):
        """Store accepted parameters."""
        for parameter_name, theta in zip(self.parameter_names, thetas):
            # self.accepted_parameters[parameter_name].append(theta)
            self._sampler_results[parameter_name].append(theta)

    def _add_distance(self, distance):
        """Store calculated distance corresponding to accepted parameters."""
        self._sampler_results["distance"].append(distance)

    def _add_rel_distance(self, rel_distance):
        """Store calculated distance corresponding to accepted parameters."""
        self._sampler_results["relative distance"].append(rel_distance)

    def _add_threshold(self, epsilon):
        """Store the threshold value corresponding to accepted parameters."""
        self._sampler_results["epsilon"].append(epsilon)

    def _add_sumstats(self, sumstats):
        """Store summary statistics corresponding to accepted parameters."""
        self._sampler_results["summary stats"].append(sumstats)

    def _process_inference(self, n_sims, n_samples, time):
        """Post-sampling summary processing and adjustment.

        To be called at the end of inference scheme in order to
            * store the number of simulations
            * store the number of accepted parameters
            * calculate the acceptance ratio

        TODO:
        - calculate sample mean, std, and similar
        - if lra, also for adjusted
        - KDE handling not decided yet
        """
        # args: lra, kde params
        # move all post-sampling into here
        '''
        if self._log:
            self.logger.info(f"Initializing post-sampling processing.")
        '''

        self._sampler_results_df = pd.DataFrame(self._sampler_results)

        # transform accepted samples into 1D arrays
        #*data, = self._params_as_arrays()
        #self._params_unadj = data

        accept_ratio = n_samples / n_sims
        self._sampler_summary["Simulations"] = n_sims
        self._sampler_summary["Posterior samples"] = n_samples
        self._sampler_summary["Acceptance ratio"] = accept_ratio
        self._sampler_summary["Wall time (s)"] = time

        self._sampler_summary_df = pd.DataFrame(self._sampler_summary,
                                                index=[0])

        for param_name in self.parameter_names:
            self._sampler_stats[param_name].append(
                self._sampler_results_df[param_name].mean())
            self._sampler_stats[param_name].append(
                self._sampler_results_df[param_name].median())
            self._sampler_stats[param_name].append(
                self._sampler_results_df[param_name].var())
            self._sampler_stats[param_name].append(
                self._sampler_results_df[param_name].std())

        self._sampler_stats_df = pd.DataFrame(self._sampler_stats,
                                              index=["mean", "median", "var", "std"])
        # posterior means
        # uncertainty

    def _params_as_arrays(self):
        """Transform data of accepted parameters to 1D arrays"""

        samples = self.accepted_parameters
        if len(self.parameter_names) > 1:
            params = (np.asarray(samples[name], float).squeeze() if np.asarray(
                samples[name], float).ndim > 1 else np.asarray(samples[name], float) for name in self.parameter_names)
        else:
            samples = np.asarray(samples[self.parameter_names[0]], float)
            params = samples.squeeze(
            ) if samples.ndim > 1 else np.array([samples])
        return params

    def params_as_arrays(self):
        """DEPRECATE :: Unpack and return 1D parameter arrays."""
        *data, = self._params_as_arrays()
        return data

    def params_unadjusted(self):
        """List of arrays of unadjusted accepted parameters"""
        return self._params_unadj

    def _set_point_estimate_statistic(self, statistic):
        # add mode
        if statistic == 'mean':
            return np.mean
        elif statistic == 'median':
            return np.median

    def _sample_point_estimates(self, statistic):
        """
        Calculate point estimate of inferred parameters.

        In statistics, point estimation involves the use of sample data to
        calculate a single value (known as a point estimate since it identifies
        a point in some parameter space) which is to serve as a "best guess" or
        "best estimate" of an unknown population parameter (for example,
        the population mean).

        https://en.wikipedia.org/wiki/Point_estimation
        """
        *samples, = self.params_as_arrays()

        stat_func = self._set_point_estimate_statistic(statistic)

        if self._n_parameters == 1:
            point_estimates = [stat_func(samples)]
        else:
            point_estimates = [stat_func(sample) for sample in samples]
        return point_estimates

    def _default_plot_style(self):
        """Default plot style."""
        sns.set()
        sns.set_context("paper")
        sns.set_style("darkgrid", {"axes.facecolor": "0.96"})

        # Set fontsizes in figures
        params = {'legend.fontsize': 'large',
                  'axes.labelsize': 'large',
                  'axes.titlesize': 'large',
                  'xtick.labelsize': 'large',
                  'ytick.labelsize': 'large',
                  'legend.fontsize': 'large',
                  'legend.handlelength': 2}
        plt.rcParams.update(params)
        plt.rc('text', usetex=True)

    def _custom_plot_style(self, plot_style, textsize, usetex):
        """Custom plot style with defaults for unspecified configurations."""
        if plot_style == 'sns':
            sns.set()
            sns.set_context("paper")
            sns.set_style("darkgrid", {"axes.facecolor": "0.96"})

        # Set fontsizes in figures
        size = 'large' if textsize is None else textsize
        params = {'legend.fontsize': size,
                  'axes.labelsize': size,
                  'axes.titlesize': size,
                  'xtick.labelsize': size,
                  'ytick.labelsize': size,
                  'legend.fontsize': size,
                  'legend.handlelength': 2}
        plt.rcParams.update(params)

        if usetex:
            plt.rc('text', usetex=True)

    def _set_plot_style(self, plot_style, textsize, usetex):
        """Set plot style"""

        check_plot_style_input(plot_style, usetex)

        if plot_style is not None:
            if plot_style == 'pylfi':
                self._default_plot_style()
            else:
                self._custom_plot_style(plot_style, textsize, usetex)

    def _set_grid(self):
        """Make grid for subplots based on the number of parameters.

        The scheme ensures rows * cols >= n_parameters and tries to get as
        close as possible to sqrt(n_parameters) x sqrt(n_parameters).
        """
        max_cols = 3
        min_cols = 2
        if self._n_parameters <= max_cols:
            return 1, self._n_parameters

        cols = round(np.clip(self._n_parameters**0.5, min_cols, max_cols))
        rows = int(np.ceil(self._n_parameters / cols))
        return rows, cols

    def _set_figure_layout(self, grid, figsize, dpi):
        """Set figure layout"""
        if grid is None:
            rows, cols = self._set_grid()
        else:
            check_grid_input(self._n_parameters, grid)
            rows, cols = grid

        if figsize is None:
            figsize = (8, 4)

        fig = plt.figure(figsize=figsize, tight_layout=True, dpi=dpi)
        gs = gridspec.GridSpec(nrows=rows, ncols=cols, figure=fig)
        return fig, gs

    def _add_histplot(self, data, ax=None, bins=10, **kwargs):
        """
        Histogram plot

        Parameters
        ----------
        data : array_like
            data
        ax : Axes, optional
            Pre-existing axes for the plot. Otherwise, call matplotlib.pyplot.gca() internally.

        Returns
        -------
        ax : Axes
            The new Axes object
        """
        if ax is None:
            ax = plt.gca()
        counts, bins = histogram(data, bins=bins)
        ax.hist(bins[:-1], bins=bins, weights=counts, histtype='bar',
                edgecolor=None, color='steelblue', alpha=0.9, **kwargs)

    def _add_rugplot(self, data, ax=None, **kwargs):
        """
        Rug plot

        """
        if ax is None:
            ax = plt.gca()
        ax.plot(data, np.full_like(data, -0.01),
                '|k', markeredgewidth=1, **kwargs)

    def _add_densityplot(self, x, density, ax=None, fill=False, **kwargs):
        """
        KDE plot


        """
        if ax is None:
            ax = plt.gca()

        if fill:
            ax.fill_between(x, density, alpha=0.5, **kwargs)
        else:
            ax.plot(x, density, **kwargs)
