
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
from matplotlib import gridspec
from pylfi.utils import setup_logger

from ._checks import *
from ._journal_base import JournalInternal


class Journal:

    def __init__(self):
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
        self._posterior_samples = {}

        self._sampler_stats = {}

        # bool used to limit access if journal has not been written to
        self._journal_started = False

    def _write_to_journal(
        self,
        observation,
        simulator,
        stat_calc,
        priors,
        distance_metric,
        inference_scheme,
        n_samples,
        n_simulations,
        posterior_samples,
        summary_stats,
        distances,
        epsilons,
        log
    ):
        # journal is started
        self._journal_started = True
        self._log = log

        if self._log:
            self.logger = setup_logger(self.__class__.__name__)
            self.logger.info("Write to journal.")

        # initialize data structures
        self._write_initialize(priors)
        # write sampler results
        self._write_results(posterior_samples,
                            summary_stats,
                            distances,
                            epsilons)

        self._sampler_results_df = pd.DataFrame(self._sampler_results)
        self._posterior_samples_df = pd.DataFrame(self._posterior_samples)

    def _write_initialize(self, priors):
        """Extract parameter names and set up data structures"""

        for parameter in priors:
            name = parameter.name
            tex = parameter.tex
            self.parameter_names.append(name)
            self._sampler_results[name] = None
            self._posterior_samples[name] = None
            self._sampler_stats[name] = None
            self.parameter_names_tex.append(tex)
            self._n_parameters += 1
            if tex is None:
                self.labels.append(name)
            else:
                self.labels.append(tex)

    def _write_results(self, posterior_samples, summary_stats, distances, epsilons):
        """Write sampler results to data structure"""

        for i, parameter_name in enumerate(self.parameter_names):
            self._sampler_results[parameter_name] = posterior_samples[:, i]
            self._posterior_samples[parameter_name] = posterior_samples[:, i]

        if summary_stats.ndim > 1:
            if len(summary_stats[0]) > 1:
                for i in range(summary_stats.ndim):
                    self._sampler_results[f"sum_stat{i+1}"] = summary_stats[:, i]
        else:
            self._sampler_results["sum_stat"] = summary_stats
        self._sampler_results["distance"] = distances
        self._sampler_results["epsilon"] = epsilons

    def _write_config(self, simulator, ):
        """Store inference configuration"""

        self.configuration["Inference scheme"] = inference_scheme
        self.configuration["Simulator model"] = simulator.__name__
        # prior?
        self.configuration["Distance metric"] = distance_metric.__name__

    def _create_idata(self):
        pass

    def _create_df(self):
        pass

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

    def get_simulator(self):
        pass

    def get_priors(self):
        pass

    def get_posterior(kernel='gaussian', bw='scott'):
        # return kde of posterior array(s)
        pass

    def sample_posterior(n_samples, kernel='gaussian', bw='scott'):
        # return n_samples from posterior kde
        pass

    def results_dict():
        check_journal_status(self._journal_started)
        return self._sampler_results

    def results_frame(self):
        check_journal_status(self._journal_started)
        return self._sampler_results_df

    def posterior_dict(self):
        check_journal_status(self._journal_started)
        return self._posterior_samples

    def posterior_frame(self):
        check_journal_status(self._journal_started)
        return self._posterior_samples_df

    @property
    def idata(self):
        posterior_dict = self.posterior_dict()
        # print(posterior_dict)
        idata = az.convert_to_inference_data(posterior_dict)
        return idata

    def plot_trace(self):
        az.plot_trace(self.idata)

    def plot_posterior(self):
        az.plot_posterior(self.idata)

    def displot(self):
        df = self.posterior_frame()
        sns.displot(df, kind="kde")

    def plot_pair(self, var_names, figsize=(6, 4)):
        ax = az.plot_pair(
            self.idata,
            var_names=var_names,
            kind=["scatter", "kde"],
            kde_kwargs={"fill_last": False},
            marginals=True,
            # coords=coords,
            point_estimate="mean",
            figsize=figsize,
        )
