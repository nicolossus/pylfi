#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

# TODO: store in pandas dataframe


class Journal:
    """Journal.

    Class for representing the Hodgkin-Huxley model.

    All model parameters can be accessed (get or set) as class attributes.

    The following solutions are available as class attributes after calling
    the class method `solve`:

    Attributes
    ----------
    t : array_like
        The time array of the spike.
    V : array_like
        The voltage array of the spike.
    """

    def __init__(self):
        """Define the model parameters.

        Parameters
        ----------
        V_rest : float, default: -65.
            Resting potential of neuron in units: mV
        Cm : float, default: 1.
            Membrane capacitance in units: μF/cm**2
        gbar_K : float, default: 36.
            Potassium conductance in units: mS/cm**2
        gbar_Na : float, default: 120.
            Sodium conductance in units: mS/cm**2
        gbar_L : float, default: 0.3.
            Leak conductance in units: mS/cm**2
        E_K : float, default: -77.
            Potassium reversal potential in units: mV
        E_Na : float, default: 50.
            Sodium reversal potential in units: mV
        E_L : float, default: -54.4
            Leak reversal potential in units: mV

        Notes
        -----
        Default parameter values as given by Hodgkin and Huxley (1952).
        """

        self.accepted_parameters = {}
        self.parameter_names = []
        self.parameter_names_tex = []
        self.labels = []
        self.distances = []
        self.sumstats = []
        self._n_parameters = 0

        self.configuration = {}
        self.sampler_summary = {}

        self._journal_started = False

    def _start_journal(self):
        self._journal_started = True

    def _check_journal_status(self):
        if not self._journal_started:
            msg = "Journal unavailable; run an inference scheme first"
            raise ValueError(msg)

    def _check_true_parameter_values(self, true_parameter_values):
        if not isinstance(true_parameter_values, list):
            msg = "True parameter values must be provided in a list"
            raise ValueError(msg)
        if self._n_parameters != len(true_parameter_values):
            msg = "The number of true parameter values in list must equal the number of inferred parameters."
            raise ValueError(msg)

    def _add_config(self, simulator, inference_scheme, distance, n_simulations, epsilon):
        """
        docs
        """

        self.configuration["Simulator model"] = simulator.__name__
        self.configuration["Inference scheme"] = inference_scheme
        self.configuration["Distance metric"] = distance.__name__
        self.configuration["Number of simulations"] = n_simulations
        self.configuration["Epsilon"] = epsilon

    def _add_parameter_names(self, priors):
        for parameter in priors:
            name = parameter.name
            tex = parameter.tex
            self.parameter_names.append(name)
            self.accepted_parameters[name] = []
            self.parameter_names_tex.append(tex)
            self._n_parameters += 1
            if tex is None:
                self.labels.append(name)
            else:
                self.labels.append(tex)

    def _add_accepted_parameters(self, thetas):
        """
        docs
        """

        for parameter_name, theta in zip(self.parameter_names, thetas):
            self.accepted_parameters[parameter_name].append(theta)

    def _add_distance(self, distance):
        """
        docs
        """

        self.distances.append(distance)

    def _add_sumstat(self, sumstat):
        """
        docs
        """

        self.sumstats.append(sumstat)

    def _add_sampler_summary(self, number_of_simulations, accepted_count):
        """
        docs
        """

        accept_ratio = accepted_count / number_of_simulations
        # number of parameters estimated
        self.sampler_summary["Number of simulations"] = number_of_simulations
        self.sampler_summary["Number of accepted simulations"] = accepted_count
        self.sampler_summary["Acceptance ratio"] = accept_ratio
        # posterior means
        # uncertainty

    @property
    def get_accepted_parameters(self):
        """
        docs
        """
        return self.accepted_parameters

    def _get_params_as_arrays(self):
        """
        Transform data of accepted parameters to 1D arrays
        """

        samples = self.get_accepted_parameters
        if len(self.parameter_names) > 1:
            params = (np.asarray(samples[name], float).squeeze() if np.asarray(
                samples[name], float).ndim > 1 else np.asarray(samples[name], float) for name in self.parameter_names)
        else:
            samples = np.asarray(samples[self.parameter_names[0]], float)
            params = samples.squeeze() if samples.ndim > 1 else samples
        return params

    def _point_estimates(self):
        """
        Calculate point estimate of inferred parameters
        """
        *samples, = self._get_params_as_arrays()
        if self._n_parameters == 1:
            point_estimates = [np.mean(samples)]
        else:
            point_estimates = [np.mean(sample) for sample in samples]
        return point_estimates

    @property
    def get_distances(self):
        """
        docs
        """
        self._check_journal_status()
        return self.distances

    @property
    def get_number_of_simulations(self):
        self._check_journal_status()
        return self.sampler_summary["Number of simulations"]

    @property
    def get_number_of_accepted_simulations(self):
        self._check_journal_status()
        return self.sampler_summary["Number of accepted simulations"]

    @property
    def get_acceptance_ratio(self):
        self._check_journal_status()
        return self.sampler_summary["Acceptance ratio"]

    def _samples(self, name):
        pass

    def _kde(self):
        pass

    def _set_plot_style(self):
        params = {'legend.fontsize': 'large',
                  'axes.labelsize': 'large',
                  'axes.titlesize': 'large',
                  'xtick.labelsize': 'large',
                  'ytick.labelsize': 'large',
                  'legend.fontsize': 'large',
                  'legend.handlelength': 2}
        plt.rcParams.update(params)
        plt.rc('text', usetex=True)

    '''
    def _add_histplot(self, data, ax, index, density, point_estimates, true_vals_bool, true_parameter_values):
        n_bins = self._freedman_diaconis_rule(data)
        ax.hist(data, density=density, histtype='bar', edgecolor=None,
                color='steelblue', alpha=0.5, bins=n_bins, label="Accepted samples")
        ax.axvline(
            point_estimates[index], color='b', label="Point estimate")
        if true_vals_bool:
            ax.axvline(
                true_parameter_values[index], color='r', linestyle='--', label="Groundtruth")
        ax.set_xlabel(self.labels[index])
        ax.set_title("Histogram of accepted " + self.labels[index])
    '''

    def histplot(self, density=True, show=True, dpi=120, path_to_save=None, true_parameter_values=None):
        """
        histogram(s) of sampled parameter(s)
        """
        # run checks
        self._check_journal_status()
        true_vals_bool = False
        if true_parameter_values is not None:
            self._check_true_parameter_values(true_parameter_values)
            true_vals_bool = True

        # get sampled parameters
        *data, = self._get_params_as_arrays()
        point_estimates = self._point_estimates()

        fig = plt.figure(figsize=(8, 6), tight_layout=True, dpi=dpi)
        self._set_plot_style()

        N = self._n_parameters

        if N == 1:
            ax = plt.subplot(111)
            legend_position = 0
            index = 0
            self._add_histplot(
                data, ax, index, density, point_estimates, true_vals_bool, true_parameter_values)
        else:
            if N == 2 or N == 4:
                cols = 2
                legend_position = 1
            else:
                cols = 3
                legend_position = 2
            rows = int(np.ceil(N / cols))
            gs = gridspec.GridSpec(ncols=cols, nrows=rows, figure=fig)
            for index, data in enumerate(data):
                ax = fig.add_subplot(gs[index])
                self._add_histplot(
                    data, ax, index, density, point_estimates, true_vals_bool, true_parameter_values)

        handles, labels = plt.gca().get_legend_handles_labels()
        if true_vals_bool:
            order = [2, 0, 1]
        else:
            order = [1, 0]

        plt.legend([handles[idx] for idx in order],
                   [labels[idx] for idx in order],
                   loc='center left',
                   bbox_to_anchor=(1.04, 0.5),
                   fancybox=True,
                   borderaxespad=0.1,
                   ncol=1
                   )

        if path_to_save is not None:
            fig.savefig(path_to_save, dpi=dpi)
        if show:
            plt.show()

    def adjusted_histplot():
        # regression adjusted
        pass

    def kdeplot(self, kernel="gaussian"):
        ax[1].plot(x, kernel.evaluate(x), label="approximate posterior")
        pass

    def distplot(self, kde=True, kde_kwds=None, ax=None):
        """
        """
        if ax is None:
            ax = plt.gca()
        pass

    def posterior_kde(self, kernel="gaussian"):
        pass

    @ property
    def summary(self):
        pass

    @ property
    def print_summary(self):
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

    '''
    @staticmethod
    def run_lra(
        theta: torch.Tensor,
        x: torch.Tensor,
        observation: torch.Tensor,
        sample_weight=None,
    ) -> torch.Tensor:
        """Return parameters adjusted with linear regression adjustment.
        Implementation as in Beaumont et al. 2002: https://arxiv.org/abs/1707.01254
        """

        theta_adjusted = theta
        for parameter_idx in range(theta.shape[1]):
            regression_model = LinearRegression(fit_intercept=True)
            regression_model.fit(
                X=x,
                y=theta[:, parameter_idx],
                sample_weight=sample_weight,
            )
            theta_adjusted[:, parameter_idx] += regression_model.predict(
                observation.reshape(1, -1)
            )
            theta_adjusted[:, parameter_idx] -= regression_model.predict(x)

        return theta_adjusted
    '''
