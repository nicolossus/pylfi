#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from ._checks import *
from ._journal_base import JournalInternal

#from ._plotting import add_densityplot, add_histplot, add_rugplot

# TODO: store in pandas dataframe


class Journal(JournalInternal):

    def __init__(self):
        super().__init__()

    '''
    @property
    def get_distances(self):
        check_journal_status(self._journal_started)
        return self.distances

    @property
    def get_raw_distances(self):
        check_journal_status(self._journal_started)
        return self.raw_distances
    '''

    @property
    def sampler_results(self):
        check_journal_status(self._journal_started)
        return self._sampler_results_df

    @property
    def sampler_summary(self):
        check_journal_status(self._journal_started)
        return self._sampler_summary_df

    @property
    def sampler_stats(self):
        check_journal_status(self._journal_started)
        return self._sampler_stats_df

    @property
    def get_number_of_simulations(self):
        check_journal_status(self._journal_started)
        return self.sampler_summary["Number of simulations"]

    @property
    def get_number_of_accepted_simulations(self):
        check_journal_status(self._journal_started)
        return self.sampler_summary["Number of accepted simulations"]

    @property
    def get_acceptance_ratio(self):
        check_journal_status(self._journal_started)
        return self.sampler_summary["Acceptance ratio"]

    @property
    def get_accepted_parameters(self):
        return self.accepted_parameters

    def histplot(self, bins=10, rug=False, point_estimate=None, true_parameter_values=None,
                 path_to_save=None, show=True,
                 plot_style=None, grid=None, figsize=None, dpi=120, textsize=None, usetex=False, **kwargs):
        """Histplot

        histogram(s) of sampled parameter(s)

        point estimate : mean, median, mode, None

        The Mode value is the value that appears the most number of times
        The median value is the value in the middle, after you have sorted all the values
        The mean value is the average value

        figsize : tuple
            Figure size. If None it will be defined automatically.
        textsize: float
            Text size scaling factor for labels, titles and lines.
            If None it will be set to 'large'.
        """

        N = self._n_parameters
        # run checks
        check_journal_status(self._journal_started)

        if point_estimate is not None:
            check_point_estimate_input(point_estimate)
            point_estimates = self._point_estimates()

        is_plot_true_vals = False
        if true_parameter_values is not None:
            check_true_parameter_values(
                self._n_parameters, true_parameter_values)
            is_plot_true_vals = True

        # get sampled parameters
        sample_data = self.params_as_arrays()

        self._set_plot_style(plot_style, textsize, usetex)
        fig, gs = self._set_figure_layout(grid, figsize, dpi)

        for index, data in enumerate(sample_data):
            ax = fig.add_subplot(gs[index])
            self._add_histplot(data, ax=ax, bins=bins,
                               label='Accepted samples', **kwargs)
            if rug:
                self._add_rugplot(data, ax=ax)
            if point_estimate is not None:
                ax.axvline(point_estimates[index], color='b',
                           ymax=0.3, label=f'Sample {point_estimate}')
            if true_parameter_values is not None:
                ax.axvline(true_parameter_values[index], color='r',
                           ls='--', ymax=0.3, label='Groundtruth')
            ax.set_xlabel(self.labels[index])
            ax.set_ylabel('Density')
            ax.set_title("Histogram of accepted " + self.labels[index])

        handles, labels = plt.gca().get_legend_handles_labels()

        # fix legend ordering
        if all(e is None for e in [point_estimate, true_parameter_values]):
            order = [0]
        elif all(e is not None for e in [point_estimate, true_parameter_values]):
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

    '''
    def histplot(self, bins=10, rug=False, point_estimate='mean', show=True, dpi=120, path_to_save=None, true_parameter_values=None, **kwargs):
        """
        histogram(s) of sampled parameter(s)

        point estimate : mean, median, mode, None

        The Mode value is the value that appears the most number of times
        The median value is the value in the middle, after you have sorted all the values
        The mean value is the average value
        """
        pass
    '''

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
