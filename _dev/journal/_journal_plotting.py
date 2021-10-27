#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from pylfi.densest import histogram
from pylfi.utils import set_plot_style


class JournalPlotting(JournalBase):

    def __init__():
        super().__init__()

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
                  'axes.labelsize': 'size,
                  'axes.titlesize': size,
                  'xtick.labelsize': size,
                  'ytick.labelsize': 'size,
                  'legend.fontsize': size,
                  'legend.handlelength': 2}
        plt.rcParams.update(params)

        if usetex:
            plt.rc('text', usetex=True)

    def _set_plot_style(self, plot_style, textsize, usetex):
        """Set plot style"""

        check_plot_style_input(plot_style, usetex)

        if plot_style is not None:
            if plot_style == 'default':
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
            figsize = (8, 6)

        fig = plt.figure(figsize=figsize, tight_layout=True, dpi=dpi)
        gs = gridspec.GridSpec(nrows=rows, ncols=cols, figure=fig)
        return fig, gs

    def histplot(self, bins=10, path_to_save=None, show=True, point_estimate=None, true_parameter_values=None,
                 plot_style=None, grid=None, figsize=None, dpi=120, textsize=None, usetex=False):
        """Histplot

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
            if point_estimate is not None:
                ax.axvline(point_estimates[index], color='b',
                           ymax=0.3, label=f'Sample {point_estimate}')
            if true_parameter_values is not None:
                ax.axvline(true_parameter_values[index], color='r',
                           ls='--', ymax=0.3, label='Groundtruth')
            ax.set_xlabel(self.labels[index])
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

    # @staticmethod
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


'''
def histplot(self, density=True, show=True, dpi=120, path_to_save=None, true_parameter_values=None):
    """
    histogram(s) of sampled parameter(s)
    """
    # run checks
    check_journal_status(self._journal_started)
    true_vals_bool = False
    if true_parameter_values is not None:
        check_true_parameter_values(
            self._n_parameters, true_parameter_values)
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

    # plt.axvline(theta_real, ymax=0.3, color='k')


def plot_hist(data, N, bins=10, dpi=120, path_to_save=None, true_parameter_values=None, **kwargs):
    """
    Parameters
    ----------
    data : array_like
        data
    N : int
        Number of parameters
    """

    fig = plt.figure(figsize=(8, 6), tight_layout=True, dpi=dpi)
    self.set_plot_style()

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
    pass


def add_histplot(data, ax=None, bins=10, **kwargs):
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


def add_rugplot(data, ax=None, **kwargs):
    """
    Rug plot

    """
    if ax is None:
        ax = plt.gca()
    ax.plot(data, np.full_like(data, -0.01), '|k', markeredgewidth=1, **kwargs)


def add_densityplot(x, density, ax=None, fill=False, **kwargs):
    """
    KDE plot


    """
    if ax is None:
        ax = plt.gca()

    if fill:
        ax.fill_between(x, density, alpha=0.5, **kwargs)
    else:
        ax.plot(x, density, **kwargs)


if __name__ == "__main__":
    from matplotlib import gridspec

    def make_data(N, f=0.3, rseed=1):
        rand = np.random.RandomState(rseed)
        x = rand.randn(N)
        x[int(f * N):] += 5
        return x

    data = make_data(500)

    '''
    fig, ax = plt.subplots(1, 2)
    histplot(data, ax[0])
    rugplot(data, ax[0])
    histplot(data, ax[1], bins='freedman')
    rugplot(data, ax[1])
    plt.show()
    '''

    rules = [30, 'sqrt', 'sturges', 'scott', 'freedman', 'knuth']

    fig = plt.figure(figsize=(10, 6), tight_layout=True, dpi=120)
    gs = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)

    for i, rule in enumerate(rules):
        ax = fig.add_subplot(gs[i])
        add_histplot(data, ax, bins=rule, label=f"bins = {rule}")
        add_rugplot(data, ax)
        ax.set_xlabel("$x$")
        ax.set_ylabel("Density")
        ax.legend(loc="upper left")

    plt.show()
