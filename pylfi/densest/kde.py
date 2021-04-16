#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings

import numpy as np
from pylfi.utils import check_1D_data, check_bandwidth, check_kernel
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KernelDensity

# warnings.filterwarnings("ignore")

# calculate initial bw with ISJ or silverman and then do a gridsearch in the neighborhood around this value.

'''
perhaps scrap sklearn and use this instead:
https://github.com/tommyod/KDEpy

sklearn:
https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/neighbors/_kde.py
https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
https://stackoverflow.com/questions/53135857/what-does-rank-test-score-stand-for-from-the-model-cv-results

performance comparison:
https://kdepy.readthedocs.io/en/latest/comparison.html

other resources:
----------------
https://stats.stackexchange.com/questions/90656/kernel-bandwidth-scotts-vs-silvermans-rules
http://www.stat.cmu.edu/~cshalizi/350/lectures/28/lecture-28.pdf
https://stats.stackexchange.com/questions/173637/generating-a-sample-from-epanechnikovs-kernel
https://github.com/statsmodels/statsmodels/blob/master/statsmodels/nonparametric/bandwidths.py
https://stackoverflow.com/questions/30145957/plotting-2d-kernel-density-estimation-with-python
'''


class KDE:
    """Kernel density estimation.

    Parameters
    ----------
    data : ndarray
        Array of data
    bandwidth : {float, str}, optional
        Kernel bandwidth either ... Default=1.

    Notes
    -----
    Grid search; ``bandwidth='auto'`` : grid with 10 uniformly spaced bandwidth
    values between

    `initial_bandwidth` keyword; calculate initial bw with ISJ or silverman and
    then do a gridsearch in the neighborhood around this value.
    """

    def __init__(self, data, bandwidth=1.0, kernel='gaussian', initial_bandwidth='isj', **kwargs):

        check_bandwidth(bandwidth)
        check_kernel(kernel)

        self._data = data
        self._bw = bandwidth
        self._kernel = kernel

        self._data_len = len(self._data)

        self._grid_scores = None
        self._best_params = None
        self._optimal_bw = None
        self._optimal_kernel = None

        self._select_kde(**kwargs)
        self._fit()

    def __call__(self, x):
        return self.density(x)

    def _select_kde(self, **kwargs):
        if isinstance(self._bw, str):
            if self._bw == "auto":
                self._bw = np.logspace(-2, 1, 100)
            elif self._bw == "scott":
                self._bw = [self._scotts_bw_rule()]
            elif self._bw == "silverman":
                self._bw = [self._silverman_bw_rule()]
        elif isinstance(self._bw, (int, float)):
            self._bw = [self._bw]

        if isinstance(self._kernel, str):
            if self._kernel == 'auto':
                self._kernel = ['gaussian', 'tophat', 'epanechnikov',
                                'exponential', 'linear', 'cosine']
            else:
                self._kernel = [self._kernel]

        self._kde = self._optimal_kde(**kwargs)

    def _scotts_bw_rule(self):
        """See https://github.com/statsmodels/statsmodels/blob/master/statsmodels/nonparametric/bandwidths.py"""
        pass

    def _silverman_bw_rule(self):
        """see same as scott's"""
        pass

    def _optimal_kde(self, **kwargs):
        """
        Hyper-parameter estimation for optimal bandwidth and kernel using
        grid search with cross-validation
        """

        grid = GridSearchCV(KernelDensity(**kwargs),
                            {'bandwidth': self._bw,
                             'kernel': self._kernel},
                            # scoring='mutual_info_score',
                            # scoring=_scoring,
                            # cv=LeaveOneOut(),
                            cv=5
                            )
        return grid

    def _fit(self):
        # print(self._kde)
        self._kde.fit(self._data[:, None])
        self._grid_scores = self._kde.cv_results_
        self._best_params = self._kde.best_params_
        self._optimal_bw = self._kde.best_estimator_.bandwidth
        self._optimal_kernel = self._kde.best_estimator_.kernel
        self._kde = self._kde.best_estimator_

    def density(self, x):
        x = np.asarray(x)
        check_1D_data(x)
        log_dens = self._kde.score_samples(x[:, None])
        return np.exp(log_dens)

    def sample(self, n_samples):
        return self._kde.sample(n_samples)

    def sample2(self, n_samples, rseed=42):
        rng = np.random.RandomState(rseed)
        i = rng.uniform(0, self._data_len, size=n_samples)

    @property
    def grid_scores(self):
        return self._grid_scores

    @property
    def best_params(self):
        '''
        # redundant with current implementation
        if self._best_params is None:
            msg = ("Best parameters only available after fitting")
            raise ValueError(msg)
        '''
        return self._best_params

    @property
    def bandwidth(self):
        return self._optimal_bw

    @property
    def kernel(self):
        return self._optimal_kernel

    def plot_kernels(self):
        # plot available kernels
        # see this instead:
        # https://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html
        kernels = ['cosine', 'epanechnikov',
                   'exponential', 'gaussian', 'linear', 'tophat']
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 7))
        plt_ind = np.arange(6) + 231

        for k, ind in zip(kernels, plt_ind):
            kde_model = KernelDensity(kernel=k)
            kde_model.fit([[0]])
            score = kde_model.score_samples(np.arange(-2, 2, 0.1)[:, None])
            plt.subplot(ind)
            plt.fill(np.arange(-2, 2, 0.1)[:, None], np.exp(score), c='blue')
            plt.title(k)

    def plot_grid_search(self):
        scores_mean = self._grid_scores['mean_test_score']
        scores_mean = np.array(scores_mean).reshape(
            len(self._kernel), len(self._bw))

        scores_std = self._grid_scores['std_test_score']
        scores_std = np.array(scores_std).reshape(
            len(self._kernel), len(self._bw))

        scores_rank = self._grid_scores['rank_test_score']
        scores_rank = np.array(scores_rank).reshape(
            len(self._kernel), len(self._bw))

        fig, ax = plt.subplots(1, 3, figsize=(14, 4))

        for idx, kernel_name in enumerate(self._kernel):
            ax[0].plot(self._bw, scores_mean[idx, :],
                       label=kernel_name + " kernel")
            ax[0].set_xscale('log')
            ax[0].set_yscale('symlog')
            ax[0].set_ylabel('CV Mean Score')
            ax[0].set_xlabel("Bandwidth")
            ax[0].grid('on')

            ax[1].plot(self._bw, scores_std[idx, :],
                       label=kernel_name + " kernel")
            ax[1].set_xscale('log')
            ax[1].set_yscale('log')
            ax[1].set_ylabel('CV Standard Deviation Score')
            ax[1].set_xlabel("Bandwidth")
            ax[1].grid('on')

            ax[2].plot(self._bw, scores_rank[idx, :] / 100,
                       label=kernel_name + " kernel")
            ax[2].set_xscale('log')
            ax[2].set_ylabel('CV Rank Score')
            ax[2].set_xlabel("Bandwidth")
            ax[2].grid('on')

        #handles, labels = ax.get_legend_handles_labels()
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles,
                   labels,
                   loc='center left',
                   bbox_to_anchor=(1.04, 0.5),
                   fancybox=True,
                   borderaxespad=0.1,
                   ncol=1
                   )
        fig.suptitle("Grid Search Scores")
        plt.grid('on')


# Maybe include


def _scoring(estimator, X):
    """
    The cosine, linear and tophat kernels might give a runtime warning due
    to some scores resulting in -inf values. This issue is addressed by
    writing a custom scoring function for GridSearchCV()
    """
    scores = estimator.score_samples(X)
    # Remove -inf
    scores = scores[scores != float('-inf')]
    #scores = scores[np.isfinite(scores)]
    # Return the mean values
    return np.abs(np.mean(scores))


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import scipy.stats as stats

    def make_data(N, f=0.3, rseed=1):
        rand = np.random.RandomState(rseed)
        x = rand.randn(N)
        #x[int(f * N):] += 5
        return x

    def make_data2(N):
        groundtruth = 2.0
        likelihood = stats.norm(loc=0., scale=np.sqrt(groundtruth))
        obs_data = likelihood.rvs(size=N)
        return obs_data

    ###
    ###
    np.random.seed(42)
    N = int(1e3)
    groundtruth = 2.0
    likelihood = stats.norm(loc=0., scale=np.sqrt(groundtruth))
    data = likelihood.rvs(size=N)

    #data = make_data2(1000)

    #kde = KDE(data, bandwidth='auto', kernel='auto')
    #kde = KDE(data, bandwidth='auto', kernel=['gaussian', 'tophat'])
    #kde = KDE(data, bandwidth='auto', kernel=['gaussian', 'epanechnikov'])
    kde = KDE(data, bandwidth='auto', kernel='gaussian')

    best_params = kde.best_params
    print('Optimal params:', best_params)

    grid_scores = kde.grid_scores
    kde.plot_grid_search()
    plt.tight_layout()
    plt.show()

    # print(grid_scores.keys())
    #plot.grid_search(grid_scores, change='n_estimators', kind='bar')

    x = np.linspace(-5, 5, int(1e4))
    density = kde(x)
    kde_samples = kde.sample(int(1e4))

    true_pdf = likelihood.pdf(x)

    sample_mean = np.mean(data)
    sample_median = np.median(data)
    sample_std = np.std(data)
    kde_mean = np.mean(kde_samples)
    kde_median = np.median(kde_samples)
    kde_std = np.std(kde_samples)
    print(f'sample mean: {sample_mean:.4f}')
    print(f'sample median: {sample_median:.4f}')
    print(f'sample std: {sample_std:.4f}')
    print(f'kde mean: {kde_mean:.4f}')
    print(f'kde median: {kde_median:.4f}')
    print(f'kde std: {kde_std:.4f}')

    fig, ax = plt.subplots(1, 1)
    ax.hist(data, density=True, histtype='bar', alpha=0.5, label='hist')
    ax.plot(x, density, color='r', label='kde')
    ax.plot(x, true_pdf, color='k', label='true pdf')
    ax.axvline(kde_mean, ymax=0.3, color='r', label='kde mean')
    ax.axvline(0, ymax=0.3, color='k', ls=':', label='true mean')
    ax.set_ylabel('Density')
    ax.set_xlabel('$x$')
    #ax.axvline(sample_mean, label='data mean')
    ax.legend()
    plt.show()
