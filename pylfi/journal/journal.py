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

plt.rc('text', usetex=True)


class Journal:
    r""" Journal class.

    Journal with results and information created by the run of
    inference schemes.
    """

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
        # simulator,
        # stat_calc,
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
        #self._simulator = simulator
        #self._stat_calc = stat_calc
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

    def thetas_pred(self, size=50):
        """Parameters drawn from posterior predictive distribution"""
        df = self.df
        idxs = np.random.randint(0, len(df.index), size)
        return df.iloc[idxs].to_numpy()

    def compute_rmspe(self, theta_true, theta_pred):
        """Root mean square percentage error (RMSPE)"""
        rmspe = np.sqrt(
            np.mean(
                np.square(
                    (theta_true - theta_pred) / theta_true)
            )
        )
        return rmspe * 100

    def compute_hdi(self, theta, hdi_prob):
        """Highest (posterior) density interval"""
        theta = theta.flatten()
        n = len(theta)
        theta = np.sort(theta)
        interval_idx_inc = int(np.floor(hdi_prob * n))
        n_intervals = n - interval_idx_inc
        interval_width = np.subtract(theta[interval_idx_inc:],
                                     theta[:n_intervals],
                                     dtype=np.float_
                                     )
        min_idx = np.argmin(interval_width)
        hdi_min = theta[min_idx]
        hdi_max = theta[min_idx + interval_idx_inc]
        return (hdi_min, hdi_max)

    def compute_point_est(self, theta, density, point_estimate):
        """Compute point estimate"""
        if point_estimate == "mean":
            p_est = theta.mean()
            handle = "mean"
        elif point_estimate == "median":
            p_est = np.median(theta)
            handle = "median"
        elif point_estimate == "map":
            idx = np.argmax(density)
            p_est = theta[idx]
            handle = "MAP"
        return p_est, handle

    def plot_prior(
        self,
        theta_name,
        x,
        color='C0',
        facecolor='lightblue',
        alpha=0.5,
        ax=None,
        **kwargs
    ):
        idx = np.where(self.df.columns.to_numpy() == theta_name)
        prior = self._priors[idx[0][0]]
        prior.plot_prior(x, ax=ax, **kwargs)

    def plot_posterior(
        self,
        theta_name,
        hdi_prob=0.95,
        point_estimate="map",
        theta_true=None,
        ax=None
    ):
        if ax is None:
            ax = plt.gca()

        if self._df_plot_exist:
            idx = np.where(self.df.columns.to_numpy() == theta_name)
            df = self._df_plot
            theta_name = self._df_plot.columns[idx][0]
        else:
            df = self.df

        xdata = df[theta_name].to_numpy()

        if theta_true is not None:
            theta_true_ary = np.ones(len(xdata.flatten())) * theta_true
            rmspe = self.compute_rmspe(theta_true_ary, xdata)

            sns.kdeplot(
                data=df,
                x=theta_name,
                color='C0',
                label=f"Posterior RMSPE: {rmspe:.2f}\%",
                ax=ax
            )

            ax.axvline(
                theta_true,
                ymax=0.3,
                color='C3',
                label=fr"$\theta_\mathrm{{true}}: {theta_true}$"
            )

        else:
            sns.kdeplot(
                data=df,
                x=x,
                color='C0',
                label="Posterior",
                ax=ax
            )

        kdeline = ax.lines[0]
        xs = kdeline.get_xdata()
        ys = kdeline.get_ydata()

        p_est, p_est_handle = self.compute_point_est(xs, ys, point_estimate)

        hdi_min, hdi_max = self.compute_hdi(xdata, hdi_prob=hdi_prob)

        ax.vlines(
            p_est,
            0,
            np.interp(p_est, xs, ys),
            color='b',
            ls=':',
            label=fr"$\hat{{\theta}}_\mathrm{{{p_est_handle}}}: {p_est:.3f}$"
        )

        ax.fill_between(
            xs,
            0,
            ys,
            facecolor='lightblue',
            alpha=0.3
        )

        ax.fill_between(
            xs,
            0,
            ys,
            where=(hdi_min <= xs) & (xs <= hdi_max),
            interpolate=True,
            facecolor='steelblue',
            alpha=0.3,
            label=f"{hdi_prob*100}\% HDI: [{hdi_min:.3f}, {hdi_max:.3f}]"
        )

        handles, labels = ax.get_legend_handles_labels()

        ax.set(yticks=[])

        ax.legend(
            handles,
            labels,
            loc='center left',
            bbox_to_anchor=(1.04, 0.5),
            fancybox=True,
            borderaxespad=0.1,
            ncol=1,
            frameon=False
        )

    def plot_joint(
        self,
        theta1_name,
        theta2_name,
        theta1_true=None,
        theta2_true=None,
        levels=6,
        alpha=0.3,
        height=4,
        **kwargs
    ):
        if self._df_plot_exist:
            df = self._df_plot
            idx1 = np.where(self.df.columns.to_numpy() == theta1_name)
            idx2 = np.where(self.df.columns.to_numpy() == theta2_name)
            theta1_name = self._df_plot.columns[idx1][0]
            theta2_name = self._df_plot.columns[idx2][0]
        else:
            df = self.df

        g = sns.jointplot(
            data=df,
            x=theta1_name,
            y=theta2_name,
            kind="kde",
            fill=True,
            height=height,
            **kwargs
        )

        g.plot_joint(
            sns.kdeplot,
            color="k",
            levels=levels,
            alpha=alpha
        )

        if theta1_true is not None and theta2_true is not None:
            g.ax_joint.plot([theta1_true], [theta2_true], 'ro')
            g.ax_joint.axvline(theta1_true, color='r', ls=":")
            g.ax_joint.axhline(theta2_true, color='r', ls=":")
            g.ax_marg_x.axvline(theta1_true, color='r')
            g.ax_marg_y.axhline(theta2_true, color='r')

        return g

    def heatmap(
        self,
        measure,
        cmap="coolwarm",
        vmin=None,
        vmax=None,
        ax=None,
        **kwargs
    ):

        if ax is None:
            ax = plt.gca()

        if self._df_plot_exist:
            df = self._df_plot
        else:
            df = self.df

        if measure == 'cov':
            res = df.cov()
        elif measure == 'corr':
            res = df.corr()
        else:
            raise ValueError(f'Unrecognized measure: {measure}')

        sns.heatmap(
            res,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            **kwargs
        )

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

        Function from ABCpy source code.

        Parameters
        ----------
        filename: string
            the location of the file to store the current object to.
        """

        with open(filename, 'wb') as output:
            pickle.dump(self, output, -1)

    @classmethod
    def load(cls, filename):
        """This method reads a saved journal from disk an returns it as an object.

        Function from ABCpy source code.

        Notes
        -----
        To store a journal use Journal.save(filename).
        Parameters
        ----------
        filename: string
            The string representing the location of a file

        Returns
        -------
        abcpy.output.Journal
            The journal object serialized in <filename>
        Example
        --------
        >>> jnl = Journal.load('example_output.jnl')
        """

        with open(filename, 'rb') as input:
            journal = pickle.load(input)

        return journal


if __name__ == "__main__":
    import pylfi
    filename = 'hh_rej_normal_best_posterior_org.jnl'
    journal = pylfi.Journal.load(filename)

    fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
    #journal.plot_joint('gbarK', 'gbarNa', 36., 120.)
    #journal.plot_prior("gbarK", np.linspace(30, 40, 1000))
    journal.plot_posterior("gbarNa", point_estimate='map', theta_true=120.,)
    plt.show()
