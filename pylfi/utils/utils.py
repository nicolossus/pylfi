#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def set_plot_style():
    """Set plot style"""
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
