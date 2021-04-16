#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from hodgkin_huxley import HodgkinHuxley
from matplotlib import gridspec
from spiking_features import SpikingFeatures
from stimulus import constant_stimulus


class HHSimulator:

    def __init__(self, T, dt, stimulus, t_stim_on, stim_duration, feature='spike_rate'):

        available_features = ['n_spikes',
                              'spike_rate',
                              'latency_to_first_spike',
                              'average_AP_overshoot',
                              'average_AHP_depth',
                              'average_AP_width',
                              'accommodation_index']

        self.feature = feature

        if not isinstance(self.feature, str):
            raise TypeError("feature must be given as str")

        if self.feature not in available_features:
            msg = (f"{self.feature} is not an available feature")
            raise ValueError(msg)

        self.T = T
        self.dt = dt
        self.I = stimulus
        self.t_stim_on = t_stim_on
        self.duration = stim_duration
        self.hh = HodgkinHuxley()

    def __call__(self, gbar_K, gbar_Na):
        # run simulation, return summary stats
        self.hh.gbar_K = gbar_K
        self.hh.gbar_Na = gbar_Na
        t, V = self._run()
        features = SpikingFeatures(V, t, self.duration, self.t_stim_on)
        sum_stat = getattr(features, self.feature)
        return sum_stat

    def _run(self):
        self.hh.solve(self.I, self.T, self.dt)
        return self.hh.t, self.hh.V

    def generate_data(self, gbar_K=36., gbar_Na=120.):
        self.hh.gbar_K = gbar_K
        self.hh.gbar_Na = gbar_Na
        t, V = self._run()
        return t, V

    def obs_sumstat(self):
        t, V = self.generate_data()
        features = SpikingFeatures(V, t, self.duration, self.t_stim_on)
        obs_sum_stat = getattr(features, self.feature)
        return obs_sum_stat
