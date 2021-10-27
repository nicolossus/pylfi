#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_widths

# for features that requires several spikes, check and evaulate to np.inf if too few spikes


class SpikingFeatures:
    """Extract spiking features of a voltage trace.


    The following features are available as class attributes:

    * :attr:`spike_rate`
    * ``latency_to_first_spike``
    * ``average_AP_overshoot``
    * ``average_AHP_depth``
    * ``average_AP_width``
    * ``accommodation_index``

    Features are from Druckmann et al. (2007).

    Parameters
    ----------
    V : array_like
        The voltage array of the voltage trace.
    t : array_like
        The time array of the voltage trace.
    stim_duration : float
        Duration of input stimulus
    t_stim_on : float
        Time of stimulus onset
    threshold : float, optional
        Threshold potential; when depolarization reaches this critical
        level a neuron will initiate an action potential. Default value is
        -55 :math:`mV`.

    Attributes
    ----------
    n_spikes : int
        The number of spikes during stimulus period in voltage trace.
    spike_rate : float
        **Feature:** Action potential firing rate, which is the number of
        action potentials divided by stimulus duration. Typical unit is ``mHz``.
    latency_to_first_spike : float
        **Feature:** Latency to first spike, which is the time between stimulus
        onset and first spike occurrence. Typical unit is ``ms``.
    average_AP_overshoot : float
        **Feature:** Average action potential peak voltage. Average AP
        overshoot is calculated by averaging the absolute peak voltage values
        of all action potentials (spikes). Typical unit is ``mV``.
    average_AHP_depth : float
        **Feature:** Average depth of after hyperpolarization (AHP), i.e.,
        average minimum voltage between action potentials. The average AHP
        depth is obtained by averaging the minimum voltage between two
        consecutive APs. Typical unit is ``mV``.
    average_AP_width : float
        **Feature:** Average action potential width. The average AP width is
        calculated by averaging the width of every AP at the midpoint between
        its onset and its peak. Typical unit is ``mV``.
    accommodation_index : float
        **Feature:** Accommodation index, which is the normalized average
        difference in length of two consecutive interspike intervals (ISIs).

    Notes
    -----
    The basis for extracting the features is the identification of action
    potentials (APs). ``scipy.signal.find_peaks`` is used to find all peaks,
    i.e. APs, greater or equal than the set threshold.

    References
    ----------
    Druckmann, S., Banitt, Y., Gidon, A. A., Schurmann, F., Markram, H., and
    Segev, I. (2007).
    "A novel multiple objective optimization framework for constraining
    conductance-based neuron models by experimental data".
    Frontiers in Neuroscience 1, 7-18. doi:10.3389/neuro.01.1.1.001.2007

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from pylfi.models import HodgkinHuxley, constant_stimulus
    >>> from pylfi.features import SpikingFeatures

    Generate voltage trace:

    >>> T = 120
    >>> dt = 0.01
    >>> I_amp = 0.32
    >>> stimulus = constant_stimulus(I_amp=0.32, T=T, dt=dt, t_stim_on=10,
    ...                              t_stim_off=110, r_soma=40)
    >>> hh.HodgkinHuxley()
    >>> hh.solve(stimulus["I"], T, dt)
    >>> t = hh.t
    >>> V = hh.V

    Initialize spike features extraction:

    >>> threshold = -55  # AP threshold
    >>> stim_duration = stimulus["duration"]
    >>> t_stim_on = stimulus["t_stim_on"]
    >>> features = SpikingFeatures(V, t, stim_duration, t_stim_on, threshold)

    Extract features:

    >>> n_spikes = features.n_spikes  # not strictly a feature
    >>> spike_rate = features.spike_rate
    >>> latency_to_first_spike = features.latency_to_first_spike
    >>> average_AP_overshoot = features.average_AP_overshoot
    >>> average_AHP_depth = features.average_AHP_depth
    >>> average_AP_width = features.average_AP_width
    >>> accommodation_index = features.accommodation_index
    """

    def __init__(self, V, t, stim_duration, t_stim_on, threshold=-55):
        self._V = V
        self._time = t
        self._duration = stim_duration
        self._t_stim_on = t_stim_on
        self._threshold = threshold

        # array with indicies in V and time array
        self._n_indices = len(self._V)
        self._ind_arr = np.linspace(0, self._n_indices, self._n_indices)

        # calls to make spiking attributes available
        self._find_spikes()
        self._spike_widths()

    def _find_spikes(self):
        """Find spikes in voltage trace."""

        # find spikes in trace that are above the specified threshold height
        self._spikes_ind, properties = find_peaks(
            self._V, height=self._threshold)

        # the voltage values at the peak of the spikes
        self._V_spikes_height = properties["peak_heights"]

        # tally the number of spikes in trace
        self._n_spikes = len(self._spikes_ind)

    def _spike_widths(self):
        """Find spike widths and translate to physical units."""

        # obtain data about width of spikes; note that returned widths will be
        # in terms of index positions in voltage array
        spikes_width_data = peak_widths(
            self._V, self._spikes_ind, rel_height=0.5)

        # retrieve left and right interpolated positions specifying horizontal
        # start and end positions of the found spike widths
        left_ips, right_ips = spikes_width_data[2:]

        # membrane potential as interpolation function
        V_interpolate = interp1d(x=self._ind_arr, y=self._V)

        # voltage spike width in terms of physical units (instead of position
        # in array)
        self._V_spike_widths = V_interpolate(
            right_ips) - V_interpolate(left_ips)

        # prepare spike width data for plotting in terms of physical units:
        # time as interpolation function
        time_interpolate = interp1d(x=self._ind_arr, y=self._time)

        # interpolated width positions in terms of physical units
        left_ips_physical = time_interpolate(left_ips)
        right_ips_physical = time_interpolate(right_ips)

        # the height of the contour lines at which the widths where evaluated
        width_heights = spikes_width_data[1]

        # assemble data of contour lines at which the widths where calculated
        self._spikes_width_data_physical = (
            width_heights, left_ips_physical, right_ips_physical)

    @ property
    def n_spikes(self):
        return self._n_spikes

    def spikes_position(self):
        """Array of spike peak positions in terms of indices.

        Returns
        -------
        spikes_position : array_like
            Spike peak positions.
        """
        return self._spikes_ind

    def V_spikes_height(self):
        """Array of voltage values at the peak of the spikes.

        Returns
        -------
        V_spikes_height : array_like
            Peak voltage values.
        """
        return self._V_spikes_height

    def width_lines(self):
        """Data for contour lines at which the widths where calculated.

        Can be used for plotting the located spike widths via e.g.:
            >>> features = SpikingFeatures(V, t, stim_duration, t_stim_on)
            >>> plt.hlines(*features.width_lines)

        Returns
        -------
        width_lines : 3-tuple of ndarrays
            Contour lines.
        """
        return self._spikes_width_data_physical

    def AHP_depth_position(self):
        """Array of positions of after hyperpolarization depths in terms of
        indices.

        Returns
        -------
        AHP_depth_position : array_like
            The positions of the minimum voltage values between consecutive
            action potentials.
        """

        ahp_depth_position = []
        for i in range(self._n_spikes - 1):
            # search for minimum value between two spikes
            spike1_ind = self._spikes_ind[i]
            spike2_ind = self._spikes_ind[i + 1]
            # slice out voltage trace between the spikes
            V_slice = self._V[spike1_ind:spike2_ind]
            # find index of min. value relative to first spike
            min_rel_pos = np.argmin(V_slice)
            # the position in total voltage trace
            min_pos = spike1_ind + min_rel_pos
            ahp_depth_position.append(min_pos)

        return ahp_depth_position

    def ISIs(self):
        """Interspike intervals (ISIs).

        ISI is the time between subsequent action potentials.

        Returns
        -------
        ISIs : array_like
            Interspike intervals.
        """

        ISIs = []
        for i in range(self.n_spikes - 1):
            ISI = self._time[self._spikes_ind[i + 1]] - \
                self._time[self._spikes_ind[i]]
            ISIs.append(ISI)

        return np.array(ISIs)

    # Features
    @ property
    def spike_rate(self):

        if self.n_spikes < 1:
            return np.inf

        return self.n_spikes / self._duration

    @ property
    def latency_to_first_spike(self):

        if self.n_spikes < 1:
            return np.inf

        return self._time[self._spikes_ind[0]] - self._t_stim_on

    @ property
    def average_AP_overshoot(self):

        if self.n_spikes < 1:
            return np.inf

        return np.sum(self._V_spikes_height) / self.n_spikes

    @ property
    def average_AHP_depth(self):

        if self.n_spikes < 3:
            return np.inf

        sum_ahp_depth = sum([np.min(self._V[self._spikes_ind[i]:self._spikes_ind[i + 1]])
                             for i in range(self.n_spikes - 1)])
        avg_ahp_depth = sum_ahp_depth / self._n_spikes
        return avg_ahp_depth

    @ property
    def average_AP_width(self):

        if self.n_spikes < 1:
            return np.inf

        average_AP_width = np.sum(self._V_spike_widths) / self.n_spikes
        return average_AP_width

    @ property
    def accommodation_index(self):

        # Excerpt from Druckmann et al. (2007):
        # "k determines the number of ISIs that will be disregarded in order not
        # to take into account possible transient behavior as observed in
        # Markram et al. (2004). A reasonable value for k is either four ISIs or
        # one-fifth of the total number of ISIs, whichever is the smaller of the
        # two."

        if self.n_spikes < 2:
            return np.inf

        ISIs = self.ISIs()
        k = min(4, int(len(ISIs) / 5))

        A = 0
        for i in range(k + 1, self.n_spikes - 1):
            A += (ISIs[i] - ISIs[i - 1]) / (ISIs[i] + ISIs[i - 1])

        return A / (self.n_spikes - k - 1)


if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib import gridspec
    from pylfi.models import HodgkinHuxley, constant_stimulus

    # plt.style.use('seaborn')
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

    # remove top and right axis from plots
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False

    # simulation parameters
    T = 120.
    dt = 0.01
    I_amp = 0.32  # 0.1 #0.31
    r_soma = 40  # 15

    # input stimulus
    stimulus = constant_stimulus(
        I_amp=I_amp, T=T, dt=dt, t_stim_on=10, t_stim_off=110, r_soma=r_soma)
    # print(stimulus["info"])
    I = stimulus["I"]
    I_stim = stimulus["I_stim"]

    # HH simulation
    hh = HodgkinHuxley()
    hh.solve(I, T, dt)
    t = hh.t
    V = hh.V

    # find spikes
    threshold = -55  # AP threshold
    duration = stimulus["duration"]
    t_stim_on = stimulus["t_stim_on"]

    features = SpikingFeatures(V, t, duration, t_stim_on, threshold)

    # number of spikes
    n_spikes = features.n_spikes
    print(f"{n_spikes=}")

    # spike rate
    spike_rate = features.spike_rate
    print(f"{spike_rate=:.4f} mHz")

    # latency to first spike
    latency_to_first_spike = features.latency_to_first_spike
    print(f"{latency_to_first_spike=:.4f} ms")

    # average AP overshoot
    average_AP_overshoot = features.average_AP_overshoot
    print(f"{average_AP_overshoot=:.4f} mV")

    # average AHP depth
    average_AHP_depth = features.average_AHP_depth
    print(f"{average_AHP_depth=:.4f} mV")

    # average AP width
    average_AP_width = features.average_AP_width
    print(f"{average_AP_width=:.4f} mV")

    # accommodation index
    accommodation_index = features.accommodation_index
    print(f"{accommodation_index=:.4f}")

    # plot voltage trace with features
    spike_pos = features.spikes_position()
    V_spikes_height = features.V_spikes_height()
    width_lines = features.width_lines()
    ahp_depth_pos = features.AHP_depth_position()

    fig = plt.figure(figsize=(8, 6), tight_layout=True, dpi=140)
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    ax = plt.subplot(gs[0])

    # voltage trace
    plt.plot(t, V, lw=1.5, label='Voltage trace')

    # AP overshoot
    plt.plot(t[spike_pos], V[spike_pos], "x",
             ms=7, color='black', label='AP overshoot')

    # AP widths
    plt.hlines(*width_lines, color="red", lw=2, label='AP width')

    # AHP depths
    plt.plot(t[ahp_depth_pos], V[ahp_depth_pos], 'o',
             ms=5, color='indianred', label='AHP depth')

    # latency to first spike
    plt.hlines(hh.V_rest, t_stim_on,
               t[spike_pos[0]], color='black', lw=1.5, ls=":")
    plt.vlines(t[spike_pos[0]], hh.V_rest, V_spikes_height[0],
               color='black', lw=1.5, ls=":", label="Latency to first spike")

    # the marked ISIs are used to compute the accommodation index
    # ISI arrow legend
    plt.plot([], [], color='g', marker=r'$\longleftrightarrow$',
             linestyle='None', markersize=15, label='ISIs')

    # ISI spike 1 -> 2
    plt.vlines(t[spike_pos[0]], V[spike_pos[0]],
               48, color='darkorange', ls='--', label='Spike rate')
    plt.annotate('', xy=(t[spike_pos[0]], 48), xycoords='data',
                 xytext=(t[spike_pos[1]], 48), textcoords='data',
                 arrowprops={'arrowstyle': '<->', 'color': 'g'})
    # ISI spike 2 -> 3
    plt.vlines(t[spike_pos[1]], V[spike_pos[1]],
               48, color='darkorange', ls='--')
    plt.annotate('', xy=(t[spike_pos[1]], 48), xycoords='data',
                 xytext=(t[spike_pos[2]], 48), textcoords='data',
                 arrowprops={'arrowstyle': '<->', 'color': 'g'})
    # ISI spike 3 -> 4
    plt.vlines(t[spike_pos[2]], V[spike_pos[2]],
               48, color='darkorange', lw=1.5, ls='--')
    plt.annotate('', xy=(t[spike_pos[2]], 48), xycoords='data',
                 xytext=(t[spike_pos[3]], 48), textcoords='data',
                 arrowprops={'arrowstyle': '<->', 'color': 'g'})
    # ISI spike 4 -> 5
    plt.vlines(t[spike_pos[3]], V[spike_pos[3]],
               48, color='darkorange', lw=1.5, ls='--')
    plt.annotate('', xy=(t[spike_pos[3]], 48), xycoords='data',
                 xytext=(t[spike_pos[4]], 48), textcoords='data',
                 arrowprops={'arrowstyle': '<->', 'color': 'g'})
    # ISI spike 5 -> 6
    plt.vlines(t[spike_pos[4]], V[spike_pos[4]],
               48, color='darkorange', lw=1.5, ls='--')
    plt.vlines(t[spike_pos[5]], V[spike_pos[5]],
               48, color='darkorange', lw=1.5, ls='--')
    plt.annotate('', xy=(t[spike_pos[4]], 48), xycoords='data',
                 xytext=(t[spike_pos[5]], 48), textcoords='data',
                 arrowprops={'arrowstyle': '<->', 'color': 'g'})

    plt.ylabel('Voltage (mV)')
    plt.ylim(-90, 60)
    ax.set_xticks([])
    ax.set_yticks([-80, -20, 40])
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles,
               labels,
               loc='center left',
               bbox_to_anchor=(1.04, 0.5),
               fancybox=True,
               borderaxespad=0.1,
               ncol=1
               )

    ax = plt.subplot(gs[1])
    plt.plot(t, I_stim, 'k', lw=2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Stimulus (nA)')
    ax.set_xticks([0, np.max(t) / 2, np.max(t)])
    ax.set_yticks([0, np.max(I_stim)])
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))

    fig.suptitle("Spiking Features")
    plt.show()
