#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def simple_stimulus(t):
    t_stim_on = 10
    t_stim_off = 110
    I_amp = 10
    if t_stim_on <= t <= t_stim_off:
        I = I_amp
    else:
        I = 0
    return I


def constant_stimulus(I_amp=0.1, T=120, dt=0.01, t_stim_on=10, t_stim_off=110, r_soma=15):
    """
    Consctructs array I of input stimulus with

            I(t) = I_amp / A_soma    for t_stim_on <= t <= t_stim_off
            I(t) = 0.0               else.

    (i.e. current pulse of length t_stim_off-t_stim_on).

    Parameters
    ----------
    I_amp : float
        Amplitude of stimulus in nA
    T : float
        End time in ms
    dt : float
        Time step
    t_stim_on : float
        Time when stimulus is turned on
    t_stim_off: float
        Time when stimulus is turend off
    r_soma : float
        Radius of soma in μm

    Notes
    -----
    The typical values of the Hodgkin-Huxley model's parameters dictates a
    stimulus with units μA/cm**2 (current per unit area). It is often more
    natural to express the input stimulus in units nA, and the  soma of a
    neuron can vary from 4 to 100 micrometers in diameter (2 to 50 micrometers
    in radius assuming spherical shape). The stimulus parameters `I_amp` and
    `r_soma` are therefore expected to be given in units nA and μm,
    respectively. Necessary unit conversions are handled internally.

    Returns
    -------
    stimulus: dict
       A dictionary containing properties of the specified stimulus.

       * 'I'
            Array of current per unit area [μA/cm**2]
       * 'I_stim'
            Array of current [nA]
       * 'A_soma'
            Surface area of soma [cm**2]
       * 't'
            Time array [ms]
       * 't_stim_on'
            Time when stimulus is turned on [ms]
       * 't_stim_off'
            Time when stimulus is turned off [ms]
       * 'duration'
            Duration of stimulus [ms]
       * 'info'
            String with information about the dictionary
    """

    A_soma = np.pi * ((r_soma * 1e-4) ** 2)  # [cm**2]
    time = np.arange(0, T + dt, dt)

    I_stim = np.zeros_like(time)
    stim_on_ind = int(np.round(t_stim_on / dt))
    stim_off_ind = int(np.round(t_stim_off / dt))

    I_stim[stim_on_ind:stim_off_ind] = I_amp
    I = I_stim.copy()
    I = (I * 1e-3) / A_soma    # current per unit area

    stimulus = dict()
    stimulus["I"] = I
    stimulus["I_stim"] = I_stim
    stimulus["A_soma"] = A_soma
    stimulus["t"] = time
    stimulus["t_stim_on"] = t_stim_on
    stimulus["t_stim_off"] = t_stim_off
    stimulus["duration"] = t_stim_off - t_stim_on
    info = ("Stimulus info\n"
            "-------------\n"
            "Stimulus type: step current\n"
            "Available dictionary keys and values:\n"
            "'I': (array) current per unit area [μA/cm**2]\n"
            "'I_stim': (array) input current [nA]\n"
            "'A_soma': (float) soma area [cm**2]\n"
            "'t': (array) time points\n"
            "'t_stim_on': (float) time when stimulus is turned on\n"
            "'t_stim_off': (float) time when stimulus is turned off\n"
            "'duration': (float) duration of stimulus")
    stimulus["info"] = info

    return stimulus


def equilibrating_stimulus(I_amp=0.1, T=200, dt=0.01, t_stim_on=10, t_stim_off=110, r_soma=15):
    """
    Consctructs array I of input stimulus with

            I(t) = I_amp / A_soma    for t_stim_on <= t <= t_stim_off
            I(t) = 0.0               else.

    (i.e. current pulse of length t_stim_off-t_stim_on).

    Parameters
    ----------
    I_amp : float
        Amplitude of stimulus in nA
    T : float
        End time in ms
    dt : float
        Time step
    t_stim_on : float
        Time when stimulus is turned on
    t_stim_off: float
        Time when stimulus is turend off
    r_soma : float
        Radius of soma in μm

    Notes
    -----
    The typical values of the Hodgkin-Huxley model's parameters dictates a
    stimulus with units μA/cm**2 (current per unit area). It is often more
    natural to express the input stimulus in units nA, and the  soma of a
    neuron can vary from 4 to 100 micrometers in diameter (2 to 50 micrometers
    in radius assuming spherical shape). The stimulus parameters `I_amp` and
    `r_soma` are therefore expected to be given in units nA and μm,
    respectively. Necessary unit conversions are handled internally.

    Returns
    -------
    stimulus: dict
       A dictionary containing properties of the specified stimulus.

       * 'I'
            Array of current per unit area [μA/cm**2]
       * 'I_stim'
            Array of current [nA]
       * 'A_soma'
            Surface area of soma [cm**2]
       * 't'
            Time array [ms]
       * 't_stim_on'
            Time when stimulus is turned on [ms]
       * 't_stim_off'
            Time when stimulus is turned off [ms]
       * 'duration'
            Duration of stimulus [ms]
       * 'info'
            String with information about the dictionary
    """

    A_soma = np.pi * ((r_soma * 1e-4) ** 2)  # [cm**2]
    time = np.arange(0, T + dt, dt)

    I_stim = np.zeros_like(time)

    for stim_on, stim_off in zip(t_stim_on, t_stim_off):
        stim_on_ind = int(np.round(stim_on / dt))
        stim_off_ind = int(np.round(stim_off / dt))
        I_stim[stim_on_ind:stim_off_ind] = I_amp

    I = I_stim.copy()
    I = (I * 1e-3) / A_soma    # current per unit area

    stimulus = dict()
    stimulus["I"] = I
    stimulus["I_stim"] = I_stim
    stimulus["A_soma"] = A_soma
    stimulus["t"] = time
    stimulus["t_stim_on"] = t_stim_on
    stimulus["t_stim_off"] = t_stim_off
    stimulus["duration"] = t_stim_off[-1] - t_stim_on[-1]
    info = ("Stimulus info\n"
            "-------------\n"
            "Stimulus type: step current\n"
            "Available dictionary keys and values:\n"
            "'I': (array) current per unit area [μA/cm**2]\n"
            "'I_stim': (array) input current [nA]\n"
            "'A_soma': (float) soma area [cm**2]\n"
            "'t': (array) time points\n"
            "'t_stim_on': (float) time when stimulus is turned on\n"
            "'t_stim_off': (float) time when stimulus is turned off\n"
            "'duration': (float) duration of stimulus")
    stimulus["info"] = info

    return stimulus


if __name__ == "__main__":
    stimulus = constant_stimulus()
    print(stimulus['info'])
