#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# ignore overflow warnings; occurs with certain np.exp() evaluations
np.warnings.filterwarnings('ignore', 'overflow')


class ODEsNotSolved(Exception):
    """Failed attempt at accessing solutions.

    A call to the ODE systems solve method must be
    carried out before the solution properties
    can be used.
    """
    pass


class HodgkinHuxley:
    r"""Class for representing the Hodgkin-Huxley model.

    The Hodgkin–Huxley model describes how action potentials in neurons are
    initiated and propagated. From a biophysical point of view, action
    potentials are the result of currents that pass through ion channels in
    the cell membrane. In an extensive series of experiments on the giant axon
    of the squid, Hodgkin and Huxley succeeded to measure these currents and
    to describe their dynamics in terms of differential equations.

    All model parameters can be accessed (get or set) as class attributes.
    Solutions are available as class attributes after calling the class method
    :meth:`solve`.

    Parameters
    ----------
    V_rest : :obj:`float`
        Resting potential of neuron in units :math:`mV`, default=-65.0.
    Cm : :obj:`float`
        Membrane capacitance in units :math:`\mu F/cm^2`, default=1.0.
    gbar_K : :obj:`float`
        Potassium conductance in units :math:`mS/cm^2`, default=36.0.
    gbar_Na : :obj:`float`
        Sodium conductance in units :math:`mS/cm^2`, default=120.0.
    gbar_L : :obj:`float`
        Leak conductance in units :math:`mS/cm^2`, default=0.3.
    E_K : :obj:`float`
        Potassium reversal potential in units :math:`mV`, default=-77.0.
    E_Na : :obj:`float`
        Sodium reversal potential in units :math:`mV`, default=50.0.
    E_L : :obj:`float`
        Leak reversal potential in units :math:`mV`, default=-54.4.

    Attributes
    ----------
    V_rest : :obj:`float`
        **Model parameter:** Resting potential.
    Cm : :obj:`float`
        **Model parameter:** Membrane capacitance.
    gbar_K : :obj:`float`
        **Model parameter:** Potassium conductance.
    gbar_Na : :obj:`float`
        **Model parameter:** Sodium conductance.
    gbar_L : :obj:`float`
        **Model parameter:** Leak conductance.
    E_K : :obj:`float`
        **Model parameter:** Potassium reversal potential.
    E_Na : :obj:`float`
        **Model parameter:** Sodium reversal potential.
    E_L : :obj:`float`
        **Model parameter:** Leak reversal potential.
    t : :term:`ndarray`
        **Solution:** Array of time points ``t``.
    V : :term:`ndarray`
        **Solution:** Array of voltage values ``V`` at ``t``.
    n : :term:`ndarray`
        **Solution:** Array of state variable values ``n`` at ``t``.
    m : :term:`ndarray`
        **Solution:** Array of state variable values ``m`` at ``t``.
    h : :term:`ndarray`
        **Solution:** Array of state variable values ``h`` at ``t``.

    Notes
    -----
    Default parameter values as given by Hodgkin and Huxley (1952).

    References
    ----------
    Hodgkin, A. L., Huxley, A.F. (1952).
    "A quantitative description of membrane current and its application
    to conduction and excitation in nerve".
    J. Physiol. 117, 500-544.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from pylfi.models import HodgkinHuxley

    Initialize the Hodgkin-Huxley system; model parameters can either be set in
    the constructor or accessed as class attributes:

    >>> hh = HodgkinHuxley(V_rest=-70)
    >>> hh.gbar_K = 36

    The simulation parameters needed are the simulation time ``T``, the time
    step ``dt``, and the input ``stimulus``, the latter either as a callable or
    ndarray with `shape=(int(T/dt)+1,)`:

    >>> T = 50.
    >>> dt = 0.025
    >>> def stimulus(t):
    ...    return 10 if 10 <= t <= 40 else 0

    The system is solved by calling the class method ``solve`` and the solutions
    can be accessed as class attributes:

    >>> hh.solve(stimulus, T, dt)
    >>> t = hh.t
    >>> V = hh.V

    >>> plt.plot(t, V)
    >>> plt.xlabel('Time [ms]')
    >>> plt.ylabel('Membrane potential [mV]')
    >>> plt.show()

    .. plot::

       import matplotlib.pyplot as plt
       import numpy as np
       plt.hist(np.random.randn(1000), 20)

    another

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: False

        import matplotlib.pyplot as plt
        from pylfi.models import HodgkinHuxley

        hh = HodgkinHuxley(V_rest=-70)
        T = 50.
        dt = 0.025

        def stimulus(t):
            return 10 if 10 <= t <= 40 else 0

        hh.solve(stimulus, T, dt)
        t = hh.t
        V = hh.V

        plt.plot(t, V)
        plt.xlabel('Time [ms]')
        plt.ylabel('Membrane potential [mV]')
        plt.show()
    """

    def __init__(self, V_rest=-65., Cm=1., gbar_K=36., gbar_Na=120.,
                 gbar_L=0.3, E_K=-77., E_Na=50., E_L=-54.4):

        # Hodgkin-Huxley model parameters
        self._V_rest = V_rest      # resting potential [mV]
        self._Cm = Cm              # membrane capacitance [μF/cm**2]
        self._gbar_K = gbar_K      # potassium conductance [mS/cm**2]
        self._gbar_Na = gbar_Na    # sodium conductance [mS/cm**2]
        self._gbar_L = gbar_L      # leak coductance [mS/cm**2]
        self._E_K = E_K            # potassium reversal potential [mV]
        self._E_Na = E_Na          # sodium reversal potential [mV]
        self._E_L = E_L            # leak reversal potential [mV]

    def __call__(self, t, y):
        r"""RHS of the Hodgkin-Huxley ODEs.

        Parameters
        ----------
        t : float
            The time point.
        y : tuple of floats
            A tuple of the state variables, ``y = (V, n, m, h)``.
        """

        V, n, m, h = y
        dVdt = (self.I(t) - self._gbar_K * (n**4) * (V - self._E_K) -
                self._gbar_Na * (m**3) * h * (V - self._E_Na) -
                self._gbar_L * (V - self._E_L)) / self._Cm
        dndt = self._alpha_n(V) * (1 - n) - self._beta_n(V) * n
        dmdt = self._alpha_m(V) * (1 - m) - self._beta_m(V) * m
        dhdt = self._alpha_h(V) * (1 - h) - self._beta_h(V) * h
        return [dVdt, dndt, dmdt, dhdt]

    # K channel kinetics
    def _alpha_n(self, V):
        return 0.01 * (V + 55.) / (1 - np.exp(-(V + 55.) / 10.))

    def _beta_n(self, V):
        return 0.125 * np.exp(-(V + 65) / 80.)

    # Na channel kinetics (activating)
    def _alpha_m(self, V):
        return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10.))

    def _beta_m(self, V):
        return 4 * np.exp(-(V + 65) / 18.)

    # Na channel kinetics (inactivating)
    def _alpha_h(self, V):
        return 0.07 * np.exp(-(V + 65) / 20.)

    def _beta_h(self, V):
        return 1. / (1 + np.exp(-(V + 35) / 10.))

    # steady-states and time constants
    def _n_inf(self, V):
        return self._alpha_n(V) / (self._alpha_n(V) + self._beta_n(V))

    def _tau_n(self, V):
        return 1. / (self._alpha_n(V) + self._alpha_n(V))

    def _m_inf(self, V):
        return self._alpha_m(V) / (self._alpha_m(V) + self._beta_m(V))

    def _tau_m(self, V):
        return 1. / (self._alpha_m(V) + self._alpha_m(V))

    def _h_inf(self, V):
        return self._alpha_h(V) / (self._alpha_h(V) + self._beta_h(V))

    def _tau_h(self, V):
        return 1. / (self._alpha_h(V) + self._alpha_h(V))

    @property
    def _initial_conditions(self):
        """Default Hodgkin-Huxley model initial conditions."""
        n0 = self._n_inf(self.V_rest)
        m0 = self._m_inf(self.V_rest)
        h0 = self._h_inf(self.V_rest)
        return (self.V_rest, n0, m0, h0)

    def solve(self, stimulus, T, dt, y0=None, **kwargs):
        r"""Solve the Hodgkin-Huxley equations.

        The equations are solved on the interval ``(0, T]`` and the solutions
        evaluted at a given interval. The solutions are not returned, but
        stored as class attributes.

        If multiple calls to solve are made, they are treated independently,
        with the newest one overwriting any old solution data.

        Parameters
        ----------
        stimulus : :obj:`callable` or :term:`ndarray`, shape=(int(T/dt)+1,)
            Input stimulus in units :math:`\mu A/cm^2`. If callable, the call
            signature must be ``(t)``.
        T : :obj:`float`
            End time in milliseconds (:math:`ms`).
        dt : :obj:`float`
            Time step where solutions are evaluated.
        y0 : :term:`array_like`, shape=(4,)
            Initial state of state variables ``V``, ``n``, ``m``, ``h``. If None,
            the default Hodgkin-Huxley model's initial conditions will be used;
            :math:`y_0 = (V_0, n_0, m_0, h_0) = (V_{rest}, n_\infty(V_0), m_\infty(V_0), h_\infty(V_0))`.
        **kwargs
            Arbitrary keyword arguments are passed along to
            :obj:`scipy.integrate.solve_ivp`.

        Notes
        -----
        The ODEs are solved numerically using the function :obj:`scipy.integrate.solve_ivp`.

        If ``stimulus`` is passed as an array, it and the time array, defined by
        ``T`` and ``dt``, will be used to create an interpolation function via
        :obj:`scipy.interpolate.interp1d`.

        Credits to supervisor Joakim Sundnes for helping unravel the following.

        ``solve_ivp`` is an ODE solver with adaptive step size. If the keyword
        argument ``first_step`` is not specified, the solver will empirically
        select an initial step size with the function ``select_initial_step``
        (found here https://github.com/scipy/scipy/blob/master/scipy/integrate/_ivp/common.py#L64).

        This function calculates two proposals and returns the smallest. It first
        calculates an intermediate proposal, ``h0``, that is based on the initial
        condition (``y0``) and the ODE's RHS evaluated for the initial condition
        (``f0``). For the standard Hodgkin-Huxley model, however, this estimated
        step size will be very large due to unfortunate circumstances (because ``norm(y0) > 0``
        while ``norm(f0) ~= 0``). Since ``h0`` only is an intermediate calculation,
        it is not used or returned by the solver. However, it is used to calculate
        the next proposal, ``h1``, by calling the RHS. Normally, this procedure
        poses no problem, but can fail if an object with a limited interval is
        present in the RHS, such as an ``interp1d`` object.

        In the case of the standard Hodgkin-Huxley model, one might be tempted
        to pass the stimulus as an array to the solver. In order for ``solve_ivp``
        to be able to evaluate the stimulus, it must be passed as a callable or
        constant. Thus, if an array is passed to the solver, an interpolation
        function must be created, in this implementation done with ``interp1d``,
        for ``solve_ivp`` to be able to evaluate it. For the reasons explained
        above, the program will hence terminate unless the ``first_step`` keyword
        is specified and is set to a sufficiently small value. In this
        implementation, ``first_step=dt`` is already set in ``solve_ivp``.

        The ``solve_ivp`` keyword ``max_step`` should be considered to be specified
        for stimuli over short time spans, in order to ensure that the solver
        does not step over them.

        Note that ``first_step`` still needs to specified even if ``max_step`` is.
        ``select_initial_step`` will be called regardless if ``first_step`` is not
        specified, and the calls for calculating h1 will be done before checking
        whether ``h0`` is larger than than the max allowed step size or not. Thus
        will only specifying ``max_step`` still result in program termination if
        ``stimulus`` is passed as an array. (Will not be a problem in this
        implementation since ``first_step`` is already specified.)
        """

        # error-handling
        if not isinstance(dt, (int, float)):
            msg = (f"{dt=}".split('=')[0]
                   + " must be given as an int or float")
            raise TypeError(msg)

        if dt <= 0:
            msg = ("dt > 0 is required")
            raise ValueError(msg)

        if not isinstance(T, (int, float)):
            msg = (f"{T=}".split('=')[0]
                   + " must be given as an int or float")
            raise TypeError(msg)

        if T <= 0:
            msg = ("T > 0 is required")
            raise ValueError(msg)

        # times at which to store the computed solutions
        t_eval = np.arange(0, T + dt, dt)

        if y0 is None:
            # use default HH initial conditions
            y0 = self._initial_conditions

        # handle the passed stimulus
        if callable(stimulus):
            self.I = stimulus
        elif isinstance(stimulus, np.ndarray):
            if not stimulus.shape == t_eval.shape:
                msg = ("stimulus numpy.ndarray must have shape (int(T/dt)+1)")
                raise ValueError(msg)
            # Interpolate stimulus
            self.I = interp1d(x=t_eval, y=stimulus)  # linear spline
        else:
            msg = ("'stimulus' must be either a callable function of t "
                   "or a numpy.ndarray of shape (int(T/dt)+1)")
            raise ValueError(msg)

        # solve HH ODEs
        solution = solve_ivp(self, t_span=(0, T), y0=y0,
                             t_eval=t_eval, first_step=dt, **kwargs)

        # store solutions
        self._time = solution.t
        self._V = solution.y[0]
        self._n = solution.y[1]
        self._m = solution.y[2]
        self._h = solution.y[3]

    # getters and setters
    @property
    def V_rest(self):
        return self._V_rest

    @V_rest.setter
    def V_rest(self, V_rest):
        if not isinstance(V_rest, (int, float)):
            msg = (f"{V_rest=}".split('=')[0]
                   + " must be set as an int or float")
            raise TypeError(msg)
        self._V_rest = V_rest

    @property
    def Cm(self):
        return self._Cm

    @Cm.setter
    def Cm(self, Cm):
        if not isinstance(Cm, (int, float)):
            msg = (f"{Cm=}".split('=')[0]
                   + " must be set as an int or float")
            raise TypeError(msg)
        self._Cm = Cm

    @property
    def gbar_K(self):
        return self._gbar_K

    @gbar_K.setter
    def gbar_K(self, gbar_K):
        if not isinstance(gbar_K, (int, float)):
            msg = (f"{gbar_K=}".split('=')[0]
                   + " must be set as an int or float")
            raise TypeError(msg)
        self._gbar_K = gbar_K

    @property
    def gbar_Na(self):
        return self._gbar_Na

    @gbar_Na.setter
    def gbar_Na(self, gbar_Na):
        if not isinstance(gbar_Na, (int, float)):
            msg = (f"{gbar_Na=}".split('=')[0]
                   + " must be set as an int or float")
            raise TypeError(msg)
        self._gbar_Na = gbar_Na

    @property
    def gbar_L(self):
        return self._gbar_L

    @gbar_L.setter
    def gbar_L(self, gbar_L):
        if not isinstance(gbar_L, (int, float)):
            msg = (f"{gbar_L=}".split('=')[0]
                   + " must be set as an int or float")
            raise TypeError(msg)
        self._gbar_L = gbar_L

    @property
    def E_K(self):
        return self._E_K

    @E_K.setter
    def E_K(self, E_K):
        if not isinstance(E_K, (int, float)):
            msg = (f"{E_K=}".split('=')[0]
                   + " must be set as an int or float")
            raise TypeError(msg)
        self._E_K = E_K

    @property
    def E_Na(self):
        return self._E_Na

    @E_Na.setter
    def E_Na(self, E_Na):
        if not isinstance(E_Na, (int, float)):
            msg = (f"{E_Na=}".split('=')[0]
                   + " must be set as an int or float")
            raise TypeError(msg)
        self._E_Na = E_Na

    @property
    def E_L(self):
        return self._E_L

    @E_L.setter
    def E_L(self, E_L):
        if not isinstance(E_L, (int, float)):
            msg = (f"{E_L=}".split('=')[0]
                   + " must be set as an int or float")
            raise TypeError(msg)
        self._E_L = E_L

    @ property
    def t(self):
        try:
            return self._time
        except AttributeError as e:
            raise ODEsNotSolved("Missing call to solve. No solution exists.")

    @ property
    def V(self):
        try:
            return self._V
        except AttributeError:
            raise ODEsNotSolved("Missing call to solve. No solution exists.")

    @ property
    def n(self):
        try:
            return self._n
        except AttributeError:
            raise ODEsNotSolved("Missing call to solve. No solution exists.")

    @ property
    def m(self):
        try:
            return self._m
        except AttributeError:
            raise ODEsNotSolved("Missing call to solve. No solution exists.")

    @ property
    def h(self):
        try:
            return self._h
        except AttributeError:
            raise ODEsNotSolved("Missing call to solve. No solution exists.")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from pylfi.models import HodgkinHuxley

    # Initialize the Hodgkin-Huxley system; model parameters can either be
    # set in the constructor or accessed as class attributes:
    hh = HodgkinHuxley(V_rest=-70)
    hh.gbar_K = 36

    # The simulation parameters needed are the simulation time ``T``, the time
    # step ``dt``, and the input ``stimulus``, the latter either as a
    # callable or ndarray with `shape=(int(T/dt)+1,)`:
    T = 50.
    dt = 0.025

    def stimulus(t):
        return 10 if 10 <= t <= 40 else 0

    # The system is solved by calling the class method ``solve`` and the
    # solutions can be accessed as class attributes:
    hh.solve(stimulus, T, dt)
    t = hh.t
    V = hh.V

    plt.plot(t, V)
    plt.xlabel('Time [ms]')
    plt.ylabel('Membrane potential [mV]')
    plt.show()
