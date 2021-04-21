#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def check_journal_status(is_journal_started):
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
    if not is_journal_started:
        msg = ("Journal unavailable; run an inference scheme first")
        raise RuntimeError(msg)


def check_true_parameter_values(n_parameters, true_parameter_values):
    """Check dtype and shape of true parameter values input.

    Parameters
    ----------
    n_parameters : int
        The number of inferred parameters
    true_parameter_values : list
        List of true parameter values

    Raises
    ------
    TypeError
        If ``true_parameter_values`` is not a list
    ValueError
        If ``true_parameter_values`` is not of length ``n_parameters``
    """
    if not isinstance(true_parameter_values, list):
        msg = "True parameter values must be provided in a list"
        raise TypeError(msg)
    if n_parameters != len(true_parameter_values):
        msg = ("The number of true parameter values in list must "
               "be equal the number of inferred parameters.")
        raise ValueError(msg)


def check_point_estimate_input(point_estimate):
    """
    """
    allowed_metrics = set(['mean', 'median', 'mode'])
    if point_estimate not in allowed_metrics:
        msg = (f"Point estimate metric must be one of {allowed_metrics}.")
        raise ValueError(msg)


def check_grid_input(n_parameters, grid):
    """Check whether custom grid match requirements,

    Parameters
    ----------
    n_parameters : int
        Number of inferred parameters, i.e. the number of panels required
    grid : tuple
        Number of rows and columns

    Raises
    ------
    TypeError
        If ``grid`` is not a tuple of integers
    ValueError
        If ``grid`` tuple contains more than two integers or the grid
        doesn't match the panels required
    """
    if not isinstance(grid, tuple):
        msg = ("'grid' must be provided as a tuple of two integers.")
        raise TypeError(msg)

    if len(grid) != 2:
        msg = ("'grid' tuple must consist of two integers specifying "
               "the number of rows and columns, respectively.")
        raise ValueError(msg)

    if not all(isinstance(e, int) for e in grid):
        msg = ("The number of rows and columns in grid must be "
               "provided as integers.")
        raise TypeError(msg)

    rows, cols = grid

    if rows * cols < n_parameters:
        msg = ("The number of rows times columns is less than "
               "the number of subplots (panels) required.")
        raise ValueError(msg)


def check_plot_style_input(plot_style, usetex):
    """Check whether provided plot configurations meet requirements.

    Parameters
    ----------
    plot_style : str or None
        Plot style; one of 'mpl' (matplotlib), 'sns' (seaborn) or 'default'
    usetex : bool
        True if text should use LaTeX rendering

    Raises
    ------
    ValueError
        If plot style is not recognized.
    TypeError
        If ``usetex`` is not a boolean.
    """
    allowed_styles = set(['pylfi', 'mpl', 'sns'])
    if plot_style is not None:
        if not plot_style in allowed_styles:
            msg = ("Plot style not recognized. "
                   f"Allowed styles: {allowed_styles}.")
            raise ValueError(msg)
    if not isinstance(usetex, bool):
        msg = ("'usetex' must be provided as boolean.")
        raise TypeError(msg)


if __name__ == "__main__":
    check_point_estimate_input('mean')
    check_grid_input(7, (1, 3, 2))
