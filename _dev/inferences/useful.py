def tune(scale, acc_rate):
    """
    tune: bool
        Flag for tuning. Defaults to True.
    tune_interval: int
        The frequency of tuning. Defaults to 100 iterations.

    Module from pymc3

    Tunes the scaling parameter for the proposal distribution
    according to the acceptance rate over the last tune_interval:

    Rate    Variance adaptation
    ----    -------------------
    <0.001        x 0.1
    <0.05         x 0.5
    <0.2          x 0.9
    >0.5          x 1.1
    >0.75         x 2
    >0.95         x 10
    """
    if acc_rate < 0.001:
        # reduce by 90 percent
        return scale * 0.1
    elif acc_rate < 0.05:
        # reduce by 50 percent
        return scale * 0.5
    elif acc_rate < 0.2:
        # reduce by ten percent
        return scale * 0.9
    elif acc_rate > 0.95:
        # increase by factor of ten
        return scale * 10.0
    elif acc_rate > 0.75:
        # increase by double
        return scale * 2.0
    elif acc_rate > 0.5:
        # increase by ten percent
        return scale * 1.1

    return scale


if not 0:
    print('h')

print(not 0)
