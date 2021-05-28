import numpy as np
import scipy.stats as stats


def set_cols(N):
    if N == 1:
        cols = 1
    elif N == 2 or N == 4:
        cols = 2
    else:
        cols = 3

    rows = int(np.ceil(N / cols))
    return rows, cols


def set_grid(N):

    max_cols = 3
    min_cols = 2

    if N <= max_cols:
        return 1, N

    cols = round(np.clip(N**0.5, min_cols, max_cols))
    rows = int(np.ceil(N / cols))
    return rows, cols

    '''
    def in_bounds(val):
        return np.clip(val, min_cols, max_cols)

    if N <= max_cols:
        return 1, N
    ideal = in_bounds(round(N ** 0.5))

    for offset in (0, 1, -1, 2, -2):
        cols = in_bounds(ideal + offset)
        rows, extra = divmod(N, cols)
        if extra == 0:
            return rows, cols
    return N // ideal + 1, ideal
    '''


'''
(1, 1)
(1, 2)
(1, 3)
(2, 2)
(1, 3)
(3, 2)
(3, 2)
(3, 3)
(5, 2)
    if grid is None:

        def in_bounds(val):
            return np.clip(val, min_cols, max_cols)

        if n_items <= max_cols:
            return 1, n_items
        ideal = in_bounds(round(n_items ** 0.5))

        for offset in (0, 1, -1, 2, -2):
            cols = in_bounds(ideal + offset)
            rows, extra = divmod(n_items, cols)
            if extra == 0:
                return rows, cols
        return n_items // ideal + 1, ideal
'''


def check_grid_input(grid):
    if not isinstance(grid, tuple):
        raise TypeError("a")

    rows, cols = grid
    if rows * cols < n_items:
        raise ValueError(
            "The number of rows times columns is less than the number of subplots")


print(set_cols(1))
print(set_cols(2))
print(set_cols(3))
print(set_cols(4))
print(set_cols(3))
print(set_cols(5))
print(set_cols(6))
print(set_cols(7))
print(set_cols(10))
print()
print(set_grid(1))
print(set_grid(2))
print(set_grid(3))
print(set_grid(4))
print(set_grid(3))
print(set_grid(5))
print(set_grid(6))
print(set_grid(7))
print(set_grid(10))
