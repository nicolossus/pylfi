import inspect

import numpy as np
import pylfi
import xarray as xr

# test 1
chains = [0]
n_samples = 10
parameter_names = ["mu", "sigma"]
idata_posterior = {}

posterior_samples = np.array([[163.83342048, 15.79297546],
                              [161.34952038,  15.58635265],
                              [161.77951193,  14.73673237],
                              [163.57101559,  13.10552356],
                              [163.19327594,  16.70799025],
                              [162.13644339,  15.95927894],
                              [163.3882918,   15.21103736],
                              [162.39004987,  16.31101312],
                              [164.23871528,  15.14406145],
                              [163.46834005,  14.77279184]])


for i, param_name in enumerate(parameter_names):
    idata_posterior[param_name] = (
        ["chain", "draw"], [posterior_samples[:, i]])

idata_coords = {"chain": chains, "draw": np.arange(n_samples, dtype=int)}

# print(idata_posterior)
# print(idata_coords)

posterior = xr.Dataset(idata_posterior, idata_coords)
print(posterior)


posterior = xr.Dataset(
    {"mu": (["chain", "draw"], [[11, 12, 13], [22, 23, 24]]),
     "sd": (["chain", "draw"], [[33, 34, 35], [44, 45, 46]])},
    coords={"draw": [1, 2, 3], "chain": [0, 1]},
)

# print(posterior)
