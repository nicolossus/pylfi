import numpy as np


def distance(s_sim, s_obs, weight=1., scale=1.):

    if isinstance(s_sim, (int, float)):
        s_sim = [s_sim]
    if isinstance(s_obs, (int, float)):
        s_obs = [s_obs]

    s_sim = np.asarray(s_sim, dtype=float)
    s_obs = np.asarray(s_obs, dtype=float)

    q = np.sqrt(weight) * (s_sim - s_obs) / scale
    dist = np.linalg.norm(q, ord=2)

    return dist


def gaussian_kernel(d, h):
    """Gaussian smoothing kernel function"""
    # return 1 / (h * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (d * d) / (h * h))
    return 1 / h * np.exp(-0.5 * (d * d) / (h * h))


def epkov_kernel(d, h):
    """Epanechnikov smoothing kernel function"""
    # return 0.75 / h * (1.0 - (d * d) / (h * h)) * (d < h)
    return 1 / h * (1.0 - (d * d) / (h * h)) * (d < h)


s_sim_pp = np.array([[2.2, 100], [1.8, 105], [2.1, 97], [1.9, 98]])
scale = np.std(s_sim_pp, axis=0)

s_obs = [2.0, 98]

s_sim = [2.1, 102]
dist1 = distance(s_sim, s_obs, scale=scale)

s_sim2 = [2.0, 98]
dist2 = distance(s_sim2, s_obs, scale=scale)

s_sim3 = [2.05, 98.05]
dist3 = distance(s_sim3, s_obs, scale=scale)

s_sim4 = [20, 200]
dist4 = distance(s_sim4, s_obs, scale=scale)

print(f"{dist1=}")
print(f"{dist2=}")
print(f"{dist3=}")
print()

h = 0.5

gkernel1 = gaussian_kernel(dist1, h)
gkernel2 = gaussian_kernel(dist2, h)
gkernel3 = gaussian_kernel(dist3, h)
gkernel4 = gaussian_kernel(dist4, h)

gnorm = gaussian_kernel(0, h)

ekernel1 = epkov_kernel(dist1, h)
ekernel2 = epkov_kernel(dist2, h)
ekernel3 = epkov_kernel(dist3, h)
ekernel4 = epkov_kernel(dist4, h)

print(f"{gkernel1=}")
print(f"{gkernel2=}")
print(f"{gkernel3=}")
print(f"{gkernel4=}")
print(f"{gnorm=}")
print(gkernel1 / gnorm)
print(gkernel2 / gnorm)
print(gkernel3 / gnorm)
print(gkernel3 * h)
print()

print(f"{ekernel1=}")
print(f"{ekernel2=}")
print(f"{ekernel3=}")
print(f"{ekernel4=}")
