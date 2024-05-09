from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from music_scripts.musicdata import MusicData


def func(x, a, h, H, C):
    b = C / (H - h) - a * (H + h)
    c = -a * h**2 - b * h
    result = []
    for i in range(len(x)):
        if x[i] <= h:
            result.append(0)
        else:
            result.append(a * x**2 + b * x + c)
    return result


def height_interface(snap):
    array = snap.rprof["scalar_1"].array()[::-1]
    cum_dib = np.cumsum(array - array[0])

    norm_2 = np.zeros(len(cum_dib) - 1)
    space = snap.grid.grids[0].cell_points()
    print(space)
    print(len(space))
    H = space[len(array) - 1]
    C = cum_dib[len(cum_dib) - 1]

    for h in range(2, len(array) - 1):
        popt, pcov = curve_fit(
            lambda x, a: func(
                x,
                a,
                h,
                H,
                C,
            ),
            space,
            cum_dib,
        )
        fit = func(space, popt[0], h, H, C)
        norm_2[h] = np.sum((fit - cum_dib) ** 2)

    return space[norm_2.argmin()]


def h_interface(snap):
    array = snap.rprof["scalar_1"].array()[::-1]
    fit = np.zeros(len(array))
    norm_2 = np.zeros(len(array) - 1)
    space = snap.grid.grids[0].cell_points()

    for h in range(len(array) - 1):
        a = (array[len(array) - 1] - array[h]) / (len(array) - 1 - h)
        fit[len(array) - 1] = array[len(array) - 1]
        for i in range(h + 1, len(array)):
            fit[i] = fit[i - 1] + a
        norm_2[h] = np.sqrt(np.sum((fit - array) ** 2))

    return space[norm_2.argmin()]


def moving_average(interfaces):
    hsw = 4
    smoothing_window = np.ones(2 * hsw) / (2 * hsw)
    smooth_interfaces = np.convolve(interfaces, smoothing_window, "valid")
    return smooth_interfaces


mdat = MusicData("/home/al1007/newfcdir/params.nml")

plt.figure()
array = mdat[-1].rprof["scalar_1"].array()[::-1]
plt.plot(np.cumsum(array - array[0]))
plt.savefig("ecdf_prof.png")


data = mdat.big_array
times = data.labels_along_axis("time")[1:]
spaces = data.labels_along_axis("x1")

height_interfaces = [height_interface(snap) for snap in mdat[1:]]
# height_interfaces_smooth = moving_average(height_interfaces)

comparison = np.sqrt(times)

comparison *= height_interfaces[-1] / comparison[-1]

plt.figure()
plt.plot(times, height_interfaces)
plt.plot(times, comparison)


plt.savefig("height_interfaces.png")
