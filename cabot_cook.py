from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from music_scripts.musicdata import MusicData

# # Option originale

# def func(x, a, h, H, C):
#     b = C / (H - h) - a * (H + h)
#     c = -a * h**2 - b * h
#     result = np.zeros(len(x))
#     for i in range(len(x)):
#         if x[i] > h:
#             result[i] = a * x[i] ** 2 + b * x[i] + c
#     return result


# def height_interface(snap):
#     array = snap.rprof["scalar_1"].array()[::-1]
#     cum_dib = np.cumsum(array - array[0])

#     norm_2 = np.zeros(len(cum_dib) - 200)
#     opt = np.zeros(len(cum_dib) - 200)
#     space = snap.grid.grids[0].cell_points()
#     H = space[len(array) - 1]
#     C = cum_dib[len(cum_dib) - 1]

#     for i in range(len(space) - 200):
#         h = space[i]
#         popt, pcov = curve_fit(
#             lambda x, a: func(
#                 x,
#                 a,
#                 h,
#                 H,
#                 C,
#             ),
#             space,
#             cum_dib,
#             bounds=(-0.05, 0.05),
#         )
#         fit = func(space, popt[0], h, H, C)
#         opt[i] = popt[0]
#         norm_2[i] = np.sum((fit - cum_dib) ** 2)

#     fit = func(space, opt[norm_2.argmin()], space[norm_2.argmin()], H, C)

#     # return space[norm_2.argmin()]
#     return fit

# Option suggérée par Thomas


def func(z, h, H, C):
    a = C / (h - H) ** 2
    b = C / (H - h) - a * (H + h)
    c = -a * h**2 - b * h
    result = np.zeros(len(z))
    for i in range(len(z)):
        if z[i] > h:
            result[i] = a * z[i] ** 2 + b * z[i] + c
    return result


def height_interface(snap):
    array = snap.rprof["scalar_1"].array()[::-1]
    cum_dib = np.cumsum(array - array[0])

    space = snap.grid.grids[0].cell_points()
    H = space[len(array) - 1]
    C = cum_dib[len(cum_dib) - 1]

    popt, pcov = curve_fit(
        lambda z, h: func(
            z,
            h,
            H,
            C,
        ),
        space,
        cum_dib,
    )

    fit = func(space, popt[0], H, C)
    print(popt[0])

    h_index = np.argmin(np.abs(space - popt[0]))
    print(cum_dib[h_index])

    return fit


# # Option intermédiaire


# def func_1(z, h, H, C):
#     a = C / (h - H) ** 2
#     b = C / (H - h) - a * (H + h)
#     c = -a * h**2 - b * h
#     result = np.zeros(len(z))
#     for i in range(len(z)):
#         if z[i] > h:
#             result[i] = a * z[i] ** 2 + b * z[i] + c
#     return result


# def func_2(z, a, h, H, C):
#     b = C / (H - h) - a * (H + h)
#     c = -a * h**2 - b * h
#     result = np.zeros(len(z))
#     for i in range(len(z)):
#         if z[i] > h:
#             result[i] = a * z[i] ** 2 + b * z[i] + c
#     return result


# def height_interface(snap):
#     array = snap.rprof["scalar_1"].array()[::-1]
#     cum_dib = np.cumsum(array - array[0])

#     space = snap.grid.grids[0].cell_points()
#     H = space[len(array) - 1]
#     C = cum_dib[len(cum_dib) - 1]

#     popt, pcov = curve_fit(
#         lambda z, h: func_1(
#             z,
#             h,
#             H,
#             C,
#         ),
#         space,
#         cum_dib,
#     )

#     norm_2 = np.zeros(100)
#     opt = np.zeros(100)
#     width = int(len(norm_2))
#     h_index = np.argmin(np.abs(space - popt[0]))

#     for i in range(width):
#         h = space[h_index + i]
#         popt, pcov = curve_fit(
#             lambda x, a: func_2(
#                 x,
#                 a,
#                 h,
#                 H,
#                 C,
#             ),
#             space,
#             cum_dib,
#             bounds=(-0.05, 0.05),
#         )
#         fit = func_2(space, popt[0], h, H, C)
#         opt[i] = popt[0]
#         norm_2[i] = np.sum((fit - cum_dib) ** 2)

#     h = space[h_index + norm_2.argmin()]
#     a = opt[norm_2.argmin()]
#     final_fit = func_2(space, a, h, H, C)

#     return final_fit


# # Scalar Profile


def h_interface(snap):
    array = snap.rprof["scalar_1"].array()[::-1]
    array -= array[0]
    norm_2 = np.zeros(len(array) - 1)  # Last point not taken into account
    space = snap.grid.grids[0].cell_points()

    # for h in range(len(array) - 1):
    #     fit = np.zeros(len(array))
    #     fit[len(array) - 1] = array[len(array) - 1]
    #     a = (array[len(array) - 1] - array[h]) / (len(array) - 1 - h)
    #     fit[h + 1] = fit[len(array) - 1] - a * (len(array) - h - 1)
    #     fit[h + 1 : len(array) - 1] = np.linspace(
    #         fit[h + 1], fit[len(array) - 2], len(array) - h - 2
    #     )
    #     norm_2[h] = np.sqrt(np.sum((fit - array) ** 2))

    # print(norm_2)

    for h in range(len(array) - 1):
        fit = np.zeros(len(array))
        fit[len(array) - 1] = array[len(array) - 1]
        a = (array[len(array) - 1] - array[h]) / (len(array) - 1 - h)
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


# # Interface Height

data = mdat.big_array
times = data.labels_along_axis("time")[1:]


# height_interfaces = [height_interface(snap) for snap in mdat[1:]]
# # height_interfaces_smooth = moving_average(height_interfaces)

# comparison_1 = np.sqrt(times)

# comparison_1 *= height_interfaces[-1] / comparison_1[-1]

# plt.figure()
# plt.plot(times, height_interfaces)
# plt.plot(times, comparison_1)


# plt.savefig("height_interfaces.png")

h_interfaces = [h_interface(snap) for snap in mdat[1:]]

comparison_2 = np.sqrt(times)

comparison_2 *= h_interfaces[-1] / comparison_2[-1]

plt.figure()
plt.plot(times, h_interfaces)
plt.plot(times, comparison_2)

plt.savefig("h_interfaces.png")

# # Tests

# snap = mdat[320]
# array = snap.rprof["scalar_1"].array()[::-1]
# cum_dib = np.cumsum(array - array[0])
# space = snap.grid.grids[0].cell_points()

# plt.figure()
# plt.plot(space, cum_dib)
# plt.plot(space, height_interface(snap))
# plt.savefig("test.png")
