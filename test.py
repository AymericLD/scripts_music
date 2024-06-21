from __future__ import annotations

import typing
import matplotlib.pyplot as plt
import numpy as np
from music_scripts.musicdata import MusicData, Snap
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from lmfit import Model
import time


if typing.TYPE_CHECKING:
    from numpy.typing import NDArray
    from pymusic.big_array import BigArray

start_time_specific = time.time()


def predicted_values(
    z: NDArray,
    C: float,
    H: float,
    h: float,
    a: float,
    m: float,
) -> NDArray:
    b = (C - m * h) / (H - h) - a * (H + h)
    c = -a * h**2 - b * h + m * h
    mask = z > h
    result = np.zeros_like(z)
    result[mask] = a * z[mask] ** 2 + b * z[mask] + c
    result[~mask] = m * z[~mask]
    return result


mdat = MusicData("/z2/users/al1007/fuentes/params.nml")
snap = mdat[700]
array = snap.rprof["scalar_1"].array()[::-1]
cum_dib = np.cumsum(array - array[0])
space = snap.grid.grids[0].cell_points()
box_height = space[len(array) - 1]
total_scalar_composition = cum_dib[len(cum_dib) - 1]


model = Model(predicted_values)
params = model.make_params(
    C=total_scalar_composition, H=box_height, h=10, a=5e-4, m=1e-4
)
params["C"].vary = False
params["H"].vary = False

result = model.fit(cum_dib, params, z=space)
print(result.fit_report())

plt.plot(space, cum_dib, "b")
plt.plot(space, result.best_fit, "r")
plt.savefig("test_lmfit.png")


def interface(
    mdat: MusicData,
) -> tuple[NDArray, NDArray]:
    data = mdat.big_array
    times = data.labels_along_axis("time")[1:]
    height_interfaces = []
    for snap in mdat[1:]:
        array = snap.rprof["scalar_1"].array()[::-1]
        cum_dib = np.cumsum(array - array[0])
        space = snap.grid.grids[0].cell_points()
        box_height = space[len(array) - 1]
        total_scalar_composition = cum_dib[len(cum_dib) - 1]
        model = Model(predicted_values)
        params = model.make_params(
            C=total_scalar_composition, H=box_height, h=17, a=5e-4, m=1e-4
        )
        params["C"].vary = False
        params["H"].vary = False

        result = model.fit(cum_dib, params, z=space)
        fitted_params = result.params

        height_interfaces.append(fitted_params["h"].value)
    height_interfaces = np.array(height_interfaces)

    return (times, height_interfaces)


plt.figure()
plt.plot(interface(mdat)[0], interface(mdat)[1])
comparison_1 = np.sqrt(interface(mdat)[0])

comparison_1 *= interface(mdat)[1][-1] / comparison_1[-1]

plt.plot(interface(mdat)[0], comparison_1)
plt.savefig("test_height_interfaces_lmfit.png")


end_time_specific = time.time()

execution_time_specific = end_time_specific - start_time_specific
print(
    f"Durée d'exécution de la partie spécifique : {execution_time_specific:.2f} secondes"
)
