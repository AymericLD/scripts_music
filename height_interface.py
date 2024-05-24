from __future__ import annotations

import typing
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from music_scripts.musicdata import MusicData, Snap
from abc import ABC, abstractmethod
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray
    from pymusic.big_array import BigArray


class Model_CDF(ABC):
    """Model of the CDF profile of the scalar"""

    scalar_profile: NDArray
    cum_dib: NDArray
    box_height: int
    total_scalar_composition: float
    space: NDArray

    @staticmethod
    def fromsnap(snap: Snap) -> Model_CDF: ...

    @abstractmethod
    def CDF_profile(self, z: NDArray) -> NDArray: ...


class Model_scalar(ABC):
    """Model of the scalar profile"""

    @abstractmethod
    def scalar_profile(self, z: NDArray) -> NDArray: ...


@dataclass(frozen=True)
class ContinuousScalarFit:
    """Fit of cumulative distribution of the scalar profile (subtracted from the top value)
    assuming the scalar is continuous at the interface"""

    scalar_profile: NDArray
    cum_dib: NDArray
    box_height: int
    total_scalar_composition: float
    space: NDArray
    fit_parameters: NDArray

    @staticmethod
    def fromsnap(snap: Snap) -> ContinuousScalarFit:
        array = snap.rprof["scalar_1"].array()[::-1]
        cum_dib = np.cumsum(array - array[0])
        space = snap.grid.grids[0].cell_points()
        box_height = space[len(array) - 1]
        total_scalar_composition = cum_dib[len(cum_dib) - 1]
        model = ContinuousScalarFit(
            scalar_profile=array,
            cum_dib=cum_dib,
            box_height=box_height,
            total_scalar_composition=total_scalar_composition,
            space=space,
        )

        return model

    def CDF_profile(self, z: NDArray, h: float) -> NDArray:
        a = self.total_scalar_composition / (h - self.box_height) ** 2
        b = self.total_scalar_composition / (self.box_height - h) - a * (
            self.box_height + h
        )
        c = -a * h**2 - b * h
        result = np.zeros(len(z))
        for i in range(len(z)):
            if z[i] > h:
                result[i] = a * z[i] ** 2 + b * z[i] + c
        return result

    def height_interface(self) -> Height_Interface_Fit:
        popt, pcov = curve_fit(
            self.CDF_profile,
            self.space,
            self.cum_dib,
        )

        fit = self.CDF_profile(self.space, *popt)

        height_interface = popt[0]

        norm2 = np.sum((fit - self.cum_dib) ** 2)

        height_interface_fit = Height_Interface_Fit(
            height_interface=height_interface, fit=fit, norm2=norm2
        )

        return height_interface_fit


@dataclass(frozen=True)
class DiscontinuousScalarFit:
    """Fit of cumulative distribution of the scalar profile (subtracted from the top value)
    allowing the scalar to be discontinuous at the interface"""

    scalar_profile: NDArray
    cum_dib: NDArray
    box_height: int
    total_scalar_composition: float
    space: NDArray
    initial_guess = [7, 0.05]

    @staticmethod
    def fromsnap(snap: Snap) -> DiscontinuousScalarFit:
        array = snap.rprof["scalar_1"].array()[::-1]
        cum_dib = np.cumsum(array - array[0])
        space = snap.grid.grids[0].cell_points()
        box_height = space[len(array) - 1]
        total_scalar_composition = cum_dib[len(cum_dib) - 1]
        model = DiscontinuousScalarFit(
            scalar_profile=array,
            cum_dib=cum_dib,
            box_height=box_height,
            total_scalar_composition=total_scalar_composition,
            space=space,
        )

        return model

    def CDF_profile(self, z, h, a):
        b = self.total_scalar_composition / (self.box_height - h) - a * (
            self.box_height + h
        )
        c = -a * h**2 - b * h
        result = np.zeros(len(z))
        for i in range(len(z)):
            if z[i] > h:
                result[i] = a * z[i] ** 2 + b * z[i] + c
        return result

    def bounds(self, constraint=0.05):
        bounds = (0, [self.box_height, constraint])
        return bounds

    def height_interface(self) -> Height_Interface_Fit:
        popt, pcov = curve_fit(
            self.CDF_profile,
            self.space,
            self.cum_dib,
            p0=self.initial_guess,
            bounds=self.bounds(),
        )

        fit = self.CDF_profile(self.space, *popt)

        height_interface = popt[0]

        norm2 = np.sum((fit - self.cum_dib) ** 2)

        height_interface_fit = Height_Interface_Fit(
            height_interface=height_interface, fit=fit, norm2=norm2
        )

        return height_interface_fit


@dataclass(frozen=True)
class Height_Interface_Fit:
    """Parameters of the profile fit to determine the height of the interface"""

    height_interface: int
    fit: NDArray
    norm2: NDArray


def interface(mdat, fit_strategy):
    data = mdat.big_array
    times = data.labels_along_axis("time")[1:]

    height_interfaces = [
        fit_strategy.height_interface(fit_strategy.fromsnap(snap)).height_interface
        for snap in mdat[1:]
    ]

    return (times, height_interfaces)


mdat = MusicData("/home/al1007/newfcdir/params.nml")


# times, height_interfaces = interface(mdat, DiscontinuousScalarFit)

# comparison_1 = np.sqrt(times)
# comparison_1 *= height_interfaces[-1] / comparison_1[-1]

# plt.figure()
# plt.plot(times, height_interfaces)
# plt.plot(times, comparison_1)


# plt.savefig("height_interfaces.png")

# # Tests


def error_1(snap: Snap, fit_strategy: Model_CDF) -> NDArray:
    model = fit_strategy.fromsnap(snap)
    cum_dib = model.cum_dib
    fit = np.zeros(len(cum_dib))
    norm_2 = np.zeros(len(cum_dib))
    for i in range(len(cum_dib)):
        h = model.space[i]
        fit = model.CDF_profile(model.space, h)
        norm_2[i] = np.sum((fit - cum_dib) ** 2)

    return norm_2


def error_2(snap: Snap, fit_strategy: Model_CDF) -> NDArray:
    model = fit_strategy.fromsnap(snap)
    cum_dib = model.cum_dib
    fit = np.zeros(len(cum_dib))
    norm_2 = np.array([np.zeros(len(cum_dib)) for i in range(100)])
    for j in range(0, 100):
        a = j / 1000
        for i in range(len(cum_dib)):
            h = model.space[i]
            fit = model.CDF_profile(model.space, h, a)
        norm_2[j, i] = np.sum((fit - cum_dib) ** 2)

    return norm_2


snap = mdat[320]
space = snap.grid.grids[0].cell_points()
plt.figure()
plt.plot(space, error_1(snap, ContinuousScalarFit))
# plt.plot(space, height_interface(snap))
plt.savefig("test_continuous.png")

plt.figure()
a = np.linspace(0, 0.1, 100)
print(error_2(snap, DiscontinuousScalarFit))
plt.contour(space, a, error_2(snap, DiscontinuousScalarFit))
# plt.plot(space, height_interface(snap))
plt.savefig("test_discontinuous.png")
