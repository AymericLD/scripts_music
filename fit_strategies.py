from __future__ import annotations

import typing
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from music_scripts.musicdata import MusicData, Snap
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
import time
from physicsimu import PhysicsSimu

start_time = time.time()


if typing.TYPE_CHECKING:
    from numpy.typing import NDArray
    from pymusic.big_array import BigArray


class FitStrategy(ABC):
    """Strategy for the fit of the scalar's CDF profile"""

    # Profile to fit with
    @abstractmethod
    def predicted_values(z: NDArray, h: float, *args: float): ...

    def curve_fit_kwargs(self) -> dict[str, object]:
        return {
            # "bounds": (0.5, 6),
            # "p0": [17],
        }


@dataclass(frozen=True)
class ContinuousScalarFit(FitStrategy):
    """Fit of cumulative distribution of the scalar profile (subtracted from the top value)
    assuming the scalar is continuous at the interface"""

    box_height: float
    total_scalar_composition: float

    def predicted_values(self, z: NDArray, h: float) -> NDArray:
        a = self.total_scalar_composition / (h - self.box_height) ** 2
        b = self.total_scalar_composition / (self.box_height - h) - a * (
            self.box_height + h
        )
        c = -a * h**2 - b * h
        mask = z > h
        result = np.zeros_like(z)
        result[mask] = a * z[mask] ** 2 + b * z[mask] + c
        return result

    def curve_fit_kwargs(self) -> dict[str, object]:
        return {
            "p0": [17],
        }


@dataclass(frozen=True)
class DiscontinuousScalarFit(FitStrategy):
    """Fit of cumulative distribution of the scalar profile (subtracted from the top value)
    allowing the scalar to be discontinuous at the interface"""

    box_height: float
    total_scalar_composition: float

    def predicted_values(
        self,
        z: NDArray,
        h: float,
        a: float,
    ) -> NDArray:
        b = self.total_scalar_composition / (self.box_height - h) - a * (
            self.box_height + h
        )
        c = -a * h**2 - b * h
        mask = z > h
        result = np.zeros_like(z)
        result[mask] = a * z[mask] ** 2 + b * z[mask] + c
        return result

    def bounds(self, H: float, constraint=0.05):
        bounds = (0, [H, constraint])
        return bounds

    def curve_fit_kwargs(self) -> dict[str, object]:
        return {
            "p0": [17, 5e-4],
        }


@dataclass(frozen=True)
class LinearScalarFit(FitStrategy):
    """Fit of cumulative distribution of the scalar profile (subtracted from the top value)
    using linear profile in the convective region and a quadratic profile in the stable region, allowing
    the scalar to be discontinuous at the interface"""

    box_height: float
    total_scalar_composition: float

    def predicted_values(
        self,
        z: NDArray,
        h: float,
        a: float,
        m: float,
    ) -> NDArray:
        b = (self.total_scalar_composition - m * h) / (self.box_height - h) - a * (
            self.box_height + h
        )
        c = -a * h**2 - b * h + m * h
        mask = z > h
        result = np.zeros_like(z)
        result[mask] = a * z[mask] ** 2 + b * z[mask] + c
        result[~mask] = m * z[~mask]
        return result

    def bounds(self, H: float, constraint=0.05):
        bounds = (0, [H, constraint])
        return bounds

    def curve_fit_kwargs(self) -> dict[str, object]:
        return {
            "p0": [17, 5e-4, 1e-4],
        }


@dataclass(frozen=True)
class Height_Interface_Fit:
    """Parameters of the profile fit to determine the height of the interface"""

    space: NDArray
    values: NDArray  # CDF values
    box_height: float
    total_scalar_composition: float
    fit_strategy: FitStrategy

    @staticmethod
    def fromsnap(snap: Snap, fit_strategy: FitStrategy) -> Height_Interface_Fit:
        array = snap.rprof["scalar_1"].array()[::-1]
        cum_dib = np.cumsum(array - array[0])
        space = snap.grid.grids[0].cell_points()
        # to modify (just to handle the spherical case)
        if abs(space[0]) > 0.1:
            space = space - space[0]
        box_height = space[len(array) - 1]
        total_scalar_composition = cum_dib[len(cum_dib) - 1]
        fit = Height_Interface_Fit(
            space=space,
            values=cum_dib,
            box_height=box_height,
            total_scalar_composition=total_scalar_composition,
            fit_strategy=fit_strategy(
                box_height=box_height, total_scalar_composition=total_scalar_composition
            ),
        )
        return fit

    @cached_property
    def fit_parameters(self) -> list[float]:
        popt, pcov = curve_fit(
            self.fit_strategy.predicted_values,
            self.space,
            self.values,
            **self.fit_strategy.curve_fit_kwargs(),
        )
        fit_parameters = popt

        return fit_parameters

    @cached_property
    def fit_profile(self) -> NDArray:
        return self.fit_strategy.predicted_values(
            self.space,
            *self.fit_parameters,
        )

    @cached_property
    def misfit(self) -> NDArray:
        return (self.fit_profile - self.values) ** 2

    def norm2_misfit(self, h: float, *args: float) -> float:
        profile = self.fit_strategy.predicted_values(self.space, h, *args)
        return np.sum((profile - self.values) ** 2)

    @cached_property
    def plot_norm2_misfit(self) -> None:
        fit_parameters = self.fit_parameters
        width = (
            1 * fit_parameters
        )  # Proportion of the fit_parameters range to look at for the misfit error
        left_width = 0.3 * fit_parameters
        right_width = 0.5 * fit_parameters
        nb_test_points = 10
        parameters = []
        for i in range(len(fit_parameters)):
            parameters.append(
                np.linspace(
                    np.abs(fit_parameters[i] - left_width[i]),
                    fit_parameters[i] + right_width[i],
                    2 * nb_test_points + 1,
                )
            )
        # The number of parameters for the fit is assumed to be either one or two
        if len(fit_parameters) == 1:
            plt.figure()
            norm2 = np.array(
                [self.norm2_misfit(parameters[0][i]) for i in range(len(parameters[0]))]
            )
            plt.plot(parameters[0], norm2)
            plt.savefig("norm2_error.png")
        else:
            # Contour Plot of norm2_error
            plt.figure()
            norm2 = np.zeros((2 * nb_test_points + 1, 2 * nb_test_points + 1))
            for j in range(len(parameters[1])):
                for i in range(len(parameters[0])):
                    norm2[i][j] = self.norm2_misfit(parameters[0][i], parameters[1][j])
            X, Y = np.meshgrid(parameters[0], parameters[1])
            plt.figure()
            cs = plt.contourf(X, Y, norm2)
            plt.colorbar(cs)
            plt.xlabel("h")
            plt.ylabel("a")
            plt.title("Contour Plot of norm2_error")
            plt.savefig("norm2_error.png")

            # Plot of norm2(h) with a fixed

            plt.figure()
            norm2 = np.array(
                [
                    self.norm2_misfit(parameters[0][i], fit_parameters[1])
                    for i in range(len(parameters[0]))
                ]
            )
            plt.plot(parameters[0], norm2)
            plt.savefig("norm2_error_h.png")

    @property
    def height_interface(self) -> float:
        return self.fit_parameters[0]

    def plot_fit_comparison(self, filepath: str) -> NDArray:
        CDF_fit_profile = self.fit_profile
        print(self.fit_parameters)
        plt.figure()
        plt.plot(self.space, self.values)
        plt.plot(self.space, CDF_fit_profile)
        strategy_type = type(self.fit_strategy)
        plt.savefig(f"{filepath}comparison{strategy_type}.png")

    def plot_zoom_fit_comparison(
        self, initial_position: float, final_position: float, filepath: str
    ) -> NDArray:
        beginning = round(initial_position * len(self.space) / self.box_height)
        end = round(final_position * len(self.space) / self.box_height)
        truncated_space = self.space[beginning:end]
        cum_dib_truncated = self.values[beginning:end]
        CDF_fit_profile_truncated = self.fit_profile[beginning:end]
        plt.figure()
        plt.plot(truncated_space, cum_dib_truncated)
        plt.plot(truncated_space, CDF_fit_profile_truncated)
        strategy_type = type(self.fit_strategy)
        plt.savefig(
            f"{filepath}zoom_comparison_{initial_position}to{final_position}{strategy_type}.png"
        )


def main() -> None:
    # Simulation results

    simu_number = "9fuentes"
    mdat = MusicData(f"{simu_number}/params.nml")
    filepath = f"{simu_number}/figures/tests_fits/"
    fit_strategy = [LinearScalarFit, ContinuousScalarFit, DiscontinuousScalarFit]

    # Tests for a given snap

    snap = mdat[500]
    CDF_fit = Height_Interface_Fit.fromsnap(snap, fit_strategy[0])
    CDF_fit.plot_fit_comparison(filepath)
    CDF_fit.plot_zoom_fit_comparison(0, 13, filepath)

    end_time = time.time()

    execution_time = end_time - start_time

    print(f"Execution time : {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
