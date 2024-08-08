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
from fit_strategies import (
    FitStrategy,
    LinearScalarFit,
    DiscontinuousScalarFit,
    ContinuousScalarFit,
    Height_Interface_Fit,
)

start_time = time.time()


if typing.TYPE_CHECKING:
    from numpy.typing import NDArray
    from pymusic.big_array import BigArray


@dataclass(frozen=True)
class Height_Interfaces:
    """Compute and plot the evolution of the interface's height for a given simulation and a given strategy"""

    physics: PhysicsSimu
    fit_strategy: FitStrategy

    @property
    def numerical_factor(self) -> float:
        Physics = self.physics
        A = np.sqrt(
            (2 * Physics.global_R0 * Physics.alpha * Physics.flux_top)
            / (Physics.rho_0 * Physics.cp)
        )
        B = 1 / np.sqrt(Physics.beta * Physics.mu_gradient)
        return A * B

    @property
    def interface(self) -> tuple[NDArray, NDArray]:
        data = self.physics.mdat.big_array
        times = data.labels_along_axis("time")[1:]
        height_interfaces = []
        for snap in self.physics.mdat[1:]:
            fit = Height_Interface_Fit.fromsnap(snap, self.fit_strategy)
            height_interfaces.append(fit.height_interface)
        height_interfaces = np.array(height_interfaces)
        return (times, height_interfaces)

    def plot_height_interfaces_comparison(self, filepath: str) -> None:
        H = self.interface
        comparison = np.sqrt(H[0])
        # comparison *= self.numerical_factor
        comparison *= H[1][-1] / comparison[-1]
        plt.figure()
        plt.plot(H[0], H[1])
        plt.plot(H[0], comparison)
        strategy_type = str(self.fit_strategy)
        plt.savefig(f"{filepath}height_interface_evolution_with{strategy_type}.png")


def main() -> None:
    # Simulation results
    mdat = MusicData("/z2/users/al1007/3fuentes/params.nml")
    fit_strategy = [LinearScalarFit, ContinuousScalarFit, DiscontinuousScalarFit]

    # # Tests for a given snap

    # snap = mdat[100]
    # CDF_fit = Height_Interface_Fit.fromsnap(snap, fit_strategy[0])
    # CDF_fit.plot_fit_comparison
    # CDF_fit.plot_zoom_fit_comparison(0, 2)

    # Height of the interface for the simulation

    Physics = PhysicsSimu(mdat)
    height_interfaces = Height_Interfaces(Physics, fit_strategy[0])
    height_interfaces.plot_height_interfaces_comparison(
        filepath="/z2/users/al1007/3fuentes/figures/height_interfaces/"
    )

    end_time = time.time()

    execution_time = end_time - start_time

    print(f"Execution time : {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
