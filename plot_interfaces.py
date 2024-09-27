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
from fit_strategies import FitStrategy, LinearScalarFit
from height_interface import Height_Interfaces

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


def plot_h_interfaces(fit_strategy: FitStrategy, simu_numbers: NDArray):
    H_I = []
    comparison = []
    times = []

    fig, ax = plt.subplots()
    ax.set_xlabel("t (s)")
    ax.set_ylabel("h(t) (cm)")

    for i in range(len(simu_numbers)):
        mdat = MusicData(f"{simu_numbers[i]}fuentes/params.nml")
        Physics = PhysicsSimu(mdat)
        H_I.append(Height_Interfaces(Physics, fit_strategy).interface[1])
        times.append(Height_Interfaces(Physics, fit_strategy).interface[0])
        comp = np.sqrt(Physics.flux_top / Physics.F_crit) * np.sqrt(times[i])
        comp *= H_I[i][-1] / comp[-1]
        comparison.append(comp)
        plt.plot(
            times[i],
            H_I[i],
            label=f"Pr={Physics.prandtl:.1f}"
            r",$\frac{F_0}{F_{crit}}=10.8,R_0=$"
            f"{Physics.initial_R_0}",
        )
        plt.plot(
            times[i],
            comparison[i],
            linestyle="dashed",
            label=f"Pr={Physics.prandtl:.1f}"
            r",$\frac{F_0}{F_{crit}}=10.8,R_0=$"
            f"{Physics.initial_R_0}",
        )
        plt.legend()
        numbers = "_".join(map(str, simu_numbers))

    plt.savefig(f"height_interfaces_comparison_{numbers}.png")


def main() -> None:
    fit_strategy = LinearScalarFit

    simu_numbers = np.array([2, 3])
    plot_h_interfaces(fit_strategy, simu_numbers)


if __name__ == "__main__":
    main()
