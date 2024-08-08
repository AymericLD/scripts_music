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
from fit_strategies import LinearScalarFit
from height_interface import Height_Interfaces


def main() -> None:
    fit_strategy = LinearScalarFit
    H_I = []
    comparison = []
    times = []

    mdat = MusicData("2fuentes/params.nml")
    Physics = PhysicsSimu(mdat)
    H = Height_Interfaces(Physics, fit_strategy).interface[1]
    t = Height_Interfaces(Physics, fit_strategy).interface[0]
    times.append(t)
    H_I.append(H)
    comp = np.sqrt(Physics.flux_top / Physics.F_crit) * np.sqrt(t)
    comp *= H[-1] / comp[-1]
    comparison.append(comp)

    mdat = MusicData("3fuentes/params.nml")
    Physics = PhysicsSimu(mdat)
    H = Height_Interfaces(Physics, fit_strategy).interface[1]
    t = Height_Interfaces(Physics, fit_strategy).interface[0]
    times.append(t)
    H_I.append(H)
    comp = np.sqrt(Physics.flux_top / Physics.F_crit) * np.sqrt(t)
    comp *= H[-1] / comp[-1]
    comparison.append(comp)

    fig, ax = plt.subplots()
    ax.set_xlabel("t (s)")
    ax.set_ylabel("h(t) (cm)")
    plt.plot(times[0], H_I[0], label=r"$Pr=0.1,\frac{F_0}{F_{crit}}=10.8$")
    plt.plot(times[1], H_I[1], label=r"$Pr=7,\frac{F_0}{F_{crit}}=10.8$")
    plt.plot(
        times[0],
        comparison[0],
        linestyle="dashed",
        label=r"$Pr=0.1,\frac{F_0}{F_{crit}}=10.8$",
    )
    plt.plot(
        times[1],
        comparison[1],
        linestyle="dashed",
        label=r"$Pr=7,\frac{F_0}{F_{crit}}=10.8$",
    )
    plt.legend()
    plt.savefig("height_interfaces.png")


if __name__ == "__main__":
    main()

# Isca version for backup

# def main() -> None:
#     fit_strategy = LinearScalarFit
#     H_I = []
#     comparison = []
#     times = []

#     mdat = MusicData("13fuentes/params.nml")
#     Physics = PhysicsSimu(mdat)
#     H = Height_Interfaces(Physics, fit_strategy).interface[1]
#     t = Height_Interfaces(Physics, fit_strategy).interface[0]
#     times.append(t)
#     H_I.append(H)
#     comp = np.sqrt(Physics.flux_top / Physics.F_crit) * np.sqrt(t)
#     comp *= H[-1] / comp[-1]
#     comparison.append(comp)

#     mdat = MusicData("14fuentes/params.nml")
#     Physics = PhysicsSimu(mdat)
#     H = Height_Interfaces(Physics, fit_strategy).interface[1]
#     t = Height_Interfaces(Physics, fit_strategy).interface[0]
#     times.append(t)
#     H_I.append(H)
#     comp = np.sqrt(Physics.flux_top / Physics.F_crit) * np.sqrt(t)
#     comp *= H[-1] / comp[-1]
#     comparison.append(comp)

#     mdat = MusicData("9fuentes/params.nml")
#     Physics = PhysicsSimu(mdat)
#     H = Height_Interfaces(Physics, fit_strategy).interface[1]
#     t = Height_Interfaces(Physics, fit_strategy).interface[0]
#     times.append(t)
#     H_I.append(H)
#     comp = np.sqrt(Physics.flux_top / Physics.F_crit) * np.sqrt(t)
#     comp *= H[-1] / comp[-1]
#     comparison.append(comp)

#     fig, ax = plt.subplots()
#     ax.set_xlabel('t (s)')
#     ax.set_ylabel('h(t) (cm)')
#     plt.plot(times[0], H_I[0], label=r"$Pr=7,\frac{F_0}{F_{crit}}=5.4$")
#     plt.plot(times[1], H_I[1], label=r"$Pr=0.1,\frac{F_0}{F_{crit}}=5.4$")
#     plt.plot(
#         times[0],
#         comparison[0],
#         linestyle="dashed",
#         label=r"$Pr=7,\frac{F_0}{F_{crit}}=5.4$",
#     )
#     plt.plot(
#         times[1],
#         comparison[1],
#         linestyle="dashed",
#         label=r"$Pr=0.1,\frac{F_0}{F_{crit}}=5.4$",
#     )
#     plt.plot(times[2], H_I[2], label=r"$Pr=0.1,\frac{F_0}{F_{crit}}=5.4,Geom=Spherical$")
#     plt.plot(
#         times[2],
#         comparison[2],
#         linestyle="dashed",
#         label=r"$Pr=0.1,\frac{F_0}{F_{crit}}=5.4$,Geom=Spherical",
#     )
#     plt.legend()
#     plt.savefig("height_interfaces.png")


# if __name__ == "__main__":
#     main()
