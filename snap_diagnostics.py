from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from pymusic.big_array import Partials, BigArray, DerivedFieldArray

from music_scripts.constants import GAS_CONSTANT
from music_scripts.eos import IdealGasMix2, EoS
from music_scripts.musicdata import MusicData, Snap
from music_scripts.derived_fields import press
from height_interface import (
    Height_Interfaces,
    FitStrategy,
    Height_Interface_Fit,
    LinearScalarFit,
)
from dataclasses import dataclass
from functools import cached_property
from pymusic.grid import CartesianGrid2D, SphericalGrid2D
from physicsimu import PhysicsSimu
import typing

if typing.TYPE_CHECKING:
    import f90nml

    from numpy.typing import NDArray
    from pymusic.big_array import BigArray
    from typing import Any, Mapping, Tuple


@dataclass(frozen=True)
class SnapPhysics:
    """Computation of physical quantities for a given snap taken from a MUSIC Simulation"""

    snap: Snap


@dataclass(frozen=True)
class R_0Snap:
    """Computation of local R_0 in different ways"""

    physics: PhysicsSimu
    snap_index: int

    @cached_property
    def snap(self) -> Snap:
        return self.physics.mdat[self.snap_index]

    @cached_property
    def LocalR_0fromGradients_Snap(self) -> NDArray:
        grad_ad = self.snap.rprof["adiab_grad"].array()[1::]
        R_0 = np.zeros(len(self.snap.grid.grids[0].cell_points()))
        if isinstance(mdat.eos, IdealGasMix2):
            temp_profile = self.snap.rprof["temp"].array()
            comp_profile = self.snap.rprof["scalar_1"].array()
            press_profile = self.snap.rprof["press"].array()
            mu_profile = mdat.eos._mu(comp_profile)
            grad_temp = np.diff(np.log(temp_profile)) / np.diff(np.log(press_profile))
            grad_mu = np.diff(np.log(mu_profile)) / np.diff(np.log(press_profile))
            R_0 = (grad_temp - grad_ad) / grad_mu
        return R_0

    def LocalR_0fromGradients(self) -> NDArray:
        R_0 = np.zeros(len(self.physics.mdat), self.snap.grid.grids[0].cell_points())
        for snap_index in len(self.physics.mdat):
            R_0[snap_index] = self.LocalR_0fromGradients_Snap(mdat[snap_index])

    def plot_radial_profile_R0(self, filepath: str, GradOrBV: bool):
        physics = self.physics
        if GradOrBV:
            R_0 = self.LocalR_0fromGradients_Snap
        else:
            R_0 = self.LocalR_0FromBVFrequencies
        space = self.snap.grid.grids[0].cell_points()
        space = (
            space
            + (1 / 2)
            * self.physics.geometrical_parameters[1]
            / self.physics.geometrical_parameters[0]
        )
        plt.figure()
        plt.plot(space[20:-1], R_0[20:])
        plt.axhline((physics.prandtl + physics.tau) / (physics.prandtl + 1))
        directory = filepath
        if GradOrBV:
            plt.savefig(f"{directory}R_0FromGrads.png")
        else:
            plt.savefig(f"{directory}R_0FromBVFrequencies.png")

    def plot_R0_height_interface(
        self, fit_strategy: FitStrategy, filepath: str, GradOrBV: bool
    ) -> None:
        physics = self.physics
        if GradOrBV:
            R_0 = self.LocalR_0fromGradients
        else:
            R_0 = self.LocalR_0FromBVFrequencies
        height_interfaces = Height_Interfaces(physics, fit_strategy)
        times, h_f = height_interfaces.interface
        plt.figure()
        plt.plot(h_f, R_0)
        plt.axhline((physics.prandtl + physics.tau) / (physics.prandtl + 1))
        directory = filepath
        if GradOrBV:
            plt.savefig(f"{directory}R_0FromGrads.png")
        else:
            plt.savefig(f"{directory}R_0FromBVFrequencies.png")

    # Brunt Vaisala Frequency Squared

    @cached_property
    def BV_frequency_2(self) -> NDArray:
        pressure = self.snap.rprof["press"]
        rho = self.snap.rprof["density"]
        gamma_1 = self.snap.rprof["gamma1"].array()
        dlnp_dr = Partials(pressure.apply(np.log), ["x1"]).array().squeeze()
        dlnrho_dr = Partials(rho.apply(np.log), ["x1"]).array().squeeze()
        N_2 = self.physics.gravity * ((1 / gamma_1) * dlnp_dr - dlnrho_dr)
        return N_2

    # Brunt Vaisala Thermal Frequency Squared

    @cached_property
    def BV_thermalfrequency_2(self) -> NDArray:
        dlnrho_dlnT = -1  # Ideal gas
        pressure = self.snap.rprof["press"]
        rho = self.snap.rprof["density"].array()
        temp = self.snap.rprof["temp"]
        adiabatic_gradient = self.snap.rprof["adiab_grad"].array()
        H_p = (pressure.array() / rho) * 1 / self.physics.gravity
        dlnT_dr = Partials(temp.apply(np.log), ["x1"]).array().squeeze()
        dlnp_dr = Partials(pressure.apply(np.log), ["x1"]).array().squeeze()
        dlnT_dlnp = dlnT_dr / dlnp_dr

        N_thermal_2 = (
            -dlnrho_dlnT
            * (self.physics.gravity / H_p)
            * (adiabatic_gradient - dlnT_dlnp)
        )  # Ideal gas

        return N_thermal_2

    @cached_property
    def LocalR_0FromBVFrequencies(self) -> NDArray:
        N_t = self.BV_thermalfrequency_2
        R_0 = N_t / (self.BV_frequency_2 - N_t)
        return R_0


@dataclass(frozen=True)
class diagnostics:
    physics: PhysicsSimu
    snap_index: int

    @cached_property
    def R_0_from_self(self) -> R_0Snap:
        return R_0Snap(self.physics, self.snap_index)

    # Lower boundary for the resolution of the thermal boundary layer
    @cached_property
    def average_resolution_thermal_bl(self) -> Tuple[float, float, float, float]:
        resolution = []
        delta_t = []
        Deltat_Hconv = []
        physics = self.physics
        for i in range(len(physics.mdat)):
            R_0 = R_0Snap(physics, i)
            N_thermal = max(np.abs(R_0.BV_thermalfrequency_2))
            delta_t.append(
                np.sqrt(physics.diff_th / N_thermal)
            )  # Thickness of thermal boundary layer
            Deltat_Hconv.append(delta_t / physics.geometrical_parameters[1])
            resolution.append(
                physics.geometrical_parameters[0]
                * delta_t
                / physics.geometrical_parameters[1]
            )  # Number of grid points on the height of the thermal boundary layer
        delta_t_avg = np.average(np.array(delta_t))
        deltat_Hconv_avg = np.average(np.array(Deltat_Hconv))
        deltat_Hconv = 1 / (physics.rayleigh * physics.prandtl) ** (1 / 4)
        resolution_1 = np.average(np.array(resolution))
        return (delta_t_avg, deltat_Hconv_avg, deltat_Hconv, resolution_1)

    @cached_property
    def resolution_thermal_bl(self) -> Tuple[float, float, float, float]:
        resolution = []
        physics = self.physics
        R_0 = self.R_0_from_self
        N_thermal = max(np.abs(R_0.BV_thermalfrequency_2))
        deltat_Hconv_1 = np.sqrt(
            physics.diff_th / (N_thermal * physics.geometrical_parameters[1] ** 2)
        )
        delta_t = np.sqrt(physics.diff_th / N_thermal)
        resolution = (
            physics.geometrical_parameters[0]
            * delta_t
            / physics.geometrical_parameters[1]
        )
        deltat_Hconv_2 = (physics.rayleigh * physics.prandtl) ** (-1 / 4)
        return (delta_t, deltat_Hconv_1, deltat_Hconv_2, resolution)

    def convection_timescale(self, snap_index: int):
        snap = mdat[snap_index]
        CDF_fit = Height_Interface_Fit.fromsnap(snap, LinearScalarFit)
        h = CDF_fit.height_interface
        physics = self.physics
        h_index = int(
            h * physics.geometrical_parameters[0] / physics.geometrical_parameters[1]
        )
        v_rms = snap.rprof["vrms"].array()
        v_rms_conv = v_rms[h_index:]
        space = np.linspace(
            0, physics.geometrical_parameters[1], physics.geometrical_parameters[0] + 2
        )
        dx = np.abs(np.mean(np.diff(space)))
        t_conv = np.sum(1 / v_rms_conv) * dx

        return t_conv

    @property
    def print_diagnostics(self, snap_index: int) -> None:
        print(
            "The number of cells of the thermal boundary layer is ",
            self.average_resolution_thermal_bl[3],
        )
        print(
            "This is computed assuming that the convective timescale is equal to the inverse of the thermal contribution of the Brunt-Vaisala frequency",
        )
        print(
            "The frequency associated to the convective timescale",
            1 / self.convection_timescale(snap_index),
        )
        print(
            "The thermal contribution of the Brunt-Vaisala frequency on the thermal boundary layer is",
            max(np.abs(self.R_0_from_self.BV_thermalfrequency_2(snap_index))),
        )


# Simulation results

mdat = MusicData("/z2/users/al1007/fuentes/params.nml")
data = mdat[100]
space = mdat[-1].grid.grids[0].cell_points()

physics = PhysicsSimu(mdat)
R_0 = R_0Snap(physics, 600)
R_0.plot_radial_profile_R0("/z2/users/al1007/fuentes/figures/R_0profiles/", True)


figure, axis = plt.subplots(ncols=3)
axis[0].plot(space, R_0.LocalR_0FromBVFrequencies, label="R_rho", c="b")
axis[1].plot(space, R_0.BV_frequency_2, label="Brunt Vaisala Frequency", c="g")
axis[2].plot(
    space[:-10],
    R_0.BV_thermalfrequency_2[:-10],
    label="Thermal Brunt Vaisala Frequency",
    c="r",
)  # Thermal boundary layer at the top : dlnT_dr too big
figure.legend()
directory = "/z2/users/al1007/2fuentes/figures/R_0profiles/"
plt.savefig(f"{directory}R0_profile.png")
plt.figure()


# physics = PhysicsSimu(mdat)
# R_0 = R_0Snap(physics, 500)
# R_0.plot_R0_height_interface(
#     LinearScalarFit, "/z2/users/al1007/fuentes/figures/R_0profiles/", True
# )
