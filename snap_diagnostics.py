from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from pymusic.big_array import Partials, BigArray, DerivedFieldArray

from music_scripts.constants import GAS_CONSTANT
from music_scripts.eos import IdealGasMix2, EoS
from music_scripts.musicdata import MusicData, Snap
from music_scripts.derived_fields import press
from height_interface import Height_Interfaces
from dataclasses import dataclass
from functools import cached_property
from pymusic.grid import CartesianGrid2D, SphericalGrid2D
from physicsimu import PhysicsSimu
from fit_strategies import (
    FitStrategy,
    LinearScalarFit,
    DiscontinuousScalarFit,
    ContinuousScalarFit,
    Height_Interface_Fit,
)
import typing

if typing.TYPE_CHECKING:
    import f90nml

    from numpy.typing import NDArray
    from pymusic.big_array import BigArray
    from typing import Any, Mapping, Tuple


@dataclass(frozen=True)
class SnapPhysics:
    """Computation of physical quantities for a given snap taken from a MUSIC Simulation"""

    physics: PhysicsSimu
    snap_index: int

    @cached_property
    def snap(self) -> Snap:
        return self.physics.mdat[self.snap_index]

    @cached_property
    def sur_adiab(self) -> NDArray:
        temp_profile = self.snap.rprof["temp"].array()
        press_profile = self.snap.rprof["press"].array()
        grad_temp = np.diff(np.log(temp_profile)) / np.diff(np.log(press_profile))
        grad_ad = self.snap.rprof[
            "adiab_grad"
        ].array()[
            1::
        ]  # Remove the first point as gradients are defined at faces and grad_ad constant
        return grad_temp - grad_ad

    @cached_property
    def grad_mu(self) -> NDArray:
        if isinstance(self.physics.mdat.eos, IdealGasMix2):
            comp_profile = self.snap.rprof["scalar_1"].array()
            mu_profile = self.physics.mdat.eos._mu(comp_profile)
            press_profile = self.snap.rprof["press"].array()
            grad_mu = np.diff(np.log(mu_profile)) / np.diff(np.log(press_profile))
        return grad_mu

    @cached_property
    def LocalR_0fromGradients_Snap(self) -> NDArray:
        return self.sur_adiab / self.grad_mu

    def LocalR_0fromGradients(self) -> NDArray:
        R_0 = np.zeros(len(self.mdat), self.snap.grid.grids[0].cell_points())
        for snap_index in len(self.mdat):
            R_0[snap_index] = self.LocalR_0fromGradients_Snap(self.mdat[snap_index])

    def plot_radial_profile_R0(self, filepath: str, Grad: bool):
        if Grad:
            R_0 = self.LocalR_0fromGradients_Snap
        else:
            R_0 = self.LocalR_0FromBVFrequencies
        space = self.snap.grid.grids[0].cell_points()
        space = space + (1 / 2) * self.physics.get_geom_parameter(
            "radial_extension"
        ) / self.physics.get_geom_parameter(
            "n_cells"
        )  # Décalage parce que les gradients sont définis aux faces
        height_interface = Height_Interface_Fit.fromsnap(
            self.snap, LinearScalarFit
        ).height_interface
        z_interface = (
            self.physics.get_geom_parameter("radial_extension") - height_interface
        )
        z_index = round(
            z_interface
            * self.physics.get_geom_parameter("n_cells")
            / self.physics.get_geom_parameter("radial_extension")
        )
        diag = diagnostics(self.physics, self.snap_index)
        fig, ax = plt.subplots()
        ax.set_xlabel("z (cm)")
        ax.set_ylabel(r"$R_0$")
        R_0 = self.moving_average
        if self.snap_index > 30 and self.physics.prandtl < 1:
            plt.plot(space[10:z_index], R_0[10:z_index], color="g")
            plt.axhline(
                (self.physics.prandtl + self.physics.tau) / (self.physics.prandtl + 1),
                label=r"$\frac{Pr + \tau}{Pr+1}$",
                color="b",
            )
            plt.axhline(np.sqrt(self.physics.prandtl), label=r"$\sqrt{Pr}$", color="r")
            plt.axvline(z_interface, label=r"Thermal Boundary Layer", color="y")
            plt.axvline(
                z_interface - diag.get_resolution_parameter("delta_t"), color="y"
            )
        elif self.snap_index > 30 and self.physics.prandtl > 1:
            plt.plot(
                space[10:z_index],
                R_0[10:z_index],
                color="g",
            )
            plt.axhline(
                (self.physics.prandtl + self.physics.tau) / (self.physics.prandtl + 1),
                label=r"$\frac{Pr + \tau}{Pr+1}$",
                color="b",
            )
            plt.axvline(z_interface, label=r"Thermal Boundary Layer", color="y")
            plt.axvline(
                z_interface - diag.get_resolution_parameter("delta_t"), color="y"
            )
        else:
            plt.plot(space[10:250], R_0[10:250], color="g")
        directory = filepath
        plt.legend()
        if Grad:
            plt.savefig(f"{directory}R_0FromGrads{self.snap_index}.png")
        else:
            plt.savefig(f"{directory}R_0FromBVFrequencies{self.snap_index}.png")

    def plot_R0_time_average_profiles(self, average_window: int, filepath: str):
        mdat = self.physics.mdat
        sliced_mdat = mdat[
            self.snap_index - average_window, self.snap_index + average_window
        ]
        space = self.snap.grid.grids[0].cell_points()
        height_interface = Height_Interface_Fit.fromsnap(
            self.snap, LinearScalarFit
        ).height_interface
        z_interface = (
            self.physics.get_geom_parameter("radial_extension") - height_interface
        )
        z_index = round(
            z_interface
            * self.physics.get_geom_parameter("n_cells")
            / self.physics.get_geom_parameter("radial_extension")
        )
        space = space + (1 / 2) * self.physics.get_geom_parameter(
            "radial_extension"
        ) / self.physics.get_geom_parameter("n_cells")
        r_0 = np.average(
            [
                SnapPhysics(self.physics, snap.idump).LocalR_0fromGradients_Snap
                for snap in sliced_mdat
            ],
            axis=0,
        )
        hsw = 4
        smoothing_window = np.ones(2 * hsw) / (2 * hsw)
        smooth_R_0 = np.convolve(r_0, smoothing_window, "valid")
        diag = diagnostics(self.physics, self.snap_index)
        fig, ax = plt.subplots()
        plt.plot(space[:z_index], smooth_R_0[:z_index])
        plt.axhline(
            (self.physics.prandtl + self.physics.tau) / (self.physics.prandtl + 1),
            label=r"$\frac{Pr + \tau}{Pr+1}$",
            color="b",
        )
        plt.axhline(np.sqrt(self.physics.prandtl), label=r"$\sqrt{Pr}$", color="r")
        plt.axvline(z_interface, label=r"Thermal Boundary Layer", color="y")
        plt.axvline(z_interface - diag.get_resolution_parameter("delta_t"), color="y")
        plt.legend()
        plt.savefig(f"{filepath}averageR_0.png")

    @cached_property
    def moving_average(self):
        R_0 = self.LocalR_0fromGradients_Snap
        hsw = 4
        smoothing_window = np.ones(2 * hsw) / (2 * hsw)
        smooth_R_0 = np.convolve(R_0, smoothing_window, "valid")
        return smooth_R_0

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
    def BV_thermalfrequency_2bis(self) -> NDArray:
        density = self.snap.rprof["density"].array()
        space = self.snap.grid.grids[0].cell_points()
        drho_dz = np.diff(density) / np.diff(space)
        N_thermal_2 = -(self.physics.gravity / density) * drho_dz

        return N_thermal_2

    @cached_property
    def LocalR_0FromBVFrequencies(self) -> NDArray:
        N_t = self.BV_thermalfrequency_2
        R_0 = N_t / (self.BV_frequency_2 - N_t)
        return R_0

    def plot_radial_profile_Nt(self, filepath: str):
        space = self.snap.grid.grids[0].cell_points()
        N_t = self.BV_thermalfrequency_2
        N = self.BV_frequency_2
        fig, ax = plt.subplots()
        plt.plot(space[:-15], N_t[:-15])
        plt.plot(space[:-15], N[:-15])
        height_interface = Height_Interface_Fit.fromsnap(
            self.snap, LinearScalarFit
        ).height_interface
        z_interface = (
            self.physics.get_geom_parameter("radial_extension") - height_interface
        )
        z_index = round(
            z_interface
            * self.physics.get_geom_parameter("n_cells")
            / self.physics.get_geom_parameter("radial_extension")
        )
        diag = diagnostics(self.physics, self.snap_index)
        # plt.axvline(z_interface,label=r"Thermal Boundary Layer",color="y")
        # plt.axvline(z_interface-diag.get_resolution_parameter("delta_t"),color="y")
        directory = filepath
        plt.legend()
        plt.savefig(f"{directory}radialN_t.png")

    def plot_radial_T_profile(self, filepath: str):
        space = self.snap.grid.grids[0].cell_points()
        temp = self.snap.rprof["temp"].array()
        dT_dz = np.diff(temp) / np.diff(space)
        fig, ax = plt.subplots()
        plt.plot(temp, space)
        plt.legend()
        plt.savefig(f"{filepath}T_profile{self.snap_index}.png")

    def plot_radial_mu_profile(self, filepath: str):
        if isinstance(self.physics.mdat.eos, IdealGasMix2):
            comp_profile = self.snap.rprof["scalar_1"].array()
            mu_profile = self.physics.mdat.eos._mu(comp_profile)
        space = self.snap.grid.grids[0].cell_points()
        dmu_dz = np.diff(mu_profile) / np.diff(space)
        fig, ax = plt.subplots()
        plt.plot(mu_profile, space)
        plt.legend()
        plt.savefig(f"{filepath}mu_profile{self.snap_index}.png")

    def plot_radial_mu_profile_h_averaged(self, filepath: str, slice: tuple[int, int]):
        scalar_1 = self.snap.field["scalar_1"]
        scalar_1_sliced = scalar_1.take_filter(
            lambda x: slice[0] < x < slice[1], "x2"
        ).mean("x2")
        space = scalar_1_sliced.labels_along_axis("x1")
        if isinstance(self.physics.mdat.eos, IdealGasMix2):
            mu_profile = self.physics.mdat.eos._mu(scalar_1_sliced.array())
        fig, ax = plt.subplots()
        plt.plot(mu_profile, space)
        plt.legend()
        plt.savefig(
            f"{filepath}mu_profile{self.snap_index}_h_averaged[{slice[0],slice[1]}].png"
        )


@dataclass(frozen=True)
class diagnostics:
    physics: PhysicsSimu
    snap_index: int

    @cached_property
    def snap_physics_from_self(self) -> SnapPhysics:
        return SnapPhysics(self.physics, self.snap_index)

    # Lower boundary for the resolution of the thermal boundary layer
    @cached_property
    def resolution_thermal_bl(self) -> dict:
        resolution = []
        physics = self.physics
        snap_physics = self.snap_physics_from_self
        N_thermal = max(np.abs(snap_physics.BV_thermalfrequency_2))
        deltat_Hconv_1 = np.sqrt(
            physics.diff_th
            / (N_thermal * physics.get_geom_parameter("radial_extension") ** 2)
        )
        delta_t = np.sqrt(physics.diff_th / N_thermal)
        resolution = (
            physics.get_geom_parameter("n_cells")
            * delta_t
            / physics.get_geom_parameter("radial_extension")
        )
        deltat_Hconv_2 = (physics.rayleigh * physics.prandtl) ** (-1 / 4)

        return {
            "delta_t": delta_t,
            "deltat_Hconv_1": deltat_Hconv_1,
            "deltat_Hconv_2": deltat_Hconv_2,
            "resolution": resolution,
        }

    def get_resolution_parameter(self, parameter_name: str) -> float | None:
        resolution = self.resolution_thermal_bl
        return resolution.get(parameter_name)

    @cached_property
    def convection_timescale(self) -> float:
        snap = self.physics.mdat[self.snap_index]
        CDF_fit = Height_Interface_Fit.fromsnap(snap, LinearScalarFit)
        h = CDF_fit.height_interface
        physics = self.physics
        h_index = int(
            h
            * physics.get_geom_parameter("n_cells")
            / physics.get_geom_parameter("radial_extension")
        )
        v_rms = snap.rprof["vrms"].array()
        v_rms_conv = v_rms[h_index:]
        dx = physics.get_geom_parameter("dx1")[h_index:]
        t_conv = np.dot(1 / v_rms_conv, dx)

        return t_conv

    @property
    def print_diagnostics(self) -> None:
        print(
            "The number of cells of the thermal boundary layer is ",
            self.get_resolution_parameter("resolution"),
        )
        print(
            "The extension of the thermal boundary layer is",
            self.get_resolution_parameter("delta_t"),
            self.get_resolution_parameter("deltat_Hconv_2")
            * self.physics.get_geom_parameter("radial_extension"),
        )
        print(
            "This is computed assuming that the convective timescale is equal to the inverse of the thermal contribution of the Brunt-Vaisala frequency",
        )
        print(
            "The convective timescale is",
            self.convection_timescale,
        )
        print(
            "The inverse of the thermal contribution of the Brunt-Vaisala frequency on the thermal boundary layer is",
            1 / np.mean(np.abs(self.snap_physics_from_self.BV_thermalfrequency_2)),
        )


def main() -> None:
    # Simulation results

    simu = "test_thomas"
    mdat = MusicData(f"{simu}/params.nml")
    snap_index = 750
    filepath = f"{simu}/figures/R_0profiles/"

    physics = PhysicsSimu(mdat)
    snap_physics = SnapPhysics(physics, snap_index)
    snap_physics.plot_radial_profile_R0(filepath, Grad=True)
    diag = diagnostics(physics, snap_index)
    diag.print_diagnostics
    snap_physics.plot_radial_profile_Nt(filepath)
    snap_physics.plot_R0_time_average_profiles(2, filepath)

    filepath = f"{simu}/figures/profiles/"
    snap_physics.plot_radial_mu_profile(filepath)
    snap_physics.plot_radial_T_profile(filepath)
    snap_physics.plot_radial_mu_profile_h_averaged(filepath, [15, 17])


if __name__ == "__main__":
    main()
