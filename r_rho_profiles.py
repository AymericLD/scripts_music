from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from pymusic.big_array import Partials, BigArray, DerivedFieldArray

from music_scripts.constants import GAS_CONSTANT
from music_ic_setup.eos import IdealMixEos
from music_scripts.musicdata import MusicData, Snap
from music_scripts.derived_fields import press
from height_interface import Height_Interfaces, LinearScalarFit, Height_Interface_Fit
from music_ic_setup.fuentes_superad import FuentesLedouxSetup
from dataclasses import dataclass
from functools import cached_property
import typing

if typing.TYPE_CHECKING:
    import f90nml

    from numpy.typing import NDArray
    from pymusic.big_array import BigArray


@dataclass
class PhysicsOfSimu(frozen=True):
    mdat: MusicData
    eos: IdealMixEos
    # setup: FuentesLedouxSetup
    height_interfaces: Height_Interfaces

    prandtl = 0.1
    lewis = 0.1

    @cached_property
    def R_0(self) -> NDArray:
        grad_ad = (self.eos.gamma - 1) / self.eos.gamma
        R = self.eos.r_gas
        R_0 = []
        for snap in mdat:
            temp_profile = snap.rprof["temp"]
            comp_profile = snap.rprof["scalar_1"]
            mu_profile = self.eos.mu_mix(comp_profile.array())
            dT_dz = Partials(temp_profile, ["x1"]).array().squeeze()
            dmu_dz = np.gradient(mu_profile)
            R_0.append(
                (R * dT_dz / mu_profile + grad_ad)
                / (R * temp_profile * dmu_dz / mu_profile**2)
            )
        return np.array(R_0)

    @cached_property
    def plot_R0_height_interface(self) -> None:
        R_0 = self.R_0
        times, h_f = self.height_interfaces.interface
        plt.figure()
        plt.plot(h_f, R_0)
        plt.axhline((self.prandtl + self.lewis) / (self.prandtl + 1))
        plt.savefig("R_0.png")


# Parameters R_rho


g = 981
r_gas = GAS_CONSTANT
diff_th = 1.42e-3
n_cells = 1024
Height = 25


# Brunt Vaisala Frequency Squared


def BV_frequency_2(snap: Snap):
    pressure = snap.rprof["press"]
    rho = snap.rprof["density"]
    gamma_1 = snap.rprof["gamma1"].array()
    dlnp_dr = Partials(pressure.apply(np.log), ["x1"]).array().squeeze()
    dlnrho_dr = Partials(rho.apply(np.log), ["x1"]).array().squeeze()
    N_2 = g * ((1 / gamma_1) * dlnp_dr - dlnrho_dr)

    return N_2


# Brunt Vaisala Thermal Frequency Squared


def BV_thermalfrequency_2(snap: Snap):
    dlnrho_dlnT = -1  # Ideal gas
    pressure = snap.rprof["press"]
    rho = snap.rprof["density"].array()
    temp = snap.rprof["temp"]
    adiabatic_gradient = snap.rprof["adiab_grad"].array()
    H_p = (pressure.array() / rho) * 1 / g
    dlnT_dr = Partials(temp.apply(np.log), ["x1"]).array().squeeze()
    dlnp_dr = Partials(pressure.apply(np.log), ["x1"]).array().squeeze()
    dlnT_dlnp = dlnT_dr / dlnp_dr

    N_thermal_2 = (
        -dlnrho_dlnT * (g / H_p) * (adiabatic_gradient - dlnT_dlnp)
    )  # Ideal gas

    return N_thermal_2


# Local R_0 computation


def R_0(snap: Snap):
    N_t = BV_thermalfrequency_2(snap)
    R_0 = N_t / (BV_frequency_2(snap) - N_t)
    return R_0


def resolution_bd(snap: Snap):
    N_thermal = max(np.abs(BV_thermalfrequency_2(snap)))
    delta_t = np.sqrt(diff_th / N_thermal)  # Thickness of thermal boundary layer
    resolution = (
        n_cells * delta_t / Height
    )  # Number of grid points on the height of the interface
    return resolution


def convection_timescale(snap: Snap):
    CDF_fit = Height_Interface_Fit.fromsnap(snap, LinearScalarFit)
    h = CDF_fit.height_interface
    h_index = int(h * n_cells / Height)
    v_rms = snap.rprof["vrms"].array()
    v_rms_conv = v_rms[h_index:]
    space = np.linspace(0, Height, n_cells + 2)
    dx = np.abs(np.mean(np.diff(space)))
    t_conv = np.sum(1 / v_rms_conv) * dx

    return t_conv


# Simulation results

mdat = MusicData("/z2/users/al1007/fuentes/params.nml")
data = mdat[500]
space = mdat[-1].grid.grids[0].cell_points()


figure, axis = plt.subplots(ncols=3)
axis[0].plot(space, R_0(data), label="R_rho", c="b")
# axis[1].plot(space, BV_frequency_2(data), label="Brunt Vaisala Frequency", c="g")
# axis[2].plot(space[:-10],BV_thermalfrequency_2(data)[:-10],label="Thermal Brunt Vaisala Frequency",c="r",)  # Thermal boundary layer at the top : dlnT_dr too big
figure.legend()
directory = "/z2/users/al1007/figures/R0_profiles/"
plt.savefig(f"{directory}R0_profile.png")
plt.figure()

print(
    "The number of cells of the thermal boundary layer is ",
    resolution_bd(data),
)
print(
    "The frequency associated to the convective timescale",
    1 / convection_timescale(data),
)


print(
    "The thermal contribution of the Brunt-Vaisala frequency on the thermal boundary layer is",
    max(np.abs(BV_thermalfrequency_2(data))),
)


# Tests

physics = PhysicsOfSimu(
    mdat=mdat,
    eos=IdealMixEos(mu1=58, mu2=18),
    height_interfaces=Height_Interfaces(mdat=mdat, fit_strategy=LinearScalarFit),
)
