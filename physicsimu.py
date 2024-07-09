from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from pymusic.big_array import Partials, BigArray, DerivedFieldArray

from music_scripts.constants import GAS_CONSTANT
from music_scripts.eos import IdealGasMix2, EoS
from music_scripts.musicdata import MusicData, Snap
from music_scripts.derived_fields import press
from music_scripts.derived_fields import FieldGetter, BaseMusicData


from dataclasses import dataclass
from functools import cached_property
from pymusic.grid import CartesianGrid2D, SphericalGrid2D
import typing

if typing.TYPE_CHECKING:
    import f90nml

    from numpy.typing import NDArray
    from pymusic.big_array import BigArray
    from typing import Any, Mapping, Tuple


@FieldGetter.register
def cp(bmdat: BaseMusicData) -> BigArray:
    return DerivedFieldArray(bmdat.big_array, "var", ["scalar_1"], bmdat.eos.heat_cap_p)


@dataclass(frozen=True)
class PhysicsSimu:
    mdat: MusicData

    @cached_property
    def rayleigh(self) -> float:
        return (
            self.alpha
            * self.gravity
            * self.geometrical_parameters[1] ** 4
            * self.flux_top
        ) / (self.diff_th**2 * self.prandtl * self.conductivity)

    @cached_property
    def physics_nml(self) -> Mapping[str, Any]:
        return self.mdat.params["physics"]

    @cached_property
    def prandtl(self) -> float:
        viscosity = self.physics_nml["viscosity"]
        return viscosity / self.diff_th

    # Inverse Lewis Number

    @cached_property
    def tau(self) -> float:
        diff_ch = self.physics_nml["scalar_diffusivities"]
        diff_th = self.diff_th
        return diff_ch / diff_th

    @cached_property
    def diff_th(self) -> float:
        return self.physics_nml["thermal_diffusion_value"]

    @cached_property
    def flux_top(self) -> float:
        boundary_conditions_nml = self.mdat.params["boundaryconditions"]
        return boundary_conditions_nml["outer_flux"]

    @cached_property
    def cp(self) -> float:
        # cp_profile = self.mdat[0].rprof["cp"].array()
        # cp = np.average(cp_profile)
        # return cp
        return 4.182e7

    @cached_property
    def conductivity(self) -> float:
        return self.rho_0 * self.cp * self.diff_th

    @cached_property
    def rho_0(self) -> float:
        density_profile = self.mdat[0].rprof["density"].array()
        rho_0 = np.average(density_profile)
        return rho_0

    # alpha is assumed to be constant

    @cached_property
    def alpha(self) -> NDArray:
        temp_profile = self.mdat[0].rprof["temp"].array()
        return 1 / temp_profile[-1]

    # beta is assumed to be constant and the eos is an ideal gas mix

    @cached_property
    def beta(self) -> NDArray:
        comp_profile = self.mdat[0].rprof["scalar_1"].array()
        if isinstance(self.mdat.eos, IdealGasMix2):
            mu_profile = self.mdat.eos._mu(comp_profile)
        return 1 / mu_profile[-1]

    @cached_property
    def gravity(self) -> float:
        gravity_nml = self.mdat.params["gravity"]
        if gravity_nml["gravity_type"] == "constant_uniform":
            gravity = gravity_nml["constant_gravity_value"]
        return gravity

    @cached_property
    def mu_gradient(self) -> NDArray:
        comp_profile = self.mdat[0].rprof["scalar_1"].array()
        if isinstance(self.mdat.eos, IdealGasMix2):
            mu_profile = self.mdat.eos._mu(comp_profile)
        dmu_dz = np.gradient(mu_profile)
        return np.abs(dmu_dz[round(self.geometrical_parameters[0] / 2)])

    @cached_property
    def geometrical_parameters(self) -> Tuple[float, float, float]:
        grid = self.mdat.grid
        n_cells = grid.shape_cells[0]  # direction x1 ?
        if isinstance(grid, CartesianGrid2D):
            height = grid.x_grid.span()
            width = grid.y_grid.span()
            return (n_cells, height, width)
        if isinstance(grid, SphericalGrid2D):
            radius = grid.r_grid.span()
            angular_aperture = grid.theta_grid.span()
            return (n_cells, radius, angular_aperture)

    @cached_property
    def global_R0(self) -> float:
        comp_profile = self.mdat[0].rprof["scalar_1"].array()
        if isinstance(self.mdat.eos, IdealGasMix2):
            mu_profile = self.mdat.eos._mu(comp_profile)
        temp_profile = self.mdat[0].rprof["temp"].array()
        delta_t = temp_profile[-1] - temp_profile[0]
        delta_mu = mu_profile[-1] - mu_profile[0]
        # return self.alpha * delta_t / (self.beta * delta_mu)
        return 1

    @cached_property
    def average_resolution_thermal_bl(self) -> Tuple[float, float, float]:
        deltat_Hconv = 1 / (self.rayleigh * self.prandtl) ** (1 / 4)
        delta_t = deltat_Hconv * self.geometrical_parameters[1]
        resolution = (
            self.geometrical_parameters[0] * delta_t / self.geometrical_parameters[1]
        )
        return (delta_t, deltat_Hconv, resolution)

    @cached_property
    def pressure_scale_height(self) -> float:
        press_profile = self.mdat[0].rprof["press"].array()
        density_profile = self.mdat[0].rprof["density"].array()
        P_avg = np.average(press_profile)
        density_avg = np.average(density_profile)
        H_P = P_avg / (density_avg * self.gravity)
        return H_P

    @cached_property
    def print_diagnostics(self) -> None:
        print(
            "The average number of cells over the thermal boundary layer is ",
            self.average_resolution_thermal_bl[2],
        )
        print(
            "The ratio between the size of the thermal boundary layer and the size of the box is",
            self.average_resolution_thermal_bl[1],
        )
        print(
            "The average ratio between the size of the box and the pressure scale height is",
            self.geometrical_parameters[0] / self.pressure_scale_height,
        )


def main() -> None:
    mdat = MusicData("/z2/users/al1007/2fuentes/params.nml")
    physics = PhysicsSimu(mdat)

    physics.print_diagnostics


if __name__ == "__main__":
    main()
