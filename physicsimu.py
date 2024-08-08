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
    from typing import Any, Mapping, Tuple, Union


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
            * self.get_geom_parameter("radial_extension") ** 4
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
        density_profile = self.mdat[1].rprof["density"].array()
        rho_0 = np.average(density_profile)
        return rho_0

    # alpha is assumed to be constant

    @cached_property
    def alpha(self) -> NDArray:
        temp_profile = self.mdat[1].rprof["temp"].array()
        return 1 / temp_profile[-1]

    # beta is assumed to be constant and the eos is an ideal gas mix

    @cached_property
    def beta(self) -> NDArray:
        comp_profile = self.mdat[1].rprof["scalar_1"].array()
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
        comp_profile = self.mdat[1].rprof["scalar_1"].array()
        space = self.mdat[0].grid.grids[0].cell_points()
        if isinstance(self.mdat.eos, IdealGasMix2):
            mu_profile = self.mdat.eos._mu(comp_profile)
        dmu_dz = np.diff(mu_profile) / np.diff(space)
        return np.abs(dmu_dz[int(len(dmu_dz) / 2)])

    @cached_property
    def F_crit(self) -> NDArray:
        k = self.conductivity
        alpha = self.alpha
        beta = self.beta
        gradient_mu = self.mu_gradient
        return k * gradient_mu * beta / alpha

    @cached_property
    def geometrical_parameters(self) -> dict:
        grid = self.mdat.grid
        n_cells = grid.shape_cells[0]  # direction x1 ?
        parameters = {"n_cells": n_cells}

        if isinstance(grid, CartesianGrid2D):
            parameters.update(
                {
                    "radial_extension": grid.x_grid.span(),
                    "orthoradial_extension": grid.y_grid.span(),
                    "dx1": grid.x_grid.cell_widths(),
                    "dx2": grid.y_grid.cell_widths(),
                }
            )

        elif isinstance(grid, SphericalGrid2D):
            parameters.update(
                {
                    "radial_extension": grid.r_grid.span(),
                    "orthoradial_extension": grid.theta_grid.span(),
                    "dx1": grid.r_grid.cell_widths(),
                    "dx2": grid.theta_grid.cell_widths(),
                }
            )

        return parameters

    def get_geom_parameter(self, parameter_name: str) -> float | None | NDArray:
        parameters = self.geometrical_parameters
        return parameters.get(parameter_name)

    @cached_property
    def average_resolution_thermal_bl(self) -> dict:
        deltat_Hconv = 1 / (self.rayleigh * self.prandtl) ** (1 / 4)
        delta_t = deltat_Hconv * self.get_geom_parameter("radial_extension")
        resolution = (
            self.get_geom_parameter("n_cells")
            * delta_t
            / self.get_geom_parameter("radial_extension")
        )
        return {
            "delta_t": delta_t,
            "deltat_Hconv": deltat_Hconv,
            "resolution": resolution,
        }

    def get_resolution_parameter(self, parameter_name: str) -> float | None:
        resolution = self.average_resolution_thermal_bl
        return resolution.get(parameter_name)

    @cached_property
    def pressure_scale_height(self) -> float:
        press_profile = self.mdat[1].rprof["press"].array()
        density_profile = self.mdat[1].rprof["density"].array()
        P_avg = np.average(press_profile)
        density_avg = np.average(density_profile)
        H_P = P_avg / (density_avg * self.gravity)
        return H_P

    @cached_property
    def print_diagnostics(self) -> None:
        print(
            "The number of cells over the thermal boundary layer is ",
            self.get_resolution_parameter("resolution"),
        )
        print(
            "The ratio between the size of the thermal boundary layer and the size of the box is",
            self.get_resolution_parameter("deltat_Hconv"),
        )
        print(
            "The average ratio between the size of the box and the pressure scale height is",
            self.get_geom_parameter("radial_extension") / self.pressure_scale_height,
        )


def main() -> None:
    mdat = MusicData("/z2/users/al1007/2fuentes/params.nml")
    physics = PhysicsSimu(mdat)

    physics.print_diagnostics


if __name__ == "__main__":
    main()
