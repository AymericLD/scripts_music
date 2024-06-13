from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import typing

import numpy as np
from scipy.integrate import solve_ivp
from music_scripts.constants import GAS_CONSTANT

from . import bcs, job
from .eos import IdealMixEos
from .grid import CartesianGeometry, Geometry, Grid
from .rb_double_diff import DoubleDiffSetup, Timescale
from .scalar import LinearProfile
from music_scripts.eos import IdealGasMix2
from .rb_double_diff import DoubleDiffSetup

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray
    from pymusic.big_array import BigArray
    from .scalar import ScalarField


@dataclass(frozen=True)
class FuentesSuperAdiabaticSetup:
    """Fuentes & Cumming Setup with an initial temperature gradient that is Scwharzschild unstable and Ledoux stable"""

    rayleigh: float
    prandtl: float
    lewis: float
    geom: Geometry
    rho_0: float
    diff_th: float
    temp_0: float
    F0_Fcrit: float
    dist_to_ledoux_gradient: float  # in [0,1]
    gravity: float
    eos: IdealMixEos
    scalar_setup: ScalarField
    timescale: Timescale
    grid_z: Grid

    @cached_property
    def mu_profile(self) -> NDArray:
        return self.eos.mu_mix(self.scalar_setup)

    @cached_property
    def dmu_dz(self) -> NDArray:
        return np.gradient(self.mu_profile)

    @cached_property
    def thermal_expansion_coef(self) -> float:
        return

    @cached_property
    def boundary_conditions(self) -> tuple[float, float]:
        return

    @cached_property
    def flux_crit(self):
        return

    # RHS of the ODE for the temperature and pressure initial profiles (along z) ; y=[temp,press]

    def RHS_ODE(self, z: float, y: NDArray) -> tuple[float, float]:
        dT_dz = (
            self.dist_to_ledoux_gradient
            * self.gravity
            * y[1]
            * self.dmu_dz[z]
            / self.mu_profile[z]
            - (self.gravity / self.eos.r_gas)
            * self.mu_profile[z]
            * (self.eos.gamma - 1)
            / self.eos.gamma
        )
        dP_dz = -(self.gravity / self.eos.r_gas) * self.mu_profile[z] * y[2] / y[1]
        return (dT_dz, dP_dz)

    # Solving the coupled ODEs for the temperature, pressure  and density profiles

    def solve_TP(self) -> tuple[NDArray, NDArray, NDArray]:
        bc = self.boundary_conditions
        z_span = [0, self.grid_z.length]
        z = np.linspace(0, self.grid_z.length, self.grid_z.ncells)
        sol = solve_ivp(self.RHS_ODE, z_span, bc, t_eval=z)
        temp_profile = sol.y[0]
        press_profile = sol.y[1]
        density_profile = self.eos.density(self.mu_profile, press_profile, temp_profile)
        return (temp_profile, press_profile, density_profile)

    @cached_property
    def dd_setup(self) -> DoubleDiffSetup:
        return DoubleDiffSetup(
            prandtl=self.prandtl,
            rayleigh=self.rayleigh,
            lewis=self.lewis,
            geom=self.geom,
            rho_0=self.rho_0,
            diff_th=self.diff_th,
            temp_0=self.temp_0,
            delta_t=self.delta_t,
            temp_setup=LinearProfile(val_bot=self.temp_0, val_top=self.temp_0),
            scalar_setup=LinearProfile(val_bot=self.c_bot, val_top=self.c_top),
            eos=self.eos,
            timescale=self.timescale,
            bc_rho_bot=bcs.Cont1(),
            bc_rho_top=bcs.Cont1(),
            bc_e_bot=bcs.Fluxc1(value=0.0),
            bc_e_top=bcs.Fluxc1(value=self.flux_top),
            bc_scalar_bot=bcs.Neumann(value=0.0),
            bc_scalar_top=bcs.Neumann(value=0.0),
        )


def main() -> None:
    setup = FuentesSuperAdiabaticSetup(
        temp_0=293,
        c_top=1.278e-2,
        delta_mu=1.65e-2,
        geom=CartesianGeometry(
            height=25.0,
            aspect_ratio=1.0,
        ),
        timescale=Timescale(total_time=1e4, dump_time=10.0),
        eos=IdealMixEos(mu1=58, mu2=18, gamma=1.12),
        diff_th=1.42e-3,
        rho_0=1.025,
        f0_fcrit=5.4,
        grav=981,
        prandtl=0.1,
    ).dd_setup.on_grid(ncells=512)

    run_dir = job.RunDir(
        path=Path("fuentes"),
        params_in=Path("params_zk.nml"),
    )
    task = job.PrepareDir(force=True) & job.PrintInfo()
    task.execute(run_dir, setup)


if __name__ == "__main__":
    main()
