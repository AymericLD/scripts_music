from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import typing
from music_pykg.format2 import Header, MusicNewFormatDumpFile
from .bcs import Bc, BCVar, Boundary
from pprint import pprint
import matplotlib.pyplot as plt

import numpy as np
from scipy.integrate import solve_ivp

from . import bcs, job
from .eos import IdealMixEos
from .grid import CartesianGeometry, Geometry, Grid, SphericalGeometry
from .rb_double_diff import DoubleDiffSetup, Timescale
from .scalar import LinearProfile, CenteredScalar
from .rb_double_diff import ConcreteSetup
from .setup import MusicStateOnGrid, PhysicalSetup

if typing.TYPE_CHECKING:
    import f90nml

    from numpy.typing import NDArray
    from pymusic.big_array import BigArray
    from .scalar import ScalarField
    from typing import Sequence

@dataclass(frozen=True)
class SuperAdiabaticStateAtHSE(MusicStateOnGrid):
    R_0: float
    temp_0: float
    c_top: float
    sur_adiab: float
    rho_0: float
    gravity: float
    on_grid: Grid
    eos: IdealMixEos
    prandtl: float
    lewis: float

    @property
    def grid(self) -> Grid:
        return self.on_grid


    def __post_init__(self) -> None:
        if not (
            0
            <= self.R_0
            <=  (self.prandtl + self.lewis) / (self.prandtl + 1)
        ):
            raise ValueError("Setup might be too stable or already unstable")

    @cached_property
    def mu_top(self) -> float:
        return self.eos.mu_mix(np.asarray(self.c_top)).item()

    @cached_property
    def P_0(self) -> float:
        P = self.rho_0 * self.eos.r_gas * self.temp_0 / self.mu_top
        return P

    @cached_property
    def boundary_conditions(self) -> tuple[float, float, float]:
        bc = [self.temp_0, self.P_0, self.mu_top]
        return bc

    # RHS of the ODE for the temperature, pressure and molecular weight initial profiles (along z) ; y=[temp,press,mu]

    # z is the depth !

    def RHS_ODE(self, z: float, y: NDArray) -> tuple[float, float, float]:
        dT_dz = (y[2]*self.gravity/self.eos.r_gas)*(self.sur_adiab+((self.eos.gamma - 1)/self.eos.gamma))
        dP_dz = (self.gravity / self.eos.r_gas) * y[2] * y[1] / y[0]
        dmu_dz=((self.sur_adiab*self.gravity)/(self.R_0*self.eos.r_gas))*(y[2]**2)/y[0]
        return (dT_dz, dP_dz, dmu_dz)

    # Solving the coupled ODEs for the temperature, pressure, molecular weight and density profiles

    @property
    def solve_TP(self) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        bc = self.boundary_conditions
        z_span = [self.grid.x_min, self.grid.x_max]
        z=self.grid.walls
        sol = solve_ivp(self.RHS_ODE, z_span, bc, t_eval=z)
        temp_profile = sol.y[0,::-1]
        press_profile = sol.y[1,::-1]
        mu_profile=sol.y[2,::-1]
        c_profile=self.eos.c1_from_mu(mu_profile)
        density_profile = self.eos.density(c_profile, press_profile, temp_profile)
        return (temp_profile, press_profile, c_profile, density_profile, mu_profile)

    @cached_property
    def density(self) -> CenteredScalar:
        return CenteredScalar.lininterp_from_walls(self.grid, self.solve_TP[3])
    
    @cached_property
    def molecular_weight(self) -> CenteredScalar:
        return CenteredScalar.lininterp_from_walls(self.grid, self.solve_TP[4])

    @cached_property
    def temperature(self) -> CenteredScalar:
        return CenteredScalar.lininterp_from_walls(self.grid, self.solve_TP[0])

    @cached_property
    def pressure(self) -> CenteredScalar:
        return CenteredScalar.lininterp_from_walls(self.grid, self.solve_TP[1])

    @cached_property
    def scalars(self) -> Sequence[CenteredScalar]:
        return (CenteredScalar.lininterp_from_walls(self.grid, self.solve_TP[2]),)

    @cached_property
    def csound(self) -> CenteredScalar:
        vals = self.eos.csound(
            self.scalars[0].at_centers_gc, self.temperature.at_centers_gc
        )
        return CenteredScalar(grid=self.grid, at_centers_gc=vals)

    @cached_property
    def e_int(self) -> CenteredScalar:
        vals = self.eos.energy_internal(
            self.scalars[0].at_centers_gc, self.temperature.at_centers_gc
        )
        return CenteredScalar(grid=self.grid, at_centers_gc=vals)

    @cached_property
    def heat_capacity_vol(self) -> CenteredScalar:
        vals = self.eos.heat_capacity_vol(self.scalars[0].at_centers_gc)
        return CenteredScalar(grid=self.grid, at_centers_gc=vals)
    
@dataclass(frozen=True)
class FuentesSuperAdiabaticSetup(PhysicalSetup):
    """Fuentes & Cumming Setup with an initial temperature gradient that is Scwharzschild unstable and Ledoux stable, the sur-adiabaticity being 
    fixed constant"""

    R_0: float
    sur_adiab: float
    rayleigh: float
    prandtl: float
    lewis: float
    geom: Geometry
    rho_0: float
    diff_th: float
    temp_0: float
    c_top: float
    F0_Fcrit: float
    eos: IdealMixEos
    timescale: Timescale

    @cached_property
    def ra_pr(self) -> float:
        return self.rayleigh * self.prandtl

    @cached_property
    def mu_top(self) -> float:
        return self.eos.mu_mix(np.asarray(self.c_top)).item()

    @cached_property
    def conductivity(self) -> float:
        cp = self.eos.gamma * self.eos.heat_capacity_vol(np.asarray(self.c_top)).item()
        return self.rho_0 * self.diff_th * cp
    
    
    # It is assumed that the molecular weight profile is quasi-linear (which is correct if mu_1>>mu_2) and this depends on the EOS

    @cached_property
    def abs_delta_mu(self) -> float:
        return (self.sur_adiab*self.gravity*self.mu_top**2)*self.geom.length_scale()/(self.R_0*self.eos.r_gas*self.temp_0)

    @cached_property
    def flux_crit(self) -> float:
        return (
            self.conductivity
            * self.temp_0
            * self.abs_delta_mu
            / (self.mu_top * self.geom.length_scale())
        )

    @cached_property
    def flux_top(self) -> float:
        return self.F0_Fcrit * self.flux_crit

    @cached_property
    def gravity(self) -> float:
        return np.sqrt((self.ra_pr*self.diff_th**2*self.eos.r_gas*self.R_0*self.temp_0)/(self.F0_Fcrit*self.mu_top*self.sur_adiab*self.geom.length_scale()**4))

    @cached_property
    def delta_t(self) -> float:
        return self.flux_top * self.geom.length_scale() / self.conductivity    
    
    @cached_property
    def boundary_conditions(self) :
        bc_rho_bot=bcs.Cont1()
        bc_rho_top=bcs.Cont1()
        bc_e_bot=bcs.Fluxc1(value=0.0)
        bc_e_top=bcs.Fluxc1(value=self.flux_top)
        bc_scalar_bot=bcs.Neumann(value=0.0)
        bc_scalar_top=bcs.Neumann(value=0.0)

        return [bc_rho_bot,bc_rho_top,bc_e_bot,bc_e_top,bc_scalar_bot,bc_scalar_top]

    def on_grid(self, ncells: int) -> ConcreteSetup:
        grid, grid_horiz = self.geom.grids(ncells)
        return ConcreteDDSuperAdiabaticSetup(
            physics=self,
            grid_horiz=grid_horiz,
            discrete_state=SuperAdiabaticStateAtHSE(
                R_0=self.R_0,
                temp_0=self.temp_0,
                c_top=self.c_top,
                sur_adiab=self.sur_adiab,
                rho_0=self.rho_0,
                gravity=self.gravity,
                on_grid=grid,
                eos=self.eos,
                prandtl=self.prandtl,
                lewis=self.lewis,
            )
        )
    
@dataclass(frozen=True)
class ConcreteDDSuperAdiabaticSetup(ConcreteSetup):
    physics: FuentesSuperAdiabaticSetup
    grid_horiz: Grid
    discrete_state: MusicStateOnGrid
    

    def create_dump(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        state = self.discrete_state
        ncells = state.grid.ncells
        ncells_h = self.grid_horiz.ncells
        v_ampl = 1e-3
        v_r = v_ampl * (np.random.random((ncells, ncells_h)) - 0.5)
        v_r[0, :] = 0.0
        one = np.ones_like(self.grid_horiz.centers)
        dump_file = MusicNewFormatDumpFile(path)
        dump_file.write(
            Header(
                xmcore=0.0,
                time=0.0,
                spherical=not self.physics.geom.is_cartesian(),
                face_loc_1=state.grid.walls,
                face_loc_2=self.grid_horiz.walls,
            ),
            dict(
                rho=np.outer(state.density.at_centers, one),
                e=np.outer(state.e_int.at_centers, one),
                v_r=v_r,
                v_t=v_ampl * (np.random.random((ncells, ncells_h)) - 0.5),
                Scalar1=np.outer(state.scalars[0].at_centers, one),
            ),
        )

    def update_nml(self, nml: f90nml.Namelist) -> None:
        physics = self.physics
        nml["physics"]["thermal_diffusion_value"] = physics.diff_th
        nml["physics"]["scalar_diffusivities"] = physics.lewis * physics.diff_th
        nml["physics"]["viscosity"] = physics.prandtl * physics.diff_th
        nml["gravity"]["constant_gravity_value"] = physics.gravity

        physics.geom.update_nml(nml)
        physics.eos.update_nml(nml)
        physics.timescale.update_nml(nml)

        bc_rho_bot,bc_rho_top,bc_e_bot,bc_e_top,bc_scalar_bot,bc_scalar_top=self.physics.boundary_conditions

        bc_rho_bot.update_nml(nml, Boundary.BOTTOM, BCVar.DENSITY)
        bc_rho_top.update_nml(nml, Boundary.TOP, BCVar.DENSITY)

        bc_e_bot.update_nml(nml, Boundary.BOTTOM, BCVar.E_INT)
        bc_e_top.update_nml(nml, Boundary.TOP, BCVar.E_INT)

        bc_scalar_bot.update_nml(nml, Boundary.BOTTOM, BCVar.SCALAR_1)
        bc_scalar_top.update_nml(nml, Boundary.TOP, BCVar.SCALAR_1)

    def diagnostics(self, out_path: Path) -> None:
        physics = self.physics
        state = self.discrete_state
        pprint(self)
        delta_t = state.temperature.at_bot - state.temperature.at_top
        delta_p = state.pressure.at_bot - state.pressure.at_top
        eps_t = delta_t / state.temperature.at_top
        eps_p = delta_p / state.pressure.at_top
        delta_rho = state.density.at_bot - state.density.at_top
        eps_rho = delta_rho / state.density.at_top
        print(f"F_crit = {physics.flux_crit:.5e}")
        print(f"F_top = {physics.flux_top:.5e}")
        print(f"|delta_mu| = {physics.abs_delta_mu:.5e}")
        print(f"rho0 = {state.density.at_top:.5e}")
        print(f"Drho/rho0 = {eps_rho:.5e}")
        print(f"T0 = {state.temperature.at_top:.5e}")
        print(f"DT/T0 = {eps_t:.5e}")
        print(f"P0 = {state.pressure.at_top:.5e}")
        print(f"DP/P0 = {eps_p:.5e}")
        print("RaPr_in = {:.5e}".format(physics.ra_pr))
        grad_ad = (physics.eos.gamma - 1.0) / physics.eos.gamma
        eps_t_ad = eps_p * grad_ad
        print(
            "RaPr_eff = {:.5e}".format(
                float(physics.ra_pr * (eps_t - eps_t_ad) / eps_t)
            )
        )
        print("min(c_s) = {:.5e}".format(np.min(state.csound.at_walls)))
        dp_dx = state.pressure.grad_at_walls
        isostatic = (dp_dx + state.density.at_walls * physics.gravity) / dp_dx
        print("max(1 + rho g / dP/dx) = {:.5e}".format(np.max(isostatic)))

        grad_temp = np.diff(np.log(state.temperature.at_walls)) / np.diff(
            np.log(state.pressure.at_walls)
        )
        grad_mu= np.diff(np.log(state.molecular_weight.at_walls)) / np.diff(
            np.log(state.pressure.at_walls)
        )
        fig, axes = plt.subplots(2, 5, figsize=(16, 11), sharex=True)
        walls = state.grid.walls
        axes[0, 0].plot(walls, state.density.at_walls / state.density.at_top)
        axes[0, 1].plot(walls, state.temperature.at_walls / state.temperature.at_top)
        axes[0, 2].plot(walls, state.pressure.at_walls / state.pressure.at_top)
        axes[0, 3].plot(walls, state.scalars[0].at_walls)
        axes[0, 4].plot(state.grid.centers, (grad_temp - grad_ad)/grad_mu)
        axes[1, 0].plot(walls, isostatic)
        axes[1, 1].plot(state.grid.centers, grad_temp - grad_ad)
        axes[1, 2].plot(walls, state.csound.at_walls)
        axes[1, 3].plot(walls, state.molecular_weight.at_walls)
        axes[1, 4].plot(state.grid.centers, grad_mu)
        axes[0, 0].set_ylabel(r"$\rho / \rho_0$")
        axes[0, 1].set_ylabel(r"$T / T_0$")
        axes[0, 2].set_ylabel(r"$P / P_0$")
        axes[0, 3].set_ylabel(r"$C$")
        axes[0, 4].set_ylabel(r"$R_0$")
        axes[1, 0].set_ylabel(r"$1 + g \rho / \nabla P$")
        axes[1, 1].set_ylabel(r"$\nabla - \nabla_{ad}$")
        axes[1, 2].set_ylabel(r"$c_s$")
        axes[1, 3].set_ylabel(r"$\mu$")
        axes[1, 4].set_ylabel(r"$\nabla \mu$")
        fig.tight_layout()
        fig.savefig(out_path / "profiles.pdf", bbox_inches="tight")

        temps = np.linspace(physics.temp_0, physics.temp_0 + physics.delta_t, 10)
        scalar_1 = self.discrete_state.scalars[0].at_walls
        c_1_0 = np.amin(scalar_1)
        conc = np.linspace(c_1_0, np.amax(scalar_1), 10)
        temps, conc = np.meshgrid(temps, conc, indexing="ij")
        press0 = physics.eos.pressure(
            c_1=c_1_0,
            density=np.asarray(physics.rho_0),
            temp=np.asarray(physics.temp_0),
        )
        rho_tc = physics.eos.density(conc, press0, temps)
        alpha = 1 / physics.temp_0
        beta = 1 / physics.eos.mu_mix(c_1_0)
        rho_tc_bsq = physics.rho_0 * (
            1 - alpha * (temps - physics.temp_0) + beta * conc
        )
        print("nonlin of EoS:", np.amax((rho_tc - rho_tc_bsq) / rho_tc_bsq))
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(temps, conc, rho_tc, label="MUSIC")
        ax.scatter(temps, conc, rho_tc_bsq, label="Boussinesq")
        ax.set_xlabel("$T$")
        ax.set_ylabel("$c$")
        ax.set_title(r"$\rho(T,c)$")
        ax.legend()
        fig.savefig(out_path / "rho_tc.pdf", bbox_inches="tight")

        print()


def main() -> None:
    setup = FuentesSuperAdiabaticSetup(
        R_0=1e-2,
        sur_adiab=0.1,
        rayleigh=4e12,
        prandtl=0.1,
        lewis=0.1,
        geom=SphericalGeometry(height=25,aspect_ratio=2,mid_radius=25),
        rho_0=1.025,
        diff_th=1.42e-3,
        temp_0=293,
        c_top=1.278e-2,
        F0_Fcrit=5.4,
        eos=IdealMixEos(mu1=58, mu2=18, gamma=1.12),
        timescale=Timescale(total_time=1e4, dump_time=10.0),
    ).on_grid(ncells=512)

    run_dir = job.RunDir(
        path=Path("31fuentes"),
        params_in=Path("params_zk.nml"),
    )
    task = job.PrepareDir(force=True) & job.PrintInfo()
    task.execute(run_dir, setup)


if __name__ == "__main__":
    main()


