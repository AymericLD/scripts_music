from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from pymusic.big_array import Partials, BigArray, DerivedFieldArray

from music_scripts.constants import GAS_CONSTANT
from music_scripts.eos import IdealGasMix2
from music_scripts.musicdata import MusicData, Snap
from music_scripts.derived_fields import press
from cabot_cook import h_interface


# Parameters R_rho

c_top = 1.278e-2
mu1 = 58
mu2 = 18
gamma = 1.12
eos = IdealGasMix2(gamma1=gamma, gamma2=gamma, mu1=mu1, mu2=mu2, c1_scalar=1)
mu_top = eos._mu(c_top)
temp_0 = 293
g = 981
r_gas = GAS_CONSTANT
diff_th = 1.42e-3
n_cells = 512
Height = 25


alpha = temp_0

# Brunt Vaisala Frequency


def BV_frequency(snap: Snap):
    pressure = snap.rprof["press"]
    rho = snap.rprof["density"]
    gamma_1 = snap.rprof["gamma1"].array()
    dlnp_dr = Partials(pressure.apply(np.log), ["x1"]).array().squeeze()
    dlnrho_dr = Partials(rho.apply(np.log), ["x1"]).array().squeeze()
    N = g * ((1 / gamma_1) * dlnp_dr - dlnrho_dr)

    return N


# Brunt Vaisala Thermal Frequency


def BV_thermalfrequency(snap: Snap):
    dlnrho_dlnT = -1  # Ideal gas
    pressure = snap.rprof["press"]
    rho = snap.rprof["density"].array()
    temp = snap.rprof["temp"]
    adiabatic_gradient = snap.rprof["adiab_grad"].array()
    H_p = (pressure.array() / rho) * 1 / g
    dlnT_dr = Partials(temp.apply(np.log), ["x1"]).array().squeeze()
    dlnp_dr = Partials(pressure.apply(np.log), ["x1"]).array().squeeze()
    dlnT_dlnp = dlnT_dr / dlnp_dr

    N_thermal = -dlnrho_dlnT * (g / H_p) * (adiabatic_gradient - dlnT_dlnp)  # Ideal gas

    return N_thermal


# Local R_rho computation


def r_rho(data):
    N_t = BV_thermalfrequency(data)
    R_rho = (BV_frequency(data) - N_t) / N_t
    return R_rho


def resolution_bd(snap: Snap):
    N_thermal = max(np.abs(BV_thermalfrequency(snap)))
    delta_t = np.sqrt(diff_th / N_thermal)  # Thickness of thermal boundary layer
    resolution = (
        n_cells * delta_t / Height
    )  # Number of grid points on the height of the interface
    return resolution


def convection_timescale(snap: Snap):
    h = h_interface(snap)
    h_index = int(h * n_cells / Height)
    v_rms = snap.rprof["vrms"].array()
    v_rms_conv = v_rms[h_index:]
    space = snap.grid.grids[0].cell_points()
    dx = np.abs(np.mean(np.diff(space)))
    t_conv = np.sum(1 / v_rms_conv) * dx

    return t_conv


# Simulation results

mdat = MusicData("/home/al1007/newfcdir/params.nml")
data = mdat[500]
space = mdat[-1].grid.grids[0].cell_points()


figure, axis = plt.subplots(ncols=3)
axis[0].plot(space, r_rho(data), label="R_rho", c="b")
axis[1].plot(space, BV_frequency(data), label="Brunt Vaisala Frequency", c="g")
axis[2].plot(
    space[:-10],
    BV_thermalfrequency(data)[:-10],
    label="Thermal Brunt Vaisala Frequency",
    c="r",
)  # Thermal boundary layer at the top : dlnT_dr too big
figure.legend()
figure.savefig("r_rho_profile.png")

plt.figure()

print(
    "Le nombre de cellules sur la hauteur de la couche limite thermique est ",
    resolution_bd(data),
)
print("La fréquence associée au temps convectif est", 1 / convection_timescale(data))
print(
    "La contribution thermique de la fréquence de Brunt-Vaisala dans la couche limite thermique est",
    max(np.abs(BV_thermalfrequency(data))),
)
