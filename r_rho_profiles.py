import pymusic
import pymusic.io as pmio
import numpy as np
import matplotlib.pyplot as plt

from __future__ import annotations

from pathlib import Path

import f90nml

from music_ic_setup.eos import IdealMixEos
from music_ic_setup.rb_double_diff import DoubleDiffSetup
from music_ic_setup.scalar import LinearProfile
from music_ic_setup import bcs, job

# Parameters R_rho

c_top = 1.278e-2
mu1 = 58
mu2 = 18
gamma = 1.12
eos = IdealMixEos(mu1=mu1, mu2=mu2, gamma=gamma)
mu_top = eos.mu_mix(c_top)
temp_0 = 293

beta = 1 / mu_top
alpha = temp_0

# Simulation Results

sim = pmio.MusicSim.from_dump_dir(
    "/home/al1007/music/tests_out/runs/2dim/rayleigh_taylor/output",  # Changer le chemin d'accès
    [
        pmio.ReflectiveArrayBC(),
        pmio.PeriodicArrayBC(),
    ],  # Changer les conditions aux bords
)

# Local R_rho computation

data = sim.big_array(verbose=True)
times = data.labels_along_axis("time")
profiles = data.xs("scalar_1", axis="var").mean("x2")

def r_rho(image):
    
