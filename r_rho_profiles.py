import pymusic
import pymusic.io as pmio
import numpy as np
import matplotlib.pyplot as plt
from music_scripts.musicdata import MusicData


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

pressure=IdealMixEos.(c_top,density,temp)



# Simulation Results

mdat = MusicData("/home/al1007/newfcdir/params.nml")

# Local R_rho computation

array = mdat[-1].rprof["scalar_1"].array()[::-1]

def r_rho(image):
    
