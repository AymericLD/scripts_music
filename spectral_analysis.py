from __future__ import annotations


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.optimize import curve_fit
from music_scripts.musicdata import MusicData
import pymusic.spec as pspec
from pymusic.big_array import FFTArray, FFTPowerSpectrumArray
import pymusic.plotting as pmplot


def main() -> None:
    # Simulation results

    simu = "test_thomas"
    mdat = MusicData(f"{simu}/params.nml")
    snap_index = 750
    filepath = f"{simu}/figures/spectral_analysis/"
    data = mdat.big_array
    dtrec = mdat.params["misc"]["dtrec"]
    space = mdat[-1].grid.grids[0].cell_points()
    dx = np.abs(np.mean(np.diff(space)))
    times = data.labels_along_axis("time")

    vel_1 = data.xs("vel_1", axis="var")

    fft_t = pspec.NuFFT1D(
        window=pspec.NormalizedWindow(
            pspec.BlackmanWindow(),
            pspec.PreservePower(),
        ),
        sampling_period=dtrec,
    )

    fft_x = pspec.FFT1D(pspec.NoWindow())

    vel_spec = FFTArray(
        FFTArray(vel_1, fft_x, axis="x2", freq_axis="fx"),
        fft_t,
        axis="time",
        freq_axis="freq",
    )

    vel_s = vel_spec.take_filter(
        lambda freq: -0.04 < freq < 0, axis="freq"
    ).take_filter(lambda fx: -5 < fx < 0, axis="fx")

    # vel_spec = FFTPowerSpectrumArray(
    #     FFTPowerSpectrumArray(vel_1, fft_x, axis="x2", freq_axis="fx"),
    #     fft_t,
    #     axis="time",
    #     freq_axis="freq",
    # )

    # Plot à fx fixé

    Fx = np.array(vel_spec.labels_along_axis("fx"))
    m_x = 1 / 25
    i = np.argmin(np.abs(Fx - m_x))

    vel_fx = vel_spec.xs(i, axis="fx")

    # Plot à z fixé

    z = 400
    vel_z = vel_s.xs(space[z], axis="x1")

    # pmplot.SinglePlotFigure(
    #     pmplot.ArrayImagePlot(
    #         vel_fx.abs2().slabbed("x1", 100).apply(np.log10),
    #         axes=("freq", "x1"),
    #         color_bounds=pmplot.FixedBounds(vmin=-13, vmax=-8),
    #         # pcolormesh_kwargs={"norm": LogNorm()},
    #     )
    # ).save_to(f"{filepath}spectrum.png")

    pmplot.SinglePlotFigure(
        pmplot.ArrayImagePlot(
            vel_z.abs2().slabbed("fx", 100).apply(np.log10),
            axes=("freq", "fx"),
            color_bounds=pmplot.FixedBounds(vmin=-13, vmax=-8),
            # pcolormesh_kwargs={"norm": LogNorm()},
        )
    ).save_to(f"{filepath}spectrum.png")


if __name__ == "__main__":
    main()
