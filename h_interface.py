import pymusic
import pymusic.io as pmio
import numpy as np
import matplotlib.pyplot as plt

sim = pmio.MusicSim.from_dump_dir(
    "/home/al1007/newfcdir/frames",
    [
        pmio.ReflectiveArrayBC(),
        pmio.PeriodicArrayBC(),
    ],
)

data = sim.big_array()
times = data.labels_along_axis("time")
profiles = data.xs("scalar_1", axis="var").mean("x2")


def height_interface(profile):
    array = profile.array()[::-1]
    cum_dib = np.cumsum(array - array[0])
    fit = np.zeros(len(cum_dib))
    norm_2 = np.zeros(len(cum_dib) - 1)

    for h in range(len(data.labels_along_axis("x2")) - 1):
        a = (array[len(array) - 1] - array[h]) / (len(array) - 1 - h)
        fit[len(cum_dib) - 1] = cum_dib[len(cum_dib) - 1]
        for i in range(h + 1, len(array)):
            fit[i] = fit[i - 1] + a
        norm_2[h] = np.sqrt(np.sum((fit - array) ** 2))

    return norm_2.argmin()


height_interfaces = np.zeros(len(times))

for i in range(len(times)):
    profile = profiles.xs(times[i], axis="time")
    height_interfaces[i] = height_interface(profile)

plt.figure()
plt.plot(times, height_interfaces)
plt.savefig("height_interfaces.png")
