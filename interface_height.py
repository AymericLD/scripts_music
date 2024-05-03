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

data = sim.big_array(verbose=True)

times = data.labels_along_axis("time")

profiles = data.xs("scalar_1", axis="var").mean("x2")


def h_interface(image):
    H = len(image.labels_along_axis("x1"))
    dz = np.mean(np.diff(image.labels_along_axis("x1")))
    X = np.zeros(H)
    X_M = np.zeros(H)
    image_data = image.array()
    for i in range(H):
        X[i] = (image_data[i] - image_data[H - 1]) / (image_data[0] - image_data[H - 1])
        if X[i] <= 0.5:
            X_M[i] = 2 * X[i]
        else:
            X_M[i] = 2 * (1 - X[i])
    m_l = sum(X_M) * dz
    height_z = image.labels_along_axis("x1")[H - 1] - image.labels_along_axis("x1")[0]
    print(X_M)
    return height_z - m_l


height_interface = np.zeros(len(times))

for i in range(len(times)):
    profile = profiles.xs(times[i], axis="time")
    height_interface[i] = h_interface(profile)


plt.plot(times, height_interface)
plt.show()
plt.savefig("interface.png")
