def func(x, a, h, H, C):
    b = C / (H - h) - a * (H + h)
    c = -a * h**2 - b * h
    result = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] > h:
            result[i] = a * x[i] ** 2 + b * x[i] + c
    return result


def height_interface(snap):
    array = snap.rprof["scalar_1"].array()[::-1]
    cum_dib = np.cumsum(array - array[0])

    norm_2 = np.zeros(len(cum_dib) - 200)
    space = snap.grid.grids[0].cell_points()
    H = space[len(array) - 1]
    C = cum_dib[len(cum_dib) - 1]

    for i in range(len(space) - 200):
        h = space[i]
        popt, pcov = curve_fit(
            lambda x, a: func(
                x,
                a,
                h,
                H,
                C,
            ),
            space,
            cum_dib,
            bounds=(-0.05, 0.05),
        )
        fit = func(space, popt[0], h, H, C)
        norm_2[i] = np.sum((fit - cum_dib) ** 2)

    return space[norm_2.argmin()]


def h_interface(snap):
    array = snap.rprof["scalar_1"].array()[::-1]
    fit = np.zeros(len(array))
    norm_2 = np.zeros(len(array) - 1)
    space = snap.grid.grids[0].cell_points()

    for h in range(len(array) - 1):
        a = (array[len(array) - 1] - array[h]) / (len(array) - 1 - h)
        fit[len(array) - 1] = array[len(array) - 1]
        for i in range(h + 1, len(array)):
            fit[i] = fit[i - 1] + a
        norm_2[h] = np.sqrt(np.sum((fit - array) ** 2))

    return space[norm_2.argmin()]


# h_interfaces = [h_interface(snap) for snap in mdat[1:]]

# comparison_2 = np.sqrt(times)

# comparison_2 *= h_interfaces[-1] / comparison_2[-1]

# plt.figure()
# plt.plot(times, h_interfaces)
# plt.plot(times, comparison_2)

# plt.savefig("h_interfaces.png")

norm_2 = np.zeros(100)
    width = int(len(norm_2))
    h_index = np.argmin(np.abs(space - popt[0]))

    for i in range(width):
        h = space[h_index + i]
        fit = func(space, h, H, C)
        norm_2[i] = np.sum((fit - cum_dib) ** 2)

    print(norm_2)
    h = space[h_index + norm_2.argmin()]
    print(h)
    final_fit = func(space, h, H, C)

def height_interface(model: Model_CDF) -> Height_Interface_Fit:
    popt, pcov = curve_fit(
        model.CDF_profile,
        model.space,
        model.cum_dib,
    )

    fit = model.CDF_profile(model.space, *popt)

    height_interface = popt[0]

    height_interface_fit = Height_Interface_Fit(
        height_interface=height_interface, fit=fit
    )

    return height_interface_fit


class Model_CDF(ABC):
    """Model of the CDF profile of the scalar"""

    scalar_profile: NDArray
    cum_dib: NDArray
    box_height: int
    total_scalar_composition: float
    space: NDArray

    @staticmethod
    def fromsnap(snap: Snap) -> Model_CDF: ...

    @abstractmethod
    def CDF_profile(self, z: NDArray) -> NDArray: ...


class Model_scalar(ABC):
    """Model of the scalar profile"""

    @abstractmethod
    def scalar_profile(self, z: NDArray) -> NDArray: ...


def error_1(snap: Snap, fit_strategy: Model_CDF) -> NDArray:
    model = fit_strategy.fromsnap(snap)
    cum_dib = model.cum_dib
    fit = np.zeros(len(cum_dib))
    norm_2 = np.zeros(len(cum_dib))
    for i in range(len(cum_dib)):
        h = model.space[i]
        fit = model.CDF_profile(model.space, h)
        norm_2[i] = np.sum((fit - cum_dib) ** 2)

    return norm_2


def error_2(snap: Snap, fit_strategy: Model_CDF) -> NDArray:
    model = fit_strategy.fromsnap(snap)
    cum_dib = model.cum_dib
    fit = np.zeros(len(cum_dib))
    norm_2 = np.array([np.zeros(len(cum_dib)) for i in range(100)])
    for j in range(0, 100):
        a = j / 1000
        for i in range(len(cum_dib)):
            h = model.space[i]
            fit = model.CDF_profile(model.space, h, a)
        norm_2[j, i] = np.sum((fit - cum_dib) ** 2)

    return norm_2

plt.plot(space, error_1(snap, ContinuousScalarFit))
plt.contour(space, a, error_2(snap, DiscontinuousScalarFit))
print(error_2(snap, DiscontinuousScalarFit))
a = np.linspace(0, 0.1, 100)

def height_interface(self) -> Height_Interface_Fit:
        parameters = int(self.fit_parameters)
        popt, pcov = curve_fit(
            lambda z, *parameters: self.CDF_profile(z, *parameters),
            self.space,
            self.cum_dib,
        )

        fit = self.CDF_profile(self.space, *popt)

        height_interface = popt[0]

        norm2 = np.sum((fit - self.cum_dib) ** 2)

        height_interface_fit = Height_Interface_Fit(
            height_interface=height_interface, fit=fit, norm2=norm2
        )

        return height_interface_fit

def height_interface(self) -> Height_Interface_Fit:
        popt, pcov = curve_fit(
            self.CDF_profile,
            self.space,
            self.cum_dib,
            p0=self.initial_guess,
            bounds=self.bounds(),
        )

        fit = self.CDF_profile(self.space, *popt)

        height_interface = popt[0]

        norm2 = np.sum((fit - self.cum_dib) ** 2)

        height_interface_fit = Height_Interface_Fit(
            height_interface=height_interface, fit=fit, norm2=norm2
        )

        return height_interface_fit

@dataclass(frozen=True)
class LinearScalarFit(FitStrategy):
    """Fit of cumulative distribution of the scalar profile (subtracted from the top value)
    using a linear profile for the convective zone and a parabola for the stable zone"""

    box_height: float
    total_scalar_composition: float

    def predicted_values(
        self,
        z: NDArray,
        h: float,
        a: float,
        m: float,
    ) -> NDArray:
        b = (self.total_scalar_composition - m * h) / (self.box_height - h) - a * (
            self.box_height + h
        )
        c = -a * h**2 - b * h + m * h
        mask = z > h
        result = np.zeros_like(z)
        result[mask] = a * z[mask] ** 2 + b * z[mask] + c
        result[~mask] = m * z[~mask]
        return result

    def bounds(self, H: float, constraint=0.05):
        bounds = (0, [H, constraint])
        return bounds


@dataclass(frozen=True)
class Linear2ScalarFit(FitStrategy):
    """Fit of cumulative distribution of the scalar profile (subtracted from the top value)
    using a linear profile for the convective zone and a parabola for the stable zone"""

    box_height: float
    total_scalar_composition: float

    def predicted_values(
        self,
        z: NDArray,
        h: float,
        m: float,
    ) -> NDArray:
        a = (self.total_scalar_composition - m * h) / (self.box_height - h) ** 2 + m / (
            h - self.box_height
        )
        b = (self.total_scalar_composition - m * h) / (self.box_height - h) - a * (
            self.box_height + h
        )
        c = -a * h**2 - b * h + m * h
        mask = z > h
        result = np.zeros_like(z)
        result[mask] = a * z[mask] ** 2 + b * z[mask] + c
        result[~mask] = m * z[~mask]
        return result

    def bounds(self, H: float, constraint=0.05):
        bounds = (0, [H, constraint])
        return bounds


if isinstance(self.fit_strategy, LinearScalarFit):

def interface(
    mdat: MusicData, fit_strategy: FitStrategy
) -> tuple[NDArray, NDArray, FitStrategy]:
    data = mdat.big_array
    times = data.labels_along_axis("time")[1:]
    height_interfaces = []
    for snap in mdat[1:]:
        fit = Height_Interface_Fit.fromsnap(snap, fit_strategy)
        height_interfaces.append(fit.height_interface)
    height_interfaces = np.array(height_interfaces)

    return (times, height_interfaces)


def plot_height_interfaces_comparison(H: tuple[NDArray, NDArray, FitStrategy]) -> None:
    comparison = np.sqrt(H[0])
    comparison *= H[1][-1] / comparison[-1]
    plt.figure()
    plt.plot(H[0], H[1])
    plt.plot(H[0], comparison)
    strategy_type = type(H[2])
    directory = "/z2/users/al1007/figures/height_interfaces/"
    plt.savefig(f"{directory}height_interface_evolution_with{strategy_type}.png")


# plt.savefig("height_interfaces.png")

# # Second step
        # width = (
        #     1 * fit_parameters
        # )  # Proportion of the fit_parameters range to look at for the misfit error
        # left_width = 2 * fit_parameters
        # right_width = 4 * fit_parameters
        # nb_test_points = 50
        # parameters = []
        # for i in range(len(fit_parameters)):
        #     parameters.append(
        #         np.linspace(
        #             np.abs(fit_parameters[i] - left_width[i]),
        #             fit_parameters[i] + right_width[i],
        #             2 * nb_test_points + 1,
        #         )
        #     )
        # # The number of parameters for the fit is assumed to be either one,two or three
        # if len(fit_parameters) == 1:
        #     norm2 = np.array(
        #         [self.norm2_misfit(parameters[0][i]) for i in range(len(parameters[0]))]
        #     )
        #     fit_parameters = [parameters[0][norm2.argmin()]]
        # elif len(fit_parameters) == 2:
        #     norm2 = np.array(
        #         [
        #             self.norm2_misfit(parameters[0][i], fit_parameters[1])
        #             for i in range(len(parameters[0]))
        #         ]
        #     )
        #     fit_parameters = [parameters[0][norm2.argmin()], fit_parameters[1]]
        # else:
        #     norm2 = np.array(
        #         [
        #             self.norm2_misfit(
        #                 parameters[0][i], fit_parameters[1], fit_parameters[2]
        #             )
        #             for i in range(len(parameters[0]))
        #         ]
        #     )
        #     fit_parameters = [
        #         parameters[0][norm2.argmin()],
        #         fit_parameters[1],
        #         fit_parameters[2],
        #     ]