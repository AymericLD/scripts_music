from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import numpy as np

if typing.TYPE_CHECKING:
    from f90nml import Namelist
    from numpy.typing import NDArray


@dataclass(frozen=True)
class Grid:
    ncells: int
    length: float
    x_min: float = 0.0

    @cached_property
    def x_max(self) -> float:
        return self.x_min + self.length

    @cached_property
    def delta_x(self) -> float:
        return self.length / self.ncells

    @cached_property
    def walls(self) -> NDArray:
        return np.linspace(self.x_min, self.x_max, self.ncells + 1)

    @cached_property
    def centers(self) -> NDArray:
        return (self.walls[:-1] + self.walls[1:]) / 2


class Geometry(ABC):
    @abstractmethod
    def length_scale(self) -> float: ...

    @abstractmethod
    def is_cartesian(self) -> bool: ...

    @abstractmethod
    def grids(self, ncells: int) -> tuple[Grid, Grid]: ...

    def update_nml(self, nml: Namelist) -> None:
        nml["geometry"]["cartesian"] = self.is_cartesian()


@dataclass(frozen=True)
class CartesianGeometry(Geometry):
    height: float
    aspect_ratio: float

    def length_scale(self) -> float:
        return self.height

    def is_cartesian(self) -> bool:
        return True

    def grids(self, ncells: int) -> tuple[Grid, Grid]:
        grid_r = Grid(ncells=ncells, length=self.height)
        grid_t = Grid(
            ncells=int(ncells * self.aspect_ratio),
            length=self.height * self.aspect_ratio,
        )
        return grid_r, grid_t


@dataclass(frozen=True)
class SphericalGeometry(Geometry):
    height: float
    aspect_ratio: float
    mid_radius: float

    def __post_init__(self) -> None:
        if self.r_min < 0 or self.aperture > np.pi:
            raise ValueError("Inconsistent Geometrical Parameters")

    @cached_property
    def r_min(self) -> float:
        return self.mid_radius - self.height / 2

    @cached_property
    def aperture(self) -> float:
        return self.aspect_ratio * self.height / self.mid_radius

    def length_scale(self) -> float:
        return self.height

    def is_cartesian(self) -> bool:
        return False

    def grids(self, ncells: int) -> tuple[Grid, Grid]:
        grid_r = Grid(ncells=ncells, length=self.height, x_min=self.r_min)
        grid_t = Grid(
            ncells=int(ncells * self.aspect_ratio),
            length=self.aperture,
            x_min=(np.pi - self.aperture) / 2,
        )
        return grid_r, grid_t
