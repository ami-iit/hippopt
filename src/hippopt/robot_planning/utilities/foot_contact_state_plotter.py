import copy
import dataclasses
import math

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from hippopt.robot_planning.utilities.terrain_descriptor import (
    PlanarTerrain,
    TerrainDescriptor,
)
from hippopt.robot_planning.variables.contacts import (
    ContactPointState,
    FootContactState,
)


@dataclasses.dataclass
class ContactPointStatePlotterSettings:
    axes: matplotlib.axes.Axes = dataclasses.field(default=None)
    terrain: TerrainDescriptor = dataclasses.field(default=None)

    input_axes: dataclasses.InitVar[matplotlib.axes.Axes] = dataclasses.field(
        default=None
    )
    input_terrain: dataclasses.InitVar[TerrainDescriptor] = dataclasses.field(
        default=None
    )

    def __post_init__(
        self, input_axes: matplotlib.axes.Axes, input_terrain: TerrainDescriptor
    ):
        self.axes = input_axes
        self.terrain = (
            input_terrain
            if isinstance(input_terrain, TerrainDescriptor)
            else PlanarTerrain()
        )


class ContactPointStatePlotter:
    def __init__(
        self,
        settings: ContactPointStatePlotterSettings = ContactPointStatePlotterSettings(),
    ):
        self.settings = settings
        self._axes = self.settings.axes
        self._fig = None

    def plot_complementarity(
        self,
        states: list[ContactPointState],
        time_s: float | list[float] | np.ndarray = None,
        title: str = "Contact Point Complementarity",
    ):
        _time_s = copy.deepcopy(time_s)
        if _time_s is None or isinstance(_time_s, float) or _time_s.size == 1:
            single_step = _time_s if _time_s is not None else 0.0
            _time_s = np.linspace(0, len(states) * single_step, len(states))

        if len(_time_s) != len(states):
            raise ValueError(
                "timestep_s and foot_contact_states have different lengths."
            )

        if self._axes is None:
            self._fig, self._axes = plt.subplots()

        height_function = self.settings.terrain.height_function()
        normal_direction_fun = self.settings.terrain.normal_direction_function()

        positions = [height_function(s.p) for s in states]
        forces = [normal_direction_fun(s.p).T @ s.f for s in states]
        self._axes.plot(_time_s, positions)
        self._axes.set_ylabel("Height [m]", color="C0")
        self._axes.tick_params(axis="y", color="C0", labelcolor="C0")
        axes_force = self._axes.twinx()
        axes_force.plot(_time_s, forces, "C1")
        axes_force.set_ylabel("Normal Force [N]", color="C1")
        axes_force.tick_params(axis="y", color="C1", labelcolor="C1")
        axes_force.spines["right"].set_color("C1")
        axes_force.spines["left"].set_color("C0")

        if self._fig is not None:
            self._fig.suptitle(title)
            plt.draw()
            plt.pause(0.001)


@dataclasses.dataclass
class FootContactStatePlotterSettings:
    number_of_columns: int = dataclasses.field(default=-1)
    terrain: TerrainDescriptor = dataclasses.field(default=None)


class FootContactStatePlotter:
    def __init__(
        self,
        settings: FootContactStatePlotterSettings = FootContactStatePlotterSettings(),
    ):
        self.settings = settings
        self.number_of_rows = -1
        self.fig = None
        self.point_plotters = []

    def plot_complementarity(
        self,
        states: list[FootContactState],
        time_s: float | list[float] | np.ndarray = None,
        title: str = "Foot Contact Complementarity",
    ):
        _time_s = copy.deepcopy(time_s)
        if _time_s is None or isinstance(_time_s, float) or _time_s.size == 1:
            single_step = _time_s if _time_s is not None else 0.0
            _time_s = np.linspace(0, len(states) * single_step, len(states))

        if len(_time_s) != len(states):
            raise ValueError(
                "timestep_s and foot_contact_states have different lengths."
            )

        if len(states) == 0:
            return

        number_of_plots = len(states[0])
        if self.settings.number_of_columns < 1:
            self.settings.number_of_columns = math.ceil(math.sqrt(number_of_plots))
        number_of_rows = math.ceil(number_of_plots / self.settings.number_of_columns)

        if self.number_of_rows != number_of_rows:
            self.fig, axes_list = plt.subplots(
                nrows=number_of_rows,
                ncols=self.settings.number_of_columns,
                squeeze=False,
            )
            self.number_of_rows = number_of_rows
            self.point_plotters = [
                ContactPointStatePlotter(
                    ContactPointStatePlotterSettings(
                        input_axes=el, terrain=self.settings.terrain
                    )
                )
                for row in axes_list
                for el in row
            ]
            assert len(self.point_plotters) == number_of_plots

        for p in range(number_of_plots):
            contact_states = [state[p] for state in states]
            self.point_plotters[p].plot_complementarity(
                states=contact_states, time_s=_time_s
            )

        self.fig.suptitle(title)
        plt.draw()
        plt.pause(0.001)
