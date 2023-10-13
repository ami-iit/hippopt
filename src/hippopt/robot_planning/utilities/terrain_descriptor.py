import abc
import dataclasses

import casadi as cs


@dataclasses.dataclass
class TerrainDescriptor(abc.ABC):
    _height_function: cs.Function = dataclasses.field(default=None)
    _normal_direction_function: cs.Function = dataclasses.field(default=None)
    _orientation_function: cs.Function = dataclasses.field(default=None)
    _point_position_name: str = dataclasses.field(default="point_position")
    _options: dict = dataclasses.field(default=None)
    point_position_name: dataclasses.InitVar[str] = dataclasses.field(
        default="point_position"
    )
    options: dataclasses.InitVar[dict] = dataclasses.field(default=None)

    def __post_init__(
        self, point_position_name: str = "point_position", options: dict = None
    ):
        self.change_options(point_position_name, options)

    def change_options(
        self, point_position_name: str = "point_position", options: dict = None, **_
    ):
        self._options = {} if options is None else options
        self._point_position_name = point_position_name

    @abc.abstractmethod
    def create_height_function(self) -> cs.Function:
        pass

    @abc.abstractmethod
    def create_normal_direction_function(self) -> cs.Function:
        pass

    @abc.abstractmethod
    def create_orientation_function(self) -> cs.Function:
        pass

    def height_function(self) -> cs.Function:
        if not isinstance(self._height_function, cs.Function):
            self._height_function = self.create_height_function()

        return self._height_function

    def normal_direction_function(self) -> cs.Function:
        if not isinstance(self._normal_direction_function, cs.Function):
            self._normal_direction_function = self.create_normal_direction_function()

        return self._normal_direction_function

    def orientation_function(self) -> cs.Function:
        if not isinstance(self._orientation_function, cs.Function):
            self._orientation_function = self.create_orientation_function()

        return self._orientation_function

    def get_point_position_name(self) -> str:
        return self._point_position_name


class PlanarTerrain(TerrainDescriptor):
    def create_height_function(self) -> cs.Function:
        point_position = cs.MX.sym(self.get_point_position_name(), 3)

        return cs.Function(
            "planar_terrain_height",
            [point_position],
            [point_position[2]],
            [self.get_point_position_name()],
            ["point_height"],
            self._options,
        )

    def create_normal_direction_function(self) -> cs.Function:
        point_position = cs.MX.sym(self.get_point_position_name(), 3)

        return cs.Function(
            "planar_terrain_normal",
            [point_position],
            [cs.MX.eye(3)[:, 2]],
            [self.get_point_position_name()],
            ["normal_direction"],
            self._options,
        )

    def create_orientation_function(self) -> cs.Function:
        point_position = cs.MX.sym(self.get_point_position_name(), 3)

        return cs.Function(
            "planar_terrain_orientation",
            [point_position],
            [cs.MX.eye(3)],
            [self.get_point_position_name()],
            ["plane_rotation"],
            self._options,
        )
