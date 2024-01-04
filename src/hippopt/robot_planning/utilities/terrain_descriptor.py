import abc
import dataclasses

import casadi as cs


@dataclasses.dataclass
class TerrainDescriptor(abc.ABC):
    _height_function: cs.Function | None = dataclasses.field(default=None)
    _normal_direction_function: cs.Function | None = dataclasses.field(default=None)
    _orientation_function: cs.Function | None = dataclasses.field(default=None)
    _point_position_name: str = dataclasses.field(default="point_position")
    _name: str = dataclasses.field(default=None)
    _options: dict = dataclasses.field(default_factory=dict)
    point_position_name: dataclasses.InitVar[str] = dataclasses.field(
        default="point_position"
    )
    options: dataclasses.InitVar[dict] = dataclasses.field(default=None)
    name: dataclasses.InitVar[str] = dataclasses.field(default=None)

    def __post_init__(
        self,
        point_position_name: str = None,
        options: dict = None,
        name: str = None,
    ):
        self.change_options(point_position_name, options)
        if name is not None:
            self._name = name

    def change_options(
        self, point_position_name: str = None, options: dict = None, **_
    ) -> None:
        if options is not None:
            self._options = options
        if point_position_name is not None:
            self._point_position_name = point_position_name

    @abc.abstractmethod
    def create_height_function(self) -> cs.Function:
        pass

    def create_normal_direction_function(self) -> cs.Function:
        point_position = cs.MX.sym(self.get_point_position_name(), 3)

        # The normal direction is the gradient of the implicit function h(x, y, z) = 0
        height_gradient = cs.gradient(
            self.height_function()(point_position), point_position
        )
        normal_direction = height_gradient / cs.norm_2(height_gradient)

        return cs.Function(
            "terrain_normal",
            [point_position],
            [normal_direction],
            [self.get_point_position_name()],
            ["normal_direction"],
            self._options,
        )

    def create_orientation_function(self) -> cs.Function:
        point_position = cs.MX.sym(self.get_point_position_name(), 3)

        normal_direction = self.normal_direction_function()(point_position)
        y_direction = cs.cross(normal_direction, cs.DM([1, 0, 0]))
        x_direction = cs.cross(y_direction, normal_direction)
        x_direction = x_direction / cs.norm_2(x_direction)
        # Make sure the y direction is orthogonal even after the transformation
        y_direction = cs.cross(normal_direction, x_direction)

        return cs.Function(
            "terrain_orientation",
            [point_position],
            [cs.horzcat(x_direction, y_direction, normal_direction)],
            [self.get_point_position_name()],
            ["plane_rotation"],
            self._options,
        )

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

    def get_name(self) -> str:
        return self._name if isinstance(self._name, str) else self.__class__.__name__

    def invalidate_functions(self) -> None:
        self._height_function = None
        self._normal_direction_function = None
        self._orientation_function = None
