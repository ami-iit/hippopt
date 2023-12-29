import abc
import dataclasses

import casadi as cs


@dataclasses.dataclass
class TerrainDescriptor(abc.ABC):
    _height_function: cs.Function = dataclasses.field(default=None)
    _normal_direction_function: cs.Function = dataclasses.field(default=None)
    _orientation_function: cs.Function = dataclasses.field(default=None)
    _point_position_name: str = dataclasses.field(default="point_position")
    _name: str = dataclasses.field(default="terrain")
    _options: dict = dataclasses.field(default=None)
    point_position_name: dataclasses.InitVar[str] = dataclasses.field(
        default="point_position"
    )
    options: dataclasses.InitVar[dict] = dataclasses.field(default=None)
    name: dataclasses.InitVar[str] = dataclasses.field(default=None)

    def __post_init__(
        self,
        point_position_name: str = "point_position",
        options: dict = None,
        name: str = None,
    ):
        self.change_options(point_position_name, options)
        self._name = name

    def change_options(
        self, point_position_name: str = "point_position", options: dict = None, **_
    ) -> None:
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

    def get_name(self) -> str:
        return self._name if isinstance(self._name, str) else self.__class__.__name__
