import abc

import casadi as cs


class TerrainDescriptor(abc.ABC):
    @abc.abstractmethod
    def height_function(
        self, point_position_name: str = "point_position", options: dict = None
    ) -> cs.Function:
        pass

    @abc.abstractmethod
    def normal_direction_function(
        self, point_position_name: str = "point_position_name", options: dict = None
    ) -> cs.Function:
        pass

    @abc.abstractmethod
    def orientation_function(
        self, point_position_name: str = "point_position_name", options: dict = None
    ) -> cs.Function:
        pass


class PlanarTerrain(TerrainDescriptor):
    def height_function(
        self, point_position_name: str = "point_position", options: dict = None
    ) -> cs.Function:
        options = {} if options is None else options
        point_position = cs.MX.sym(point_position_name, 3)

        return cs.Function(
            "planar_terrain_height",
            [point_position],
            [point_position[2]],
            [point_position_name],
            ["point_height"],
            options,
        )

    def normal_direction_function(
        self, point_position_name: str = "point_position_name", options: dict = None
    ) -> cs.Function:
        options = {} if options is None else options
        point_position = cs.MX.sym(point_position_name, 3)

        return cs.Function(
            "planar_terrain_normal",
            [point_position],
            [cs.MX.eye(3)[:, 2]],
            [point_position_name],
            ["normal_direction"],
            options,
        )

    def orientation_function(
        self, point_position_name: str = "point_position_name", options: dict = None
    ) -> cs.Function:
        options = {} if options is None else options
        point_position = cs.MX.sym(point_position_name, 3)

        return cs.Function(
            "planar_terrain_orientation",
            [point_position],
            [cs.MX.eye(3)],
            [point_position_name],
            ["plane_rotation"],
            options,
        )
