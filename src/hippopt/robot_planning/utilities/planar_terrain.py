import casadi as cs

from hippopt.robot_planning.utilities.terrain_descriptor import TerrainDescriptor


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
