import casadi as cs
import numpy as np

import hippopt.robot_planning


def test_planar_terrain():
    planar_terrain = hippopt.robot_planning.PlanarTerrain()

    dummy_point = cs.DM.zeros(3)
    dummy_point[2] = 0.5

    height_function = planar_terrain.height_function(point_position_name="p")

    assert height_function(dummy_point) == 0.5

    normal_function = planar_terrain.normal_direction_function()
    normal_direction = np.zeros((3, 1))
    normal_direction[2] = 1.0
    output = normal_function(dummy_point).full()

    assert (normal_direction == output).all()

    orientation_fun = planar_terrain.orientation_function()
    expected_orientation = np.eye(3)

    output = orientation_fun(dummy_point).full()

    assert (expected_orientation == output).all()  # noqa
