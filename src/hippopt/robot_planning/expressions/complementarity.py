import casadi as cs

from hippopt.robot_planning.utilities.terrain_descriptor import TerrainDescriptor


def dcc_planar_complementarity(
    terrain: TerrainDescriptor,
    point_position_name: str = None,
    height_multiplier_name: str = "kt",
    point_position_control_name: str = "u_p",
    options: dict = None,
    **_
) -> cs.Function:
    options = {} if options is None else options
    point_position_name = (
        terrain.get_point_position_name()
        if point_position_name is None
        else point_position_name
    )
    point_position = cs.MX.sym(point_position_name, 3)
    height_multiplier = cs.MX.sym(height_multiplier_name, 1)
    point_control = cs.MX.sym(point_position_control_name, 3)

    terrain_orientation_fun = terrain.orientation_function()
    height_function = terrain.height_function()

    height = height_function(point_position)
    terrain_orientation = terrain_orientation_fun(point_position)

    planar_multiplier = cs.tanh(height_multiplier * height)
    multipliers = cs.diag(cs.horzcat([planar_multiplier, planar_multiplier, 1]))
    planar_complementarity = terrain_orientation @ multipliers @ point_control

    return cs.Function(
        "planar_complementarity_dcc",
        [point_position, height_multiplier, point_control],
        [planar_complementarity],
        [point_position_name, height_multiplier_name, point_position_control_name],
        ["planar_complementarity"],
        options,
    )


def dcc_complementarity_margin(
    terrain: TerrainDescriptor,
    point_position_name: str = None,
    point_force_name: str = "point_force",
    point_velocity_name: str = "point_velocity",
    point_force_derivative_name: str = "point_force",
    dcc_gain_name: str = "k_bs",
    epsilon_name: str = "eps",
    options: dict = None,
    **_
) -> cs.Function:
    options = {} if options is None else options
    point_position_name = (
        terrain.get_point_position_name()
        if point_position_name is None
        else point_position_name
    )
    point_position = cs.MX.sym(point_position_name, 3)
    point_force = cs.MX.sym(point_force_name, 3)
    point_velocity = cs.MX.sym(point_velocity_name, 3)
    point_force_derivative = cs.MX.sym(point_force_derivative_name, 3)
    dcc_gain = cs.MX.sym(dcc_gain_name, 1)
    eps = cs.MX.sym(epsilon_name, 1)

    normal_direction_fun = terrain.normal_direction_function()
    height_function = terrain.height_function()

    height = height_function(point_position)
    normal_direction = normal_direction_fun(point_position)

    # See Sec III.A of https://ieeexplore.ieee.org/abstract/document/9847574
    height_derivative = cs.jtimes(height, point_position, point_velocity)
    normal_derivative = cs.jtimes(normal_direction, point_position, point_velocity)
    normal_force = normal_direction.T() @ point_force
    normal_force_derivative = normal_direction.T() @ point_force_derivative
    complementarity = height * normal_force

    csi = (
        height_derivative * normal_force
        + height * point_force.T() @ normal_derivative
        + height * normal_force_derivative
    )

    margin = eps - dcc_gain * complementarity - csi

    return cs.Function(
        "dcc_complementarity_margin",
        [
            point_position,
            point_force,
            point_velocity,
            point_force_derivative,
            dcc_gain,
            eps,
        ],
        [margin],
        [
            point_position_name,
            point_force_name,
            point_velocity_name,
            point_force_derivative_name,
            dcc_gain_name,
            epsilon_name,
        ],
        ["dcc_complementarity_margin"],
        options,
    )
