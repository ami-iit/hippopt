import casadi as cs

from hippopt.robot_planning.utilities.terrain_descriptor import TerrainDescriptor


def normal_force_component(
    terrain: TerrainDescriptor,
    point_force_name: str = "point_force",
    options: dict = None,
    **_
):
    options = {} if options is None else options
    point_position = cs.MX.sym(terrain.get_point_position_name(), 3)
    point_force = cs.MX.sym(point_force_name, 3)

    normal_direction_fun = terrain.normal_direction_function()

    normal_component = normal_direction_fun(point_position).T() @ point_force

    return cs.Function(
        "normal_force_component",
        [point_position, point_force],
        [normal_component],
        [terrain.get_point_position_name(), point_force_name],
        ["normal_force"],
        options,
    )


def friction_cone_square_margin(
    terrain: TerrainDescriptor,
    point_force_name: str = "point_force",
    static_friction_name: str = "mu_s",
    options: dict = None,
    **_
):
    options = {} if options is None else options
    point_position = cs.MX.sym(terrain.get_point_position_name(), 3)
    point_force = cs.MX.sym(point_force_name, 3)
    static_friction = cs.MX.sym(static_friction_name, 1)

    orientation_fun = terrain.orientation_function()
    terrain_orientation = orientation_fun(point_position)
    force_in_contact = terrain_orientation.T() @ point_force

    # In principle, it should be sqrt(fx^2 + fy^2) <= u * fz,
    # but since both sides are positive, we square them both.
    # Their difference needs to remain positive, i.e.
    # (u * fz)^2 - (fx^2 + fy^2) >= 0
    margin = cs.sumsqr(static_friction * force_in_contact[2]) - cs.sumsqr(
        force_in_contact[:2]
    )

    return cs.Function(
        "friction_cone_square_margin",
        [point_position, point_force, static_friction],
        [margin],
        [terrain.get_point_position_name(), point_force_name, static_friction_name],
        ["margin"],
        options,
    )
