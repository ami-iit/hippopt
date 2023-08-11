import casadi as cs

from hippopt.robot_planning.utilities.terrain_descriptor import TerrainDescriptor


def normal_force_component(
    terrain: TerrainDescriptor,
    point_position_name: str = None,
    point_force_name: str = "point_force",
    options: dict = None,
    **_,
) -> cs.Function:
    options = {} if options is None else options
    point_position_name = (
        terrain.get_point_position_name()
        if point_position_name is None
        else point_position_name
    )
    point_position = cs.MX.sym(point_position_name, 3)
    point_force = cs.MX.sym(point_force_name, 3)

    normal_direction_fun = terrain.normal_direction_function()

    normal_component = normal_direction_fun(point_position).T() @ point_force

    return cs.Function(
        "normal_force_component",
        [point_position, point_force],
        [normal_component],
        [point_position_name, point_force_name],
        ["normal_force"],
        options,
    )


def friction_cone_square_margin(
    terrain: TerrainDescriptor,
    point_position_name: str = None,
    point_force_name: str = "point_force",
    static_friction_name: str = "mu_s",
    options: dict = None,
    **_,
) -> cs.Function:
    options = {} if options is None else options
    point_position_name = (
        terrain.get_point_position_name()
        if point_position_name is None
        else point_position_name
    )
    point_position = cs.MX.sym(point_position_name, 3)
    point_force = cs.MX.sym(point_force_name, 3)
    static_friction = cs.MX.sym(static_friction_name, 1)

    orientation_fun = terrain.orientation_function()
    terrain_orientation = orientation_fun(point_position)
    force_in_contact = terrain_orientation.T() @ point_force

    # In principle, it should be sqrt(fx^2 + fy^2) <= u * fz,
    # but since both sides are positive, we square them both.
    # Their difference needs to remain positive, i.e.
    # (u * fz)^2 - (fx^2 + fy^2) >= 0
    # that is equal to
    # [-1, -1, u^2] * f.^2
    margin = cs.horzcat([-1, -1, cs.constpow(static_friction, 2)]) * cs.constpow(
        force_in_contact, 2
    )

    return cs.Function(
        "friction_cone_square_margin",
        [point_position, point_force, static_friction],
        [margin],
        [point_position_name, point_force_name, static_friction_name],
        ["friction_cone_square_margin"],
        options,
    )


def contact_points_centroid(
    number_of_points: int,
    point_position_names: list[str] = None,
    options: dict = None,
    **_,
) -> cs.Function:
    options = {} if options is None else options

    if point_position_names is None:
        point_position_names = []
        for i in range(number_of_points):
            point_position_names.append(f"p{i}")

    assert len(point_position_names) == number_of_points

    input_vars = []
    p = []
    for i in range(number_of_points):
        p.append(cs.MX.sym(point_position_names[i], 3))
        input_vars.append(p[i])

    input_names = []
    for var in input_vars:
        input_names.append(var.name())

    centroid = cs.DM.zeros(3, 1)

    for point in p:
        centroid = centroid + point

    if number_of_points > 0:
        centroid = centroid / number_of_points

    return cs.Function(
        "contact_points_centroid",
        input_vars,
        [centroid],
        input_names,
        ["centroid"],
        options,
    )


def contact_points_yaw_alignment_error(
    first_point_name: str = "p_0",
    second_point_name: str = "p_1",
    desired_yaw_name: str = "desired_yaw",
    options: dict = None,
    **_,
) -> cs.Function:
    options = {} if options is None else options

    p0 = cs.MX.sym(first_point_name, 3)
    p1 = cs.MX.sym(second_point_name, 3)
    yaw = cs.MX.sym(desired_yaw_name, 1)

    yaw_alignment = cs.horzcat([-cs.sin(yaw), cs.cos(yaw)]) @ (p1 - p0)[:2]

    return cs.Function(
        "contact_points_yaw_alignment_error",
        [p0, p1, yaw],
        [yaw_alignment],
        [first_point_name, second_point_name, desired_yaw_name],
        ["yaw_alignment_error"],
        options,
    )


def swing_height_heuristic(
    terrain: TerrainDescriptor,
    point_position_name: str = "p",
    point_velocity_name: str = "p_dot",
    desired_height_name: str = "h_desired",
    options: dict = None,
    **_,
) -> cs.Function:
    options = {} if options is None else options

    point = cs.MX.sym(point_position_name, 3)
    point_velocity = cs.MX.sym(point_velocity_name, 3)
    desired_height = cs.MX.sym(desired_height_name, 1)

    height_fun = terrain.height_function()
    terrain_height = height_fun(point)
    terrain_orientation_fun = terrain.orientation_function()
    terrain_orientation = terrain_orientation_fun(point)

    height_difference = terrain_height - desired_height
    planar_velocity = (terrain_orientation.T() @ point_velocity)[:2]

    heuristic = 0.5 * (cs.constpow(height_difference, 2) + cs.sumsqr(planar_velocity))

    return cs.Function(
        "swing_height_heuristic",
        [point, point_velocity, desired_height],
        [heuristic],
        [point_position_name, point_velocity_name, desired_height_name],
        ["heuristic"],
        options,
    )
