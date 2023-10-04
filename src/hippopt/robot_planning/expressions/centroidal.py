import casadi as cs


def centroidal_dynamics_with_point_forces(
    number_of_points: int,
    mass_name: str = "m",
    gravity_name: str = "g",
    com_name: str = "com",
    point_position_names: list[str] = None,
    point_force_names: list[str] = None,
    options: dict = None,
    **_,
) -> cs.Function:
    options = {} if options is None else options

    if point_position_names is None:
        point_position_names = []
        for i in range(number_of_points):
            point_position_names.append(f"p{i}")

    assert len(point_position_names) == number_of_points

    if point_force_names is None:
        point_force_names = []
        for i in range(number_of_points):
            point_force_names.append(f"f{i}")

    assert len(point_force_names) == number_of_points

    input_vars = []

    m = cs.MX.sym(mass_name, 1)
    input_vars.append(m)

    g = cs.MX.sym(gravity_name, 6)
    input_vars.append(g)

    x = cs.MX.sym(com_name, 3)
    input_vars.append(x)

    p = []
    f = []
    for i in range(number_of_points):
        p.append(cs.MX.sym(point_position_names[i], 3))
        input_vars.append(p[i])
        f.append(cs.MX.sym(point_force_names[i], 3))
        input_vars.append(f[i])

    input_names = []
    for var in input_vars:
        input_names.append(var.name())

    h_g = m @ g

    for i in range(number_of_points):
        h_g = h_g + cs.vertcat(f[i], cs.cross(p[i] - x, f[i]))

    return cs.Function(
        "centroidal_dynamics_with_point_forces",
        input_vars,
        [h_g],
        input_names,
        ["h_g_dot"],
        options,
    )


def com_dynamics_from_momentum(
    mass_name: str = "m",
    momentum_name: str = "h_g",
    options: dict = None,
    **_,
) -> cs.Function:
    options = {} if options is None else options

    m = cs.MX.sym(mass_name, 1)
    h_g = cs.MX.sym(momentum_name, 6)

    x_dot = h_g[0:3] / m

    return cs.Function(
        "com_dynamics_from_momentum",
        [m, h_g],
        [x_dot],
        [mass_name, momentum_name],
        ["x_dot"],
        options,
    )
