import casadi as cs


def centroidal_dynamics_with_point_forces(
    number_of_points: int,
    assume_unitary_mass: bool = False,
    mass_name: str = "m",
    gravity_name: str = "g",
    com_name: str = "com",
    point_position_names: list[str] = None,
    point_force_names: list[str] = None,
    options: dict = None,
    **_,
) -> cs.Function:
    options = {} if options is None else options

    point_position_names = (
        point_position_names
        if point_position_names is not None
        else [f"p{i}" for i in range(number_of_points)]
    )

    if len(point_position_names) != number_of_points:
        raise ValueError(
            f"Expected {number_of_points} point position names,"
            f" got {len(point_position_names)}"
        )

    if point_force_names is None:
        point_force_names = []
        for i in range(number_of_points):
            point_force_names.append(f"f{i}")

    assert len(point_force_names) == number_of_points

    input_vars = []

    if assume_unitary_mass:
        m = 1.0
    else:
        m = cs.MX.sym(mass_name, 1)
        input_vars.append(m)

    g = cs.MX.sym(gravity_name, 6)
    input_vars.append(g)

    x = cs.MX.sym(com_name, 3)
    input_vars.append(x)

    p = []
    f = []
    for point_position_name, point_force_name in zip(
        point_position_names, point_force_names
    ):
        p.append(cs.MX.sym(point_position_name, 3))
        input_vars.append(p[-1])
        f.append(cs.MX.sym(point_force_name, 3))
        input_vars.append(f[-1])

    input_names = [var.name() for var in input_vars]

    h_g = m * g + cs.sum2(
        cs.horzcat(*[cs.vertcat(fi, cs.cross(pi - x, fi)) for fi, pi in zip(f, p)])
    )

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
