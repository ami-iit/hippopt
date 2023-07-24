import casadi as cs
import liecasadi


def quaternion_xyzw_normalization(
    quaternion_xyzw_name: str = "quaternion",
    options: dict = None,
    **_,
) -> cs.Function:
    options = {} if options is None else options
    quaternion = cs.MX.sym(quaternion_xyzw_name, 4)

    normalized_quaternion = liecasadi.Quaternion(xyzw=quaternion).normalize()

    return cs.Function(
        "quaternion_xyzw_normalization",
        [quaternion],
        [normalized_quaternion.xyzw],
        [quaternion_xyzw_name],
        ["quaternion_normalized"],
        options,
    )


def quaternion_xyzw_velocity_to_right_trivialized_angular_velocity(
    quaternion_xyzw_name: str = "quaternion",
    quaternion_xyzw_velocity_name: str = "quaternion_velocity",
    options: dict = None,
    **_,
) -> cs.Function:
    options = {} if options is None else options
    quaternion = cs.MX.sym(quaternion_xyzw_name, 4)
    quaternion_velocity = cs.MX.sym(quaternion_xyzw_velocity_name, 4)

    q_w = quaternion[3]
    q_i = quaternion[1:3]

    q_dot_w = quaternion_velocity[3]
    q_dot_i = quaternion_velocity[1:3]

    # See Sec. 1.5.3 of https://arxiv.org/pdf/0811.2889.pdf
    angular_velocity = 2 * (-q_dot_w * q_i + q_w * q_dot_i - q_dot_i.cross(q_i))

    return cs.Function(
        "quaternion_xyzw_velocity_to_right_trivialized_angular_velocity",
        [quaternion, quaternion_velocity],
        [angular_velocity],
        [quaternion_xyzw_name, quaternion_xyzw_velocity_name],
        ["right_trivialized_angular_velocity"],
        options,
    )
