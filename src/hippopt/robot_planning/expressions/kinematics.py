import casadi as cs
import liecasadi
from adam.casadi import KinDynComputations

from hippopt.robot_planning.utilities.quaternion import (
    quaternion_xyzw_velocity_to_right_trivialized_angular_velocity,
)


def centroidal_momentum_from_kinematics(
    kindyn_object: KinDynComputations,
    base_position_name: str = "base_position",
    base_quaternion_xyzw_name: str = "base_quaternion",
    joint_positions_name: str = "joint_positions",
    base_position_derivative_name: str = "base_position_derivative",
    base_quaternion_xyzw_derivative_name: str = "base_quaternion_derivative",
    joint_velocities_name: str = "joint_velocities",
    options: dict = None,
    **_
) -> cs.Function:
    options = {} if options is None else options

    base_position = cs.MX.sym(base_position_name, 3)
    base_quaternion = cs.MX.sym(base_quaternion_xyzw_name, 4)
    joint_positions = cs.MX.sym(joint_positions_name, kindyn_object.NDoF)
    base_position_derivative = cs.MX.sym(base_position_derivative_name, 3)
    base_quaternion_derivative = cs.MX.sym(base_quaternion_xyzw_derivative_name, 4)
    joint_velocities = cs.MX.sym(joint_velocities_name, kindyn_object.NDoF)

    cmm_function = kindyn_object.centroidal_momentum_matrix_fun()

    base_pose = liecasadi.SE3.from_position_quaternion(
        base_position, base_quaternion
    ).as_matrix()  # The quaternion is supposed normalized

    base_angular_velocity_fun = (
        quaternion_xyzw_velocity_to_right_trivialized_angular_velocity(
            quaternion_xyzw_name=base_quaternion_xyzw_name,
            base_quaternion_xyzw_derivative_name=base_quaternion_xyzw_derivative_name,
            options=options,
        )
    )
    base_angular_velocity = base_angular_velocity_fun(
        **{
            base_quaternion_xyzw_name: base_quaternion,
            base_quaternion_xyzw_derivative_name: base_quaternion_derivative,
        }
    )

    robot_velocity = cs.vertcat(
        base_position_derivative, base_angular_velocity, joint_velocities
    )

    momentum = cmm_function(base_pose, joint_positions) @ robot_velocity

    return cs.Function(
        "centroidal_momentum_from_kinematics",
        [
            base_position,
            base_quaternion,
            joint_positions,
            base_position_derivative,
            base_quaternion_derivative,
            joint_velocities,
        ],
        [momentum],
        [
            base_position_name,
            base_quaternion_xyzw_name,
            joint_positions_name,
            base_position_derivative_name,
            base_quaternion_xyzw_derivative_name,
            joint_velocities_name,
        ],
        ["h_g"],
        options,
    )
