import casadi as cs
import liecasadi
from adam.casadi import KinDynComputations

from hippopt.robot_planning.expressions.quaternion import (
    quaternion_xyzw_error,
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
            quaternion_xyzw_velocity_name=base_quaternion_xyzw_derivative_name,
            options=options,
        )
    )
    base_angular_velocity = base_angular_velocity_fun(
        **{
            base_quaternion_xyzw_name: base_quaternion,
            base_quaternion_xyzw_derivative_name: base_quaternion_derivative,
        }
    )["right_trivialized_angular_velocity"]

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


def center_of_mass_position_from_kinematics(
    kindyn_object: KinDynComputations,
    base_position_name: str = "base_position",
    base_quaternion_xyzw_name: str = "base_quaternion",
    joint_positions_name: str = "joint_positions",
    options: dict = None,
    **_
) -> cs.Function:
    options = {} if options is None else options

    base_position = cs.MX.sym(base_position_name, 3)
    base_quaternion = cs.MX.sym(base_quaternion_xyzw_name, 4)
    joint_positions = cs.MX.sym(joint_positions_name, kindyn_object.NDoF)

    com_function = kindyn_object.CoM_position_fun()

    base_pose = liecasadi.SE3.from_position_quaternion(
        base_position, base_quaternion
    ).as_matrix()  # The quaternion is supposed normalized

    com_position = com_function(base_pose, joint_positions)

    return cs.Function(
        "center_of_mass_position_from_kinematics",
        [
            base_position,
            base_quaternion,
            joint_positions,
        ],
        [com_position],
        [
            base_position_name,
            base_quaternion_xyzw_name,
            joint_positions_name,
        ],
        ["com_position"],
        options,
    )


def point_position_from_kinematics(
    kindyn_object: KinDynComputations,
    frame_name: str,
    point_position_in_frame_name: str = "point_position",
    base_position_name: str = "base_position",
    base_quaternion_xyzw_name: str = "base_quaternion",
    joint_positions_name: str = "joint_positions",
    options: dict = None,
    **_
) -> cs.Function:
    options = {} if options is None else options

    base_position = cs.MX.sym(base_position_name, 3)
    base_quaternion = cs.MX.sym(base_quaternion_xyzw_name, 4)
    joint_positions = cs.MX.sym(joint_positions_name, kindyn_object.NDoF)
    point_position_in_frame = cs.MX.sym(point_position_in_frame_name, 3)

    fk_function = kindyn_object.forward_kinematics_fun(frame=frame_name)

    base_pose = liecasadi.SE3.from_position_quaternion(
        base_position, base_quaternion
    ).as_matrix()  # The quaternion is supposed normalized

    frame_pose = fk_function(base_pose, joint_positions)

    point_position = frame_pose[:3, :3] @ point_position_in_frame + frame_pose[:3, 3]

    return cs.Function(
        "point_position_from_kinematics",
        [
            base_position,
            base_quaternion,
            joint_positions,
            point_position_in_frame,
        ],
        [point_position],
        [
            base_position_name,
            base_quaternion_xyzw_name,
            joint_positions_name,
            point_position_in_frame_name,
        ],
        ["point_position"],
        options,
    )


def frames_relative_position(
    kindyn_object: KinDynComputations,
    reference_frame: str,
    target_frame: str,
    joint_positions_name: str = "joint_positions",
    options: dict = None,
    **_
) -> cs.Function:
    options = {} if options is None else options
    joint_positions = cs.MX.sym(joint_positions_name, kindyn_object.NDoF)

    base_pose = cs.DM_eye(4)

    reference_fk_function = kindyn_object.forward_kinematics_fun(frame=reference_frame)
    target_fk_function = kindyn_object.forward_kinematics_fun(frame=target_frame)

    reference_pose = liecasadi.SE3.from_matrix(
        reference_fk_function(base_pose, joint_positions)
    )
    reference_pose_inverse = reference_pose.inverse()
    target_pose = target_fk_function(base_pose, joint_positions)

    relative_position = (
        reference_pose_inverse.rotation().act(target_pose[:3, 3])
        + reference_pose_inverse.translation()
    )

    return cs.Function(
        "frames_relative_position",
        [joint_positions],
        [relative_position],
        [joint_positions_name],
        ["relative_position"],
        options,
    )


def rotation_error_from_kinematics(
    kindyn_object: KinDynComputations,
    target_frame: str,
    base_position_name: str = "base_position",
    base_quaternion_xyzw_name: str = "base_quaternion",
    joint_positions_name: str = "joint_positions",
    desired_quaternion_xyzw_name: str = "desired_quaternion",
    options: dict = None,
    **_
) -> cs.Function:
    options = {} if options is None else options
    base_position = cs.MX.sym(base_position_name, 3)
    base_quaternion = cs.MX.sym(base_quaternion_xyzw_name, 4)
    joint_positions = cs.MX.sym(joint_positions_name, kindyn_object.NDoF)
    desired_quaternion = cs.MX.sym(desired_quaternion_xyzw_name, 4)

    base_pose = liecasadi.SE3.from_position_quaternion(
        base_position, base_quaternion
    ).as_matrix()  # The quaternion is supposed normalized

    target_fk_function = kindyn_object.forward_kinematics_fun(frame=target_frame)
    target_pose = target_fk_function(base_pose, joint_positions)
    target_orientation = target_pose[:3, :3]

    rotation_error = (
        target_orientation @ liecasadi.SO3.from_quat(desired_quaternion).as_matrix().T
    )

    return cs.Function(
        "quaternion_error_from_kinematics",
        [
            base_position,
            base_quaternion,
            joint_positions,
            desired_quaternion,
        ],
        [rotation_error],
        [
            base_position_name,
            base_quaternion_xyzw_name,
            joint_positions_name,
            desired_quaternion_xyzw_name,
        ],
        ["rotation_error"],
        options,
    )
