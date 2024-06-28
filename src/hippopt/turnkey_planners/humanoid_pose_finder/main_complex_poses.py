import logging

import adam.model
import casadi as cs
import hdf5storage
import idyntree.bindings as idyntree
import liecasadi
import numpy as np
import resolve_robotics_uri_py

import hippopt.robot_planning as hp_rp
import hippopt.turnkey_planners.humanoid_pose_finder.planner as pose_finder


def get_planner_settings(
    input_terrain: hp_rp.TerrainDescriptor,
    use_joint_limits: bool = True,
) -> pose_finder.Settings:
    urdf_path = resolve_robotics_uri_py.resolve_robotics_uri(
        "package://ergoCub/robots/ergoCubGazeboV1_minContacts/model.urdf"
    )
    output_settings = pose_finder.Settings()
    output_settings.terrain = input_terrain
    output_settings.robot_urdf = str(urdf_path)
    output_settings.joints_name_list = [
        "torso_pitch",
        "torso_roll",
        "torso_yaw",
        "l_shoulder_pitch",
        "l_shoulder_roll",
        "l_shoulder_yaw",
        "l_elbow",
        "r_shoulder_pitch",
        "r_shoulder_roll",
        "r_shoulder_yaw",
        "r_elbow",
        "l_hip_pitch",
        "l_hip_roll",
        "l_hip_yaw",
        "l_knee",
        "l_ankle_pitch",
        "l_ankle_roll",
        "r_hip_pitch",
        "r_hip_roll",
        "r_hip_yaw",
        "r_knee",
        "r_ankle_pitch",
        "r_ankle_roll",
    ]
    number_of_joints = len(output_settings.joints_name_list)
    idyntree_model_loader = idyntree.ModelLoader()
    idyntree_model_loader.loadReducedModelFromFile(
        output_settings.robot_urdf, output_settings.joints_name_list
    )
    idyntree_model = idyntree_model_loader.model()
    output_settings.root_link = "root_link"
    output_settings.desired_frame_quaternion_cost_frame_name = "chest"
    output_settings.contact_points = hp_rp.FeetContactPointDescriptors()
    output_settings.contact_points.left = hp_rp.ContactPointDescriptor.rectangular_foot(
        foot_frame="l_sole",
        x_length=0.232,
        y_length=0.1,
        top_left_point_position=np.array([0.116, 0.05, 0.0]),
    )
    output_settings.contact_points.right = (
        hp_rp.ContactPointDescriptor.rectangular_foot(
            foot_frame="r_sole",
            x_length=0.232,
            y_length=0.1,
            top_left_point_position=np.array([0.116, 0.05, 0.0]),
        )
    )
    output_settings.relaxed_complementarity_epsilon = 1.0
    output_settings.static_friction = 0.3
    output_settings.maximum_joint_positions = cs.inf * np.ones(number_of_joints)
    output_settings.minimum_joint_positions = -cs.inf * np.ones(number_of_joints)
    for i in range(number_of_joints):
        joint = idyntree_model.getJoint(i)
        if joint.hasPosLimits() and use_joint_limits:
            output_settings.maximum_joint_positions[i] = joint.getMaxPosLimit(i)
            output_settings.minimum_joint_positions[i] = joint.getMinPosLimit(i)
    output_settings.joint_regularization_cost_weights = np.ones(number_of_joints)
    output_settings.joint_regularization_cost_weights[:3] = 0.1  # torso
    output_settings.joint_regularization_cost_weights[3:11] = 10.0  # arms
    output_settings.joint_regularization_cost_weights[11:] = 1.0  # legs
    output_settings.base_quaternion_cost_multiplier = 50.0
    output_settings.desired_frame_quaternion_cost_multiplier = 100.0
    output_settings.joint_regularization_cost_multiplier = 0.1
    output_settings.force_regularization_cost_multiplier = 0.2
    output_settings.com_regularization_cost_multiplier = 10.0
    output_settings.average_force_regularization_cost_multiplier = 10.0
    output_settings.point_position_regularization_cost_multiplier = 100.0
    output_settings.casadi_function_options = {}
    output_settings.casadi_opti_options = {}
    output_settings.casadi_solver_options = {}

    return output_settings


def get_references(
    desired_left_foot_pose: liecasadi.SE3,
    desired_right_foot_pose: liecasadi.SE3,
) -> pose_finder.References:
    output_references = pose_finder.References(
        contact_point_descriptors=planner_settings.contact_points,
        number_of_joints=len(planner_settings.joints_name_list),
    )
    desired_com_position = (
        desired_left_foot_pose.translation() + desired_right_foot_pose.translation()
    ) / 2.0
    desired_com_position[2] = 0.7
    output_references.state.contact_points.left = (
        hp_rp.FootContactState.from_parent_frame_transform(
            descriptor=planner_settings.contact_points.left,
            transform=desired_left_foot_pose,
        )
    )
    output_references.state.contact_points.right = (
        hp_rp.FootContactState.from_parent_frame_transform(
            descriptor=planner_settings.contact_points.right,
            transform=desired_right_foot_pose,
        )
    )
    output_references.state.com = desired_com_position
    output_references.state.kinematics.base.quaternion_xyzw = (
        liecasadi.SO3.Identity().as_quat().coeffs()
    )
    output_references.frame_quaternion_xyzw = (
        liecasadi.SO3.Identity().as_quat().coeffs()
    )
    output_references.state.kinematics.joints.positions = np.deg2rad(
        [
            7,
            0.12,
            -0.01,
            12.0,
            7.0,
            -12.0,
            40.769,
            12.0,
            7.0,
            -12.0,
            40.769,
            5.76,
            1.61,
            -0.31,
            -31.64,
            -20.52,
            -1.52,
            5.76,
            1.61,
            -0.31,
            -31.64,
            -20.52,
            -1.52,
        ]
    )

    return output_references


def get_visualizer_settings(
    input_planner_settings: pose_finder.Settings, robot_model: adam.model.Model
) -> hp_rp.HumanoidStateVisualizerSettings:
    output_settings = hp_rp.HumanoidStateVisualizerSettings()
    output_settings.robot_model = robot_model
    output_settings.considered_joints = input_planner_settings.joints_name_list
    output_settings.contact_points = input_planner_settings.contact_points
    output_settings.terrain = input_planner_settings.terrain
    output_settings.overwrite_terrain_files = True
    output_settings.working_folder = "./"
    return output_settings


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Large step-up with limits

    step_length = 0.45
    step_height = 0.15

    terrain = hp_rp.SmoothTerrain.step(
        length=0.5,
        width=0.8,
        height=step_height,
        position=np.array([0.45, 0.0, 0.0]),
    )
    terrain.set_name(f"high_step_{step_height}")

    planner_settings = get_planner_settings(input_terrain=terrain)

    planner = pose_finder.Planner(settings=planner_settings)

    references = get_references(
        desired_left_foot_pose=liecasadi.SE3.from_translation_and_rotation(
            np.array([0.0, 0.1, 0.0]), liecasadi.SO3.Identity()
        ),
        desired_right_foot_pose=liecasadi.SE3.from_translation_and_rotation(
            np.array([step_length, -0.1, step_height]), liecasadi.SO3.Identity()
        ),
    )

    planner.set_references(references)

    output = planner.solve()

    visualizer_settings = get_visualizer_settings(
        input_planner_settings=planner_settings, robot_model=planner.get_adam_model()
    )

    visualizer = hp_rp.HumanoidStateVisualizer(settings=visualizer_settings)
    visualizer.visualize(output.values.state)  # noqa

    complex_poses = {}
    complex_poses["high_step"] = output.values.state.to_dict(flatten=False)

    hdf5storage.savemat(
        file_name="complex_poses.mat",
        mdict=complex_poses,
        truncate_existing=True,
    )

    print("Press [Enter] to close.")
    input()
