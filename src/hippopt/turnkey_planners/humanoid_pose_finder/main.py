import logging

import casadi as cs
import idyntree.bindings as idyntree
import liecasadi
import numpy as np
import resolve_robotics_uri_py

import hippopt.robot_planning as hp_rp
import hippopt.turnkey_planners.humanoid_pose_finder.planner as pose_finder

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    urdf_path = resolve_robotics_uri_py.resolve_robotics_uri(
        "package://ergoCub/robots/ergoCubGazeboV1_minContacts/model.urdf"
    )

    planner_settings = pose_finder.Settings()
    planner_settings.robot_urdf = str(urdf_path)
    planner_settings.joints_name_list = [
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
    number_of_joints = len(planner_settings.joints_name_list)

    idyntree_model_loader = idyntree.ModelLoader()
    idyntree_model_loader.loadReducedModelFromFile(
        planner_settings.robot_urdf, planner_settings.joints_name_list
    )
    idyntree_model = idyntree_model_loader.model()

    planner_settings.root_link = "root_link"
    planner_settings.desired_frame_quaternion_cost_frame_name = "chest"

    planner_settings.contact_points = hp_rp.FeetContactPointDescriptors()
    planner_settings.contact_points.left = (
        hp_rp.ContactPointDescriptor.rectangular_foot(
            foot_frame="l_sole",
            x_length=0.232,
            y_length=0.1,
            top_left_point_position=np.array([0.116, 0.05, 0.0]),
        )
    )
    planner_settings.contact_points.right = (
        hp_rp.ContactPointDescriptor.rectangular_foot(
            foot_frame="r_sole",
            x_length=0.232,
            y_length=0.1,
            top_left_point_position=np.array([0.116, 0.05, 0.0]),
        )
    )

    planner_settings.relaxed_complementarity_epsilon = 0.0001
    planner_settings.static_friction = 0.3

    planner_settings.maximum_joint_positions = cs.inf * np.ones(number_of_joints)
    planner_settings.minimum_joint_positions = -cs.inf * np.ones(number_of_joints)

    for i in range(number_of_joints):
        joint = idyntree_model.getJoint(i)
        if joint.hasPosLimits():
            planner_settings.maximum_joint_positions[i] = joint.getMaxPosLimit(i)
            planner_settings.minimum_joint_positions[i] = joint.getMinPosLimit(i)

    planner_settings.joint_regularization_cost_weights = np.ones(number_of_joints)
    planner_settings.joint_regularization_cost_weights[:3] = 0.1  # torso
    planner_settings.joint_regularization_cost_weights[3:11] = 10.0  # arms
    planner_settings.joint_regularization_cost_weights[11:] = 1.0  # legs

    planner_settings.base_quaternion_cost_multiplier = 50.0
    planner_settings.desired_frame_quaternion_cost_multiplier = 100.0
    planner_settings.joint_regularization_cost_multiplier = 0.1
    planner_settings.force_regularization_cost_multiplier = 0.2
    planner_settings.com_regularization_cost_multiplier = 10.0
    planner_settings.average_force_regularization_cost_multiplier = 10.0
    planner_settings.point_position_regularization_cost_multiplier = 100.0
    planner_settings.casadi_function_options = {}
    planner_settings.casadi_opti_options = {}
    planner_settings.casadi_solver_options = {}

    planner = pose_finder.Planner(settings=planner_settings)

    references = pose_finder.References(
        contact_point_descriptors=planner_settings.contact_points,
        number_of_joints=number_of_joints,
    )

    references.state.com = np.array([0.0, 0.0, 0.7])
    desired_left_foot_pose = liecasadi.SE3.from_translation_and_rotation(
        np.array([0.0, 0.1, 0.0]), liecasadi.SO3.Identity()
    )
    desired_right_foot_pose = liecasadi.SE3.from_translation_and_rotation(
        np.array([0.0, -0.1, 0.0]), liecasadi.SO3.Identity()
    )
    references.state.contact_points.left = (
        hp_rp.FootContactState.from_parent_frame_transform(
            descriptor=planner_settings.contact_points.left,
            transform=desired_left_foot_pose,
        )
    )
    references.state.contact_points.right = (
        hp_rp.FootContactState.from_parent_frame_transform(
            descriptor=planner_settings.contact_points.right,
            transform=desired_right_foot_pose,
        )
    )

    references.state.kinematics.base.quaternion_xyzw = (
        liecasadi.SO3.Identity().as_quat().coeffs()
    )

    references.frame_quaternion_xyzw = liecasadi.SO3.Identity().as_quat().coeffs()

    references.state.kinematics.joints.positions = np.deg2rad(
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

    planner.set_references(references)

    output = planner.solve()

    visualizer_settings = hp_rp.HumanoidStateVisualizerSettings()
    visualizer_settings.robot_model = planner.get_adam_model()
    visualizer_settings.considered_joints = planner_settings.joints_name_list
    visualizer_settings.contact_points = planner_settings.contact_points
    visualizer_settings.terrain = planner_settings.terrain
    visualizer_settings.working_folder = "./"

    visualizer = hp_rp.HumanoidStateVisualizer(settings=visualizer_settings)
    visualizer.visualize(output.values.state)  # noqa

    print("Press [Enter] to move to the next pose.")
    input()

    references.com = np.array([0.15, 0.0, 0.7])
    desired_left_foot_pose = liecasadi.SE3.from_translation_and_rotation(
        np.array([0.3, 0.1, 0.0]), liecasadi.SO3.Identity()
    )
    desired_right_foot_pose = liecasadi.SE3.from_translation_and_rotation(
        np.array([0.0, -0.1, 0.0]), liecasadi.SO3.Identity()
    )
    references.state.contact_points.left = (
        hp_rp.FootContactState.from_parent_frame_transform(
            descriptor=planner_settings.contact_points.left,
            transform=desired_left_foot_pose,
        )
    )
    references.state.contact_points.right = (
        hp_rp.FootContactState.from_parent_frame_transform(
            descriptor=planner_settings.contact_points.right,
            transform=desired_right_foot_pose,
        )
    )

    planner.set_references(references)

    output = planner.solve()
    visualizer.visualize(output.values.state)

    print("Press [Enter] to close.")
    input()
