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

    # The model is available at
    # https://github.com/icub-tech-iit/ergocub-gazebo-simulations/tree/↵
    # 1179630a88541479df51ebb108a21865ea251302/models/stickBot
    urdf_path = resolve_robotics_uri_py.resolve_robotics_uri(
        "package://stickBot/model.urdf"
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

    planner_settings.parametric_link_names = [
        "r_upper_arm",
        "l_upper_arm",
        "r_lower_leg",
        "l_lower_leg",
        "root_link",
        "torso_1",
        "torso_2",
        "chest",
    ]

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

    parametric_link_densities = [
        1578.8230690646876,
        1578.8230690646876,
        1907.2410446310623,
        1907.2410446310623,
        2013.8319822728106,
        1134.0550335996697,
        844.6779189491116,
        628.0724496264946,
    ]
    parametric_link_length_multipliers = [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    initial_guess = planner.get_initial_guess()
    initial_guess.parametric_link_length_multipliers = (
        parametric_link_length_multipliers
    )
    initial_guess.parametric_link_densities = parametric_link_densities
    initial_guess.references = references
    planner.set_initial_guess(initial_guess)
    output_ipopt = planner.solve()
    planner.change_opti_options(
        inner_solver="sqpmethod",
        options_solver={},
        options_plugin={"qpsol": "qrqp"},
    )
    # planner.set_initial_guess(output_ipopt.values)

    planner_function = planner.to_function(input_name_prefix="guess.")

    parametric_link_length_multipliers_sym = cs.MX.sym(
        "parametric_link_length_multipliers",
        len(planner_settings.parametric_link_names),
    )
    parametric_link_densities_sym = cs.MX.sym(
        "parametric_link_densities", len(planner_settings.parametric_link_names)
    )

    initial_guess = output_ipopt.values
    initial_guess.parametric_link_length_multipliers = (
        parametric_link_length_multipliers_sym
    )
    initial_guess.parametric_link_densities = parametric_link_densities_sym
    initial_guess_dict = output_ipopt.values.to_dict(prefix="guess.")

    output_dict = planner_function(**initial_guess_dict)
    output = planner.get_variables_structure()
    output.from_dict(output_dict)
    com_height = output.state.com[2]

    com_height_function = cs.Function(
        "com_height_function",
        [parametric_link_length_multipliers_sym, parametric_link_densities_sym],
        [com_height],
    )

    com_height = com_height_function(
        parametric_link_length_multipliers,
        parametric_link_densities,
    )

    com_height_jacobian_function = com_height_function.jacobian()
    link_jacobian, density_jacobian = com_height_jacobian_function(
        parametric_link_length_multipliers, parametric_link_densities, com_height
    )

    print(f"Com Height: {com_height}")
    print(f"Link Jacobian: {link_jacobian}")
    print(f"Density Jacobian: {density_jacobian}")

    visualizer_settings = hp_rp.HumanoidStateVisualizerSettings()
    visualizer_settings.robot_model = planner.get_adam_model()
    visualizer_settings.considered_joints = planner_settings.joints_name_list
    visualizer_settings.contact_points = planner_settings.contact_points
    visualizer_settings.terrain = planner_settings.terrain
    visualizer_settings.working_folder = "./"
    visualizer = hp_rp.HumanoidStateVisualizer(settings=visualizer_settings)

    min_sensitivity = np.min(link_jacobian)
    print("Min length sensitivity: ", min_sensitivity)
    max_sensitivity = np.max(link_jacobian)
    print("Max length sensitivity: ", max_sensitivity)

    for i, link in enumerate(planner_settings.parametric_link_names):
        sensitivity = link_jacobian[i].full().item()
        alpha = (sensitivity - min_sensitivity) / (max_sensitivity - min_sensitivity)
        visualizer.change_link_color(link, [alpha, 1 - alpha, 0.0, 1])

    visualizer.visualize(
        states=output_ipopt.values.state, save=True, file_name_stem="length_sensitivity"
    )

    print("Press [Enter] to visualize density sensitivity.")
    input()

    min_sensitivity = np.min(density_jacobian)
    print("Min density sensitivity: ", min_sensitivity)
    max_sensitivity = np.max(density_jacobian)
    print("Max density sensitivity: ", max_sensitivity)

    for i, link in enumerate(planner_settings.parametric_link_names):
        sensitivity = density_jacobian[i].full().item()
        alpha = (sensitivity - min_sensitivity) / (max_sensitivity - min_sensitivity)
        visualizer.change_link_color(link, [alpha, 1 - alpha, 0.0, 1])

    visualizer.visualize(
        states=output_ipopt.values.state,
        save=True,
        file_name_stem="density_sensitivity",
    )

    print("Press [Enter] to close.")
    input()
