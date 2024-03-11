import logging

import casadi as cs
import idyntree.bindings as idyntree
import liecasadi
import numpy as np
import resolve_robotics_uri_py

import hippopt
import hippopt.robot_planning as hp_rp
import hippopt.turnkey_planners.humanoid_kinodynamic.planner as walking_planner
import hippopt.turnkey_planners.humanoid_kinodynamic.settings as walking_settings
import hippopt.turnkey_planners.humanoid_kinodynamic.variables as walking_variables
import hippopt.turnkey_planners.humanoid_pose_finder.planner as pose_finder


def get_planner_settings() -> walking_settings.Settings:
    # The model is available at
    # https://github.com/icub-tech-iit/ergocub-gazebo-simulations/tree/↵
    # 1179630a88541479df51ebb108a21865ea251302/models/stickBot
    urdf_path = resolve_robotics_uri_py.resolve_robotics_uri(
        "package://stickBot/model.urdf"
    )
    settings = walking_settings.Settings()
    settings.robot_urdf = str(urdf_path)
    settings.joints_name_list = [
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

    number_of_joints = len(settings.joints_name_list)
    idyntree_model_loader = idyntree.ModelLoader()
    idyntree_model_loader.loadReducedModelFromFile(
        settings.robot_urdf, settings.joints_name_list
    )
    idyntree_model = idyntree_model_loader.model()
    settings.root_link = "root_link"
    settings.horizon_length = 30
    settings.time_step = 0.1
    settings.contact_points = hp_rp.FeetContactPointDescriptors()
    settings.contact_points.left = hp_rp.ContactPointDescriptor.rectangular_foot(
        foot_frame="l_sole",
        x_length=0.232,
        y_length=0.1,
        top_left_point_position=np.array([0.116, 0.05, 0.0]),
    )
    settings.contact_points.right = hp_rp.ContactPointDescriptor.rectangular_foot(
        foot_frame="r_sole",
        x_length=0.232,
        y_length=0.1,
        top_left_point_position=np.array([0.116, 0.05, 0.0]),
    )
    settings.planar_dcc_height_multiplier = 10.0
    settings.dcc_gain = 40.0
    settings.dcc_epsilon = 0.005
    settings.static_friction = 0.3
    settings.maximum_velocity_control = [2.0, 2.0, 5.0]
    settings.maximum_force_derivative = [500.0, 500.0, 500.0]
    settings.maximum_angular_momentum = 5.0
    settings.minimum_com_height = 0.3
    settings.minimum_feet_lateral_distance = 0.1
    settings.maximum_feet_relative_height = 0.05
    settings.maximum_joint_positions = cs.inf * np.ones(number_of_joints)
    settings.minimum_joint_positions = -cs.inf * np.ones(number_of_joints)
    for i in range(number_of_joints):
        joint = idyntree_model.getJoint(i)
        if joint.hasPosLimits():
            settings.maximum_joint_positions[i] = joint.getMaxPosLimit(i)
            settings.minimum_joint_positions[i] = joint.getMinPosLimit(i)
    settings.maximum_joint_velocities = np.ones(number_of_joints) * 2.0
    settings.minimum_joint_velocities = np.ones(number_of_joints) * -2.0
    settings.joint_regularization_cost_weights = np.ones(number_of_joints)
    settings.joint_regularization_cost_weights[:3] = 0.1  # torso
    settings.joint_regularization_cost_weights[3:11] = 10.0  # arms
    settings.joint_regularization_cost_weights[11:] = 1.0  # legs
    settings.contacts_centroid_cost_multiplier = 0.0
    settings.com_linear_velocity_cost_weights = [10.0, 0.1, 1.0]
    settings.com_linear_velocity_cost_multiplier = 1.0
    settings.desired_frame_quaternion_cost_frame_name = "chest"
    settings.desired_frame_quaternion_cost_multiplier = 200.0
    settings.base_quaternion_cost_multiplier = 50.0
    settings.base_quaternion_velocity_cost_multiplier = 0.001
    settings.joint_regularization_cost_multiplier = 10.0
    settings.force_regularization_cost_multiplier = 10.0
    settings.foot_yaw_regularization_cost_multiplier = 2000.0
    settings.swing_foot_height_cost_multiplier = 1000.0
    settings.contact_velocity_control_cost_multiplier = 5.0
    settings.contact_force_control_cost_multiplier = 0.0001
    settings.final_state_expression_type = hippopt.ExpressionType.subject_to
    settings.periodicity_expression_type = hippopt.ExpressionType.subject_to
    settings.casadi_function_options = {"cse": True}
    settings.casadi_opti_options = {"expand": True, "detect_simple_bounds": True}
    settings.casadi_solver_options = {
        "max_iter": 4000,
        "linear_solver": "mumps",
        "alpha_for_y": "dual-and-full",
        "fast_step_computation": "yes",
        "hessian_approximation": "limited-memory",
        "tol": 1e-3,
        "dual_inf_tol": 1000.0,
        "compl_inf_tol": 1e-2,
        "constr_viol_tol": 1e-4,
        "acceptable_tol": 10,
        "acceptable_iter": 2,
        "acceptable_compl_inf_tol": 1000.0,
        "warm_start_bound_frac": 1e-2,
        "warm_start_bound_push": 1e-2,
        "warm_start_mult_bound_push": 1e-2,
        "warm_start_slack_bound_frac": 1e-2,
        "warm_start_slack_bound_push": 1e-2,
        "warm_start_init_point": "yes",
        "required_infeasibility_reduction": 0.8,
        "perturb_dec_fact": 0.1,
        "max_hessian_perturbation": 100.0,
        "acceptable_obj_change_tol": 1e0,
    }

    return settings


def get_pose_finder_settings(
    input_settings: walking_settings.Settings,
) -> pose_finder.Settings:
    number_of_joints = len(input_settings.joints_name_list)
    settings = pose_finder.Settings()
    settings.robot_urdf = input_settings.robot_urdf
    settings.joints_name_list = input_settings.joints_name_list
    settings.parametric_link_names = input_settings.parametric_link_names

    settings.root_link = input_settings.root_link
    settings.desired_frame_quaternion_cost_frame_name = (
        input_settings.desired_frame_quaternion_cost_frame_name
    )

    settings.contact_points = input_settings.contact_points

    settings.relaxed_complementarity_epsilon = 0.0001
    settings.static_friction = input_settings.static_friction

    settings.maximum_joint_positions = input_settings.maximum_joint_positions
    settings.minimum_joint_positions = input_settings.minimum_joint_positions

    settings.joint_regularization_cost_weights = np.ones(number_of_joints)
    settings.joint_regularization_cost_weights[:3] = 0.1  # torso
    settings.joint_regularization_cost_weights[3:11] = 10.0  # arms
    settings.joint_regularization_cost_weights[11:] = 1.0  # legs

    settings.base_quaternion_cost_multiplier = 50.0
    settings.desired_frame_quaternion_cost_multiplier = 100.0
    settings.joint_regularization_cost_multiplier = 0.1
    settings.force_regularization_cost_multiplier = 0.2
    settings.com_regularization_cost_multiplier = 10.0
    settings.average_force_regularization_cost_multiplier = 10.0
    settings.point_position_regularization_cost_multiplier = 100.0
    settings.casadi_function_options = input_settings.casadi_function_options
    settings.casadi_opti_options = input_settings.casadi_opti_options
    settings.casadi_solver_options = {}

    return settings


def get_visualizer_settings(
    input_settings: walking_settings.Settings,
) -> hp_rp.HumanoidStateVisualizerSettings:
    output_viz_settings = hp_rp.HumanoidStateVisualizerSettings()
    output_viz_settings.robot_model = input_settings.robot_urdf
    output_viz_settings.considered_joints = input_settings.joints_name_list
    output_viz_settings.contact_points = input_settings.contact_points
    output_viz_settings.terrain = input_settings.terrain
    output_viz_settings.working_folder = "./"

    return output_viz_settings


def compute_state(
    input_settings: walking_settings.Settings,
    pf_input: pose_finder.Planner,
    desired_com_position: np.ndarray,
    desired_left_foot_pose: liecasadi.SE3,
    desired_right_foot_pose: liecasadi.SE3,
    pf_parametric_link_densities: list[float] | None,
    pf_parametric_link_length_multipliers: list[float] | None,
) -> hp_rp.HumanoidState:
    desired_joints = np.deg2rad(
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
    assert len(input_settings.joints_name_list) == len(desired_joints)

    pf_ref = pose_finder.References(
        contact_point_descriptors=pf_settings.contact_points,
        number_of_joints=len(desired_joints),
    )

    pf_ref.state.com = desired_com_position
    pf_ref.state.contact_points.left = (
        hp_rp.FootContactState.from_parent_frame_transform(
            descriptor=input_settings.contact_points.left,
            transform=desired_left_foot_pose,
        )
    )
    pf_ref.state.contact_points.right = (
        hp_rp.FootContactState.from_parent_frame_transform(
            descriptor=input_settings.contact_points.right,
            transform=desired_right_foot_pose,
        )
    )

    pf_ref.state.kinematics.base.quaternion_xyzw = (
        liecasadi.SO3.Identity().as_quat().coeffs()
    )

    pf_ref.frame_quaternion_xyzw = liecasadi.SO3.Identity().as_quat().coeffs()

    pf_ref.state.kinematics.joints.positions = desired_joints

    pf_guess = pf_input.get_initial_guess()
    pf_guess.references = pf_ref
    if pf_parametric_link_densities is not None:
        pf_guess.parametric_link_densities = pf_parametric_link_densities
    if pf_parametric_link_length_multipliers is not None:
        pf_guess.parametric_link_length_multipliers = (
            pf_parametric_link_length_multipliers
        )
    pf_input.set_initial_guess(pf_guess)

    output_pf = pf_input.solve()
    return output_pf.values.state


def compute_initial_state(
    input_settings: walking_settings.Settings,
    pf_input: pose_finder.Planner,
    contact_guess: hp_rp.FeetContactPhasesDescriptor,
    pf_parametric_link_densities: list[float] | None,
    pf_parametric_link_length_multipliers: list[float] | None,
) -> walking_variables.ExtendedHumanoidState:
    desired_left_foot_pose = contact_guess.left[0].transform
    desired_right_foot_pose = contact_guess.right[0].transform
    desired_com_position = (
        desired_left_foot_pose.translation() + desired_right_foot_pose.translation()
    ) / 2.0
    desired_com_position[2] = 0.7
    output_pf = compute_state(
        input_settings=input_settings,
        pf_input=pf_input,
        desired_com_position=desired_com_position,
        desired_left_foot_pose=desired_left_foot_pose,
        desired_right_foot_pose=desired_right_foot_pose,
        pf_parametric_link_densities=pf_parametric_link_densities,
        pf_parametric_link_length_multipliers=pf_parametric_link_length_multipliers,
    )

    output_state = walking_variables.ExtendedHumanoidState()
    output_state.contact_points = output_pf.contact_points
    output_state.kinematics = output_pf.kinematics
    output_state.com = output_pf.com

    output_state.centroidal_momentum = np.zeros((6, 1))

    return output_state


def compute_middle_state(
    input_settings: walking_settings.Settings,
    pf_input: pose_finder.Planner,
    contact_guess: hp_rp.FeetContactPhasesDescriptor,
    pf_parametric_link_densities: list[float] | None,
    pf_parametric_link_length_multipliers: list[float] | None,
) -> hp_rp.HumanoidState:
    desired_left_foot_pose = contact_guess.left[1].transform
    desired_right_foot_pose = contact_guess.right[0].transform
    desired_com_position = (
        desired_left_foot_pose.translation() + desired_right_foot_pose.translation()
    ) / 2.0
    desired_com_position[2] = 0.7
    return compute_state(
        input_settings=input_settings,
        pf_input=pf_input,
        desired_com_position=desired_com_position,
        desired_left_foot_pose=desired_left_foot_pose,
        desired_right_foot_pose=desired_right_foot_pose,
        pf_parametric_link_densities=pf_parametric_link_densities,
        pf_parametric_link_length_multipliers=pf_parametric_link_length_multipliers,
    )


def compute_final_state(
    input_settings: walking_settings.Settings,
    pf_input: pose_finder.Planner,
    contact_guess: hp_rp.FeetContactPhasesDescriptor,
    pf_parametric_link_densities: list[float] | None,
    pf_parametric_link_length_multipliers: list[float] | None,
) -> hp_rp.HumanoidState:
    desired_left_foot_pose = contact_guess.left[1].transform
    desired_right_foot_pose = contact_guess.right[1].transform
    desired_com_position = (
        desired_left_foot_pose.translation() + desired_right_foot_pose.translation()
    ) / 2.0
    desired_com_position[2] = 0.7
    return compute_state(
        input_settings=input_settings,
        pf_input=pf_input,
        desired_com_position=desired_com_position,
        desired_left_foot_pose=desired_left_foot_pose,
        desired_right_foot_pose=desired_right_foot_pose,
        pf_parametric_link_densities=pf_parametric_link_densities,
        pf_parametric_link_length_multipliers=pf_parametric_link_length_multipliers,
    )


def get_references(
    input_settings: walking_settings.Settings,
    desired_states: list[hp_rp.HumanoidState],
) -> list[walking_variables.References]:
    output_list = []

    for i in range(input_settings.horizon_length):
        output_reference = walking_variables.References(
            number_of_joints=len(input_settings.joints_name_list),
            number_of_points_left=len(input_settings.contact_points.left),
            number_of_points_right=len(input_settings.contact_points.right),
        )

        output_reference.contacts_centroid_cost_weights = [100, 100, 10]
        output_reference.contacts_centroid = [0.3, 0.0, 0.0]
        output_reference.joint_regularization = desired_states[
            i
        ].kinematics.joints.positions
        output_reference.com_linear_velocity = [0.1, 0.0, 0.0]
        output_list.append(output_reference)

    return output_list


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    planner_settings = get_planner_settings()

    planner_settings.parametric_link_names = [
        "r_upper_arm",
        "r_forearm",
        "l_hip_3",
        "l_lower_leg",
        "root_link",
        "torso_1",
        "torso_2",
        "chest",
    ]

    parametric_link_densities = [
        1578.8230690646876,
        687.9855671524874,
        568.2817642184916,
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
        2.0,
    ]

    planner = walking_planner.Planner(settings=planner_settings)

    pf_settings = get_pose_finder_settings(input_settings=planner_settings)
    pf = pose_finder.Planner(settings=pf_settings)

    horizon = planner_settings.horizon_length * planner_settings.time_step

    step_length = 0.6

    contact_phases_guess = hp_rp.FeetContactPhasesDescriptor()
    contact_phases_guess.left = [
        hp_rp.FootContactPhaseDescriptor(
            transform=liecasadi.SE3.from_translation_and_rotation(
                np.array([0.0, 0.1, 0.0]), liecasadi.SO3.Identity()
            ),
            mid_swing_transform=liecasadi.SE3.from_translation_and_rotation(
                np.array([step_length / 2, 0.1, 0.05]), liecasadi.SO3.Identity()
            ),
            force=np.array([0, 0, 100.0]),
            activation_time=None,
            deactivation_time=horizon / 6.0,
        ),
        hp_rp.FootContactPhaseDescriptor(
            transform=liecasadi.SE3.from_translation_and_rotation(
                np.array([step_length, 0.1, 0.0]), liecasadi.SO3.Identity()
            ),
            mid_swing_transform=None,
            force=np.array([0, 0, 100.0]),
            activation_time=horizon / 3.0,
            deactivation_time=None,
        ),
    ]

    contact_phases_guess.right = [
        hp_rp.FootContactPhaseDescriptor(
            transform=liecasadi.SE3.from_translation_and_rotation(
                np.array([step_length / 2, -0.1, 0.0]), liecasadi.SO3.Identity()
            ),
            mid_swing_transform=liecasadi.SE3.from_translation_and_rotation(
                np.array([step_length, -0.1, 0.05]), liecasadi.SO3.Identity()
            ),
            force=np.array([0, 0, 100.0]),
            activation_time=None,
            deactivation_time=horizon * 2.0 / 3.0,
        ),
        hp_rp.FootContactPhaseDescriptor(
            transform=liecasadi.SE3.from_translation_and_rotation(
                np.array([1.5 * step_length, -0.1, 0.0]), liecasadi.SO3.Identity()
            ),
            mid_swing_transform=None,
            force=np.array([0, 0, 100.0]),
            activation_time=horizon * 5.0 / 6.0,
            deactivation_time=None,
        ),
    ]

    initial_state = compute_initial_state(
        input_settings=planner_settings,
        pf_input=pf,
        contact_guess=contact_phases_guess,
        pf_parametric_link_densities=parametric_link_densities,
        pf_parametric_link_length_multipliers=parametric_link_length_multipliers,
    )

    final_state = compute_final_state(
        input_settings=planner_settings,
        pf_input=pf,
        contact_guess=contact_phases_guess,
        pf_parametric_link_densities=parametric_link_densities,
        pf_parametric_link_length_multipliers=parametric_link_length_multipliers,
    )
    final_state.centroidal_momentum = np.zeros((6, 1))

    middle_state = compute_middle_state(
        input_settings=planner_settings,
        pf_input=pf,
        contact_guess=contact_phases_guess,
        pf_parametric_link_densities=parametric_link_densities,
        pf_parametric_link_length_multipliers=parametric_link_length_multipliers,
    )

    first_half_guess_length = planner_settings.horizon_length // 2
    first_half_guess = hp_rp.humanoid_state_interpolator(
        initial_state=initial_state,
        final_state=middle_state,
        contact_phases=contact_phases_guess,
        contact_descriptor=planner_settings.contact_points,
        number_of_points=first_half_guess_length,
        dt=planner_settings.time_step,
    )

    second_half_guess_length = planner_settings.horizon_length - first_half_guess_length
    second_half_guess = hp_rp.humanoid_state_interpolator(
        initial_state=middle_state,
        final_state=final_state,
        contact_phases=contact_phases_guess,
        contact_descriptor=planner_settings.contact_points,
        number_of_points=second_half_guess_length,
        dt=planner_settings.time_step,
        t0=first_half_guess_length * planner_settings.time_step,
    )

    guess = first_half_guess + second_half_guess

    print("Starting the planner...")

    references = get_references(
        input_settings=planner_settings,
        desired_states=guess,
    )

    planner.set_references(references)
    planner_guess = planner.get_initial_guess()
    planner_guess.system = [
        walking_variables.ExtendedHumanoid.from_humanoid_state(s) for s in guess
    ]
    planner_guess.initial_state = initial_state
    planner_guess.final_state = final_state
    if parametric_link_densities is not None:
        planner_guess.parametric_link_densities = parametric_link_densities
    if parametric_link_length_multipliers is not None:
        planner_guess.parametric_link_length_multipliers = (
            parametric_link_length_multipliers
        )
    planner.set_initial_guess(planner_guess)

    output = planner.solve()

    humanoid_states = [s.to_humanoid_state() for s in output.values.system]
    left_contact_points = [s.contact_points.left for s in humanoid_states]
    right_contact_points = [s.contact_points.right for s in humanoid_states]

    visualizer_settings = get_visualizer_settings(input_settings=planner_settings)
    visualizer_settings.robot_model = planner.get_adam_model()
    visualizer = hp_rp.HumanoidStateVisualizer(settings=visualizer_settings)
    print("Press [Enter] to visualize the solution.")
    input()

    visualizer.visualize(
        states=humanoid_states,
        timestep_s=output.values.dt,
        time_multiplier=1.0,
        save=True,
        file_name_stem="humanoid_walking_periodic_parametric",
    )
