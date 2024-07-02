import math

import adam.model
import hdf5storage
import idyntree.bindings as idyntree
import liecasadi
import numpy as np
import resolve_robotics_uri_py

import hippopt
import hippopt.robot_planning as hp_rp
import hippopt.turnkey_planners.humanoid_pose_finder.planner as pose_finder

joint_names = [
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


def get_planner_settings(
    input_terrain: hp_rp.TerrainDescriptor,
    use_joint_limits: bool = True,
    constrain_left_foot_position: bool = False,
    constrain_right_foot_position: bool = False,
) -> pose_finder.Settings:
    urdf_path = resolve_robotics_uri_py.resolve_robotics_uri(
        "package://ergoCub/robots/ergoCubGazeboV1_minContacts/model.urdf"
    )
    output_settings = pose_finder.Settings()
    output_settings.terrain = input_terrain
    output_settings.robot_urdf = str(urdf_path)
    output_settings.joints_name_list = joint_names
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
    output_settings.maximum_joint_positions = math.pi * np.ones(number_of_joints)
    output_settings.maximum_joint_positions[
        output_settings.joints_name_list.index("l_knee")
    ] = -0.01
    output_settings.maximum_joint_positions[
        output_settings.joints_name_list.index("r_knee")
    ] = -0.01
    output_settings.minimum_joint_positions = -math.pi * np.ones(number_of_joints)
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
    output_settings.left_point_position_expression_type = (
        hippopt.ExpressionType.subject_to
        if constrain_left_foot_position
        else hippopt.ExpressionType.minimize
    )
    output_settings.right_point_position_expression_type = (
        hippopt.ExpressionType.subject_to
        if constrain_right_foot_position
        else hippopt.ExpressionType.minimize
    )
    output_settings.casadi_function_options = {}
    output_settings.casadi_opti_options = {}
    output_settings.casadi_solver_options = {
        "alpha_for_y": "dual-and-full",
        "tol": 1e-3,
        "dual_inf_tol": 1000.0,
        "compl_inf_tol": 1e-2,
        "constr_viol_tol": 1e-4,
        "acceptable_tol": 10,
        "acceptable_iter": 2,
        "acceptable_compl_inf_tol": 1000,
    }

    return output_settings


def get_references(
    desired_left_foot_pose: liecasadi.SE3,
    desired_right_foot_pose: liecasadi.SE3,
    desired_com_position: np.ndarray,
    planner_settings: pose_finder.Settings,
) -> pose_finder.References:
    output_references = pose_finder.References(
        contact_point_descriptors=planner_settings.contact_points,
        number_of_joints=len(planner_settings.joints_name_list),
    )
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


def print_ankle_bounds_multipliers(
    input_solution: hippopt.Output[pose_finder.Variables],
    tag: str,
    joint_name_list: list,
):
    print(
        f"Left ankle roll constraint multiplier {tag}: ",
        input_solution.constraint_multipliers["joint_position_bounds"][
            joint_name_list.index("l_ankle_roll")
        ],
    )
    print(
        f"Right ankle roll constraint multiplier {tag}: ",
        input_solution.constraint_multipliers["joint_position_bounds"][
            joint_name_list.index("r_ankle_roll")
        ],
    )
    print(
        f"Left ankle pitch constraint multiplier {tag}: ",
        input_solution.constraint_multipliers["joint_position_bounds"][
            joint_name_list.index("l_ankle_pitch")
        ],
    )
    print(
        f"Right ankle pitch constraint multiplier {tag}: ",
        input_solution.constraint_multipliers["joint_position_bounds"][
            joint_name_list.index("r_ankle_pitch")
        ],
    )


def complex_pose(
    terrain_height,
    terrain_origin,
    use_joint_limits,
    desired_left_foot_position,
    desired_right_foot_position,
    desired_com_position,
    casadi_solver_options=None,
    com_regularization_cost_multiplier=None,
    force_regularization_cost_multiplier=None,
    constrain_left_foot_position=False,
    constrain_right_foot_position=False,
) -> hippopt.Output[pose_finder.Variables]:
    terrain = hp_rp.SmoothTerrain.step(
        length=0.5,
        width=0.8,
        height=terrain_height,
        position=terrain_origin,
    )
    terrain.set_name(f"high_step_{terrain_height}")
    planner_settings = get_planner_settings(
        input_terrain=terrain,
        use_joint_limits=use_joint_limits,
        constrain_left_foot_position=constrain_left_foot_position,
        constrain_right_foot_position=constrain_right_foot_position,
    )
    if casadi_solver_options is not None:
        planner_settings.casadi_solver_options.update(casadi_solver_options)
    if com_regularization_cost_multiplier is not None:
        planner_settings.com_regularization_cost_multiplier = (
            com_regularization_cost_multiplier
        )
    if force_regularization_cost_multiplier is not None:
        planner_settings.force_regularization_cost_multiplier = (
            force_regularization_cost_multiplier
        )
    planner = pose_finder.Planner(settings=planner_settings)
    references = get_references(
        desired_left_foot_pose=liecasadi.SE3.from_translation_and_rotation(
            desired_left_foot_position, liecasadi.SO3.Identity()
        ),
        desired_right_foot_pose=liecasadi.SE3.from_translation_and_rotation(
            desired_right_foot_position, liecasadi.SO3.Identity()
        ),
        desired_com_position=desired_com_position,
        planner_settings=planner_settings,
    )
    planner.set_references(references)
    solution = planner.solve()
    visualizer_settings = get_visualizer_settings(
        input_planner_settings=planner_settings, robot_model=planner.get_adam_model()
    )
    visualizer = hp_rp.HumanoidStateVisualizer(settings=visualizer_settings)
    visualizer.visualize(solution.values.state)  # noqa
    return solution


if __name__ == "__main__":

    # Large step-up 20cm centered

    step_length = 0.45
    step_height = 0.2
    output = complex_pose(
        terrain_height=step_height,
        terrain_origin=np.array([0.45, 0.0, 0.0]),
        use_joint_limits=True,
        desired_left_foot_position=np.array([0.0, 0.1, 0.0]),
        desired_right_foot_position=np.array([step_length, -0.1, step_height]),
        desired_com_position=np.array([step_length / 2, 0.0, 0.7]),
    )
    complex_poses = {"high_step_20cm": output.values.state.to_dict(flatten=False)}
    print_ankle_bounds_multipliers(
        input_solution=output, tag="up20", joint_name_list=joint_names
    )

    print("Press [Enter] to move to next pose.")
    input()

    # Large step-up 20cm left foot centered

    step_length = 0.45
    step_height = 0.2
    output = complex_pose(
        terrain_height=step_height,
        terrain_origin=np.array([0.45, 0.0, 0.0]),
        use_joint_limits=True,
        desired_left_foot_position=np.array([0.0, 0.1, 0.0]),
        desired_right_foot_position=np.array([step_length, -0.1, step_height]),
        desired_com_position=np.array([0.0, 0.1, 0.7]),
        casadi_solver_options={"alpha_for_y": "primal"},
        com_regularization_cost_multiplier=200.0,
        force_regularization_cost_multiplier=0.0,
    )
    complex_poses["high_step_20cm_left"]: output.values.state.to_dict(flatten=False)
    print_ankle_bounds_multipliers(
        input_solution=output, tag="up20left", joint_name_list=joint_names
    )

    print("Press [Enter] to move to next pose.")
    input()

    # Large step-up 20cm right foot centered

    step_length = 0.45
    step_height = 0.2
    output = complex_pose(
        terrain_height=step_height,
        terrain_origin=np.array([0.45, 0.0, 0.0]),
        use_joint_limits=True,
        desired_left_foot_position=np.array([0.0, 0.1, 0.0]),
        desired_right_foot_position=np.array([step_length, -0.1, step_height]),
        desired_com_position=np.array([step_length, -0.1, 0.7]),
        com_regularization_cost_multiplier=200.0,
        force_regularization_cost_multiplier=0.0,
        constrain_left_foot_position=True,
        constrain_right_foot_position=True,
    )
    complex_poses["high_step_20cm_right"]: output.values.state.to_dict(flatten=False)
    print_ankle_bounds_multipliers(
        input_solution=output, tag="up20right", joint_name_list=joint_names
    )

    print("Press [Enter] to move to next pose.")
    input()

    # Large step-up 40cm
    step_length = 0.45
    step_height = 0.4
    output = complex_pose(
        terrain_height=step_height,
        terrain_origin=np.array([0.45, 0.0, 0.0]),
        use_joint_limits=True,
        desired_left_foot_position=np.array([0.0, 0.1, 0.0]),
        desired_right_foot_position=np.array([step_length, -0.1, step_height]),
        desired_com_position=np.array([step_length / 2, 0.0, 0.7]),
    )
    complex_poses["high_step_40cm"] = output.values.state.to_dict(flatten=False)
    print_ankle_bounds_multipliers(
        input_solution=output, tag="up40", joint_name_list=joint_names
    )

    print("Press [Enter] to move to next pose.")
    input()

    # Large step-up 40cm left foot centered
    step_length = 0.45
    step_height = 0.4
    output = complex_pose(
        terrain_height=step_height,
        terrain_origin=np.array([0.45, 0.0, 0.0]),
        use_joint_limits=True,
        desired_left_foot_position=np.array([0.0, 0.1, 0.0]),
        desired_right_foot_position=np.array([step_length, -0.1, step_height]),
        desired_com_position=np.array([0.0, 0.1, 0.7]),
        casadi_solver_options={"alpha_for_y": "primal"},
        com_regularization_cost_multiplier=200.0,
        force_regularization_cost_multiplier=0.0001,
    )
    complex_poses["high_step_40cm_left"] = output.values.state.to_dict(flatten=False)
    print_ankle_bounds_multipliers(
        input_solution=output, tag="up40left", joint_name_list=joint_names
    )

    print("Press [Enter] to move to next pose.")
    input()

    # Large step-down 10cm with limits
    step_length = 0.45
    step_height = 0.1
    output = complex_pose(
        terrain_height=step_height,
        terrain_origin=np.array([0.0, 0.0, 0.0]),
        use_joint_limits=True,
        desired_left_foot_position=np.array([0.0, 0.1, step_height]),
        desired_right_foot_position=np.array([step_length, -0.1, 0.0]),
        desired_com_position=np.array([step_length / 2, 0.0, 0.7]),
    )
    complex_poses["high_step_down_10cm"] = output.values.state.to_dict(flatten=False)
    print_ankle_bounds_multipliers(
        input_solution=output, tag="down10", joint_name_list=joint_names
    )

    print("Press [Enter] to move to next pose.")
    input()

    # Large step-down 20cm without limits

    step_length = 0.45
    step_height = 0.2
    output = complex_pose(
        terrain_height=step_height,
        terrain_origin=np.array([0.0, 0.0, 0.0]),
        use_joint_limits=False,
        desired_left_foot_position=np.array([0.0, 0.1, step_height]),
        desired_right_foot_position=np.array([step_length, -0.1, 0.0]),
        desired_com_position=np.array([step_length / 2, 0.0, 0.7]),
        constrain_left_foot_position=True,
        constrain_right_foot_position=True,
    )
    complex_poses["high_step_down_20cm_no_limit"] = output.values.state.to_dict(
        flatten=False
    )

    hdf5storage.savemat(
        file_name="complex_poses.mat",
        mdict=complex_poses,
        truncate_existing=True,
    )

    print("Press [Enter] to exit.")
    input()
