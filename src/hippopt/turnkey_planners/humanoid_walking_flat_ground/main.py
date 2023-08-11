import argparse

import hippopt.turnkey_planners.humanoid_walking_flat_ground.planner as walking_planner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trajectory Optimization of a forward walking motion on ergoCub.",
    )

    parser.add_argument(
        "--urdf",
        type=str,
        required=True,
        help="Path to the ergoCubGazeboV1_minContacts URDF file.",
    )

    planner_settings = walking_planner.Settings()
    planner_settings.robot_urdf = parser.parse_args().urdf
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
    planner_settings.root_link = "root_link"
    planner_settings.horizon_length = 20
    planner_settings.time_step = 0.1
    # planner_settings.contact_points: FeetContactPointDescriptors = dataclasses.field(default=None)

    # planner_settings.planar_dcc_height_multiplier: float = dataclasses.field(default=None)
    # planner_settings.dcc_gain: float = dataclasses.field(default=None)
    # planner_settings.dcc_epsilon: float = dataclasses.field(default=None)
    # planner_settings.static_friction: float = dataclasses.field(default=None)
    # planner_settings.maximum_velocity_control: np.ndarray = dataclasses.field(default=None)
    # planner_settings.maximum_force_derivative: np.ndarray = dataclasses.field(default=None)
    # planner_settings.maximum_angular_momentum: float = dataclasses.field(default=None)
    # planner_settings.minimum_com_height: float = dataclasses.field(default=None)
    # planner_settings.minimum_feet_lateral_distance: float = dataclasses.field(default=None)
    # planner_settings.maximum_feet_relative_height: float = dataclasses.field(default=None)
    # planner_settings.maximum_joint_positions: np.ndarray = dataclasses.field(default=None)
    # planner_settings.minimum_joint_positions: np.ndarray = dataclasses.field(default=None)
    # planner_settings.maximum_joint_velocities: np.ndarray = dataclasses.field(default=None)
    # planner_settings.minimum_joint_velocities: np.ndarray = dataclasses.field(default=None)
    # planner_settings.contacts_centroid_cost_multiplier: float = dataclasses.field(default=None)
    # planner_settings.com_linear_velocity_cost_weights: np.ndarray = dataclasses.field(default=None)
    # planner_settings.com_linear_velocity_cost_multiplier: float = dataclasses.field(default=None)
    # planner_settings.desired_frame_quaternion_cost_frame_name: str = dataclasses.field(default=None)
    # planner_settings.desired_frame_quaternion_cost_multiplier: float = dataclasses.field(default=None)
    # planner_settings.base_quaternion_cost_multiplier: float = dataclasses.field(default=None)
    # planner_settings.base_quaternion_velocity_cost_multiplier: float = dataclasses.field(default=None)
    # planner_settings.joint_regularization_cost_weights: np.ndarray = dataclasses.field(default=None)
    # planner_settings.joint_regularization_cost_multiplier: float = dataclasses.field(default=None)
    # planner_settings.force_regularization_cost_multiplier: float = dataclasses.field(default=None)
    # planner_settings.foot_yaw_regularization_cost_multiplier: float = dataclasses.field(default=None)
    # planner_settings.swing_foot_height_cost_multiplier: float = dataclasses.field(default=None)
    # planner_settings.contact_velocity_control_cost_multiplier: float = dataclasses.field(default=None)
    # planner_settings.contact_force_control_cost_multiplier: float = dataclasses.field(default=None)
    # planner_settings.casadi_function_options: dict = dataclasses.field(default_factory=dict)
    # planner_settings.casadi_opti_options: dict = dataclasses.field(default_factory=dict)
    # planner_settings.casadi_solver_options: dict = dataclasses.field(default_factory=dict)
