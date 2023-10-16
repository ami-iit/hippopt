import argparse

import casadi as cs
import idyntree.bindings as idyntree
import numpy as np

import hippopt.robot_planning as hp_rp
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
    number_of_joints = len(planner_settings.joints_name_list)

    idyntree_model_loader = idyntree.ModelLoader()
    idyntree_model_loader.loadReducedModelFromFile(
        planner_settings.robot_urdf, planner_settings.joints_name_list
    )
    idyntree_model = idyntree_model_loader.model()

    planner_settings.root_link = "root_link"
    planner_settings.horizon_length = 20
    planner_settings.time_step = 0.1

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

    planner_settings.planar_dcc_height_multiplier = 10.0
    planner_settings.dcc_gain = 20.0
    planner_settings.dcc_epsilon = 0.05
    planner_settings.static_friction = 0.3
    planner_settings.maximum_velocity_control = [2.0, 2.0, 5.0]
    planner_settings.maximum_force_derivative = [100.0, 100.0, 100.0]
    planner_settings.maximum_angular_momentum = 10.0
    planner_settings.minimum_com_height = 0.3
    planner_settings.minimum_feet_lateral_distance = 0.1
    planner_settings.maximum_feet_relative_height = 0.05

    planner_settings.maximum_joint_positions = cs.inf * np.ones(number_of_joints)
    planner_settings.minimum_joint_positions = -cs.inf * np.ones(number_of_joints)

    for i in range(number_of_joints):
        joint = idyntree_model.getJoint(i)
        if joint.hasPosLimits():
            planner_settings.maximum_joint_positions[i] = joint.getMaxPosLimit(i)
            planner_settings.minimum_joint_positions[i] = joint.getMinPosLimit(i)

    planner_settings.maximum_joint_velocities = np.ones(number_of_joints) * 2.0
    planner_settings.minimum_joint_velocities = np.ones(number_of_joints) * -2.0

    planner_settings.joint_regularization_cost_weights = np.ones(number_of_joints)
    planner_settings.joint_regularization_cost_weights[:3] = 0.1  # torso
    planner_settings.joint_regularization_cost_weights[3:11] = 10.0  # arms
    planner_settings.joint_regularization_cost_weights[11:] = 1.0  # legs

    planner_settings.contacts_centroid_cost_multiplier = 100.0
    planner_settings.com_linear_velocity_cost_weights = [10.0, 0.1, 1.0]
    planner_settings.com_linear_velocity_cost_multiplier = 1.0
    planner_settings.desired_frame_quaternion_cost_frame_name = "chest"
    planner_settings.desired_frame_quaternion_cost_multiplier = 90.0
    planner_settings.base_quaternion_cost_multiplier = 50.0
    planner_settings.base_quaternion_velocity_cost_multiplier = 0.001
    planner_settings.joint_regularization_cost_multiplier = 0.1
    planner_settings.force_regularization_cost_multiplier = 0.2
    planner_settings.foot_yaw_regularization_cost_multiplier = 2000.0
    planner_settings.swing_foot_height_cost_multiplier = 1000.0
    planner_settings.contact_velocity_control_cost_multiplier = 5.0
    planner_settings.contact_force_control_cost_multiplier = 0.0001
    planner_settings.casadi_function_options = {}
    planner_settings.casadi_opti_options = {}
    planner_settings.casadi_solver_options = {}

    planner = walking_planner.Planner(settings=planner_settings)
