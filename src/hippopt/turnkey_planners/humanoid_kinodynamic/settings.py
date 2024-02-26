import dataclasses
import logging
import typing

import numpy as np

import hippopt as hp
from hippopt import integrators as hp_int
from hippopt import robot_planning as hp_rp


@dataclasses.dataclass
class Settings:
    robot_urdf: str = dataclasses.field(default=None)
    joints_name_list: list[str] = dataclasses.field(default=None)
    contact_points: hp_rp.FeetContactPointDescriptors = dataclasses.field(default=None)
    root_link: str = dataclasses.field(default=None)
    gravity: np.array = dataclasses.field(default=None)
    horizon_length: int = dataclasses.field(default=None)
    time_step: float = dataclasses.field(default=None)
    integrator: typing.Type[hp.SingleStepIntegrator] = dataclasses.field(default=None)
    terrain: hp_rp.TerrainDescriptor = dataclasses.field(default=None)
    planar_dcc_height_multiplier: float = dataclasses.field(default=None)
    dcc_gain: float = dataclasses.field(default=None)
    dcc_epsilon: float = dataclasses.field(default=None)
    static_friction: float = dataclasses.field(default=None)
    maximum_velocity_control: np.ndarray = dataclasses.field(default=None)
    maximum_force_derivative: np.ndarray = dataclasses.field(default=None)
    maximum_angular_momentum: float = dataclasses.field(default=None)
    minimum_com_height: float = dataclasses.field(default=None)
    minimum_feet_lateral_distance: float = dataclasses.field(default=None)
    maximum_feet_relative_height: float = dataclasses.field(default=None)
    maximum_joint_positions: np.ndarray = dataclasses.field(default=None)
    minimum_joint_positions: np.ndarray = dataclasses.field(default=None)
    maximum_joint_velocities: np.ndarray = dataclasses.field(default=None)
    minimum_joint_velocities: np.ndarray = dataclasses.field(default=None)

    final_state_expression_type: hp.ExpressionType = dataclasses.field(default=None)
    final_state_expression_weight: float = dataclasses.field(default=None)

    periodicity_expression_type: hp.ExpressionType = dataclasses.field(default=None)
    periodicity_expression_weight: float = dataclasses.field(default=None)

    contacts_centroid_cost_multiplier: float = dataclasses.field(default=None)

    com_linear_velocity_cost_weights: np.ndarray = dataclasses.field(default=None)
    com_linear_velocity_cost_multiplier: float = dataclasses.field(default=None)

    desired_frame_quaternion_cost_frame_name: str = dataclasses.field(default=None)

    desired_frame_quaternion_cost_multiplier: float = dataclasses.field(default=None)

    base_quaternion_cost_multiplier: float = dataclasses.field(default=None)

    base_quaternion_velocity_cost_multiplier: float = dataclasses.field(default=None)

    joint_regularization_cost_weights: np.ndarray = dataclasses.field(default=None)
    joint_regularization_cost_multiplier: float = dataclasses.field(default=None)

    force_regularization_cost_multiplier: float = dataclasses.field(default=None)

    foot_yaw_regularization_cost_multiplier: float = dataclasses.field(default=None)

    swing_foot_height_cost_multiplier: float = dataclasses.field(default=None)

    contact_velocity_control_cost_multiplier: float = dataclasses.field(default=None)

    contact_force_control_cost_multiplier: float = dataclasses.field(default=None)

    opti_solver: str = dataclasses.field(default="ipopt")

    problem_type: str = dataclasses.field(default="nlp")

    use_opti_callback: bool = dataclasses.field(default=None)

    acceptable_constraint_violation: float = dataclasses.field(default=None)

    opti_callback_save_costs: bool = dataclasses.field(default=None)

    opti_callback_save_constraint_multipliers: bool = dataclasses.field(default=None)

    casadi_function_options: dict = dataclasses.field(default_factory=dict)

    casadi_opti_options: dict = dataclasses.field(default_factory=dict)

    casadi_solver_options: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.root_link is None:
            self.root_link = "root_link"

        if self.gravity is None:
            self.gravity = np.array([0.0, 0.0, -9.80665, 0.0, 0.0, 0.0])

        if self.integrator is None:
            self.integrator = hp_int.ImplicitTrapezoid

        if self.terrain is None:
            self.terrain = hp_rp.PlanarTerrain()

        if self.planar_dcc_height_multiplier is None:
            self.planar_dcc_height_multiplier = 10.0

        if self.dcc_gain is None:
            self.dcc_gain = 20.0

        if self.dcc_epsilon is None:
            self.dcc_epsilon = 0.05

        if self.static_friction is None:
            self.static_friction = 0.3

        if self.maximum_velocity_control is None:
            self.maximum_velocity_control = np.array([2.0, 2.0, 5.0])

        if self.maximum_force_derivative is None:
            self.maximum_force_derivative = np.array([100.0, 100.0, 100.0])

        if self.maximum_angular_momentum is None:
            self.maximum_angular_momentum = 10.0

        if self.final_state_expression_type is None:
            self.final_state_expression_type = hp.ExpressionType.skip

        if self.final_state_expression_weight is None:
            self.final_state_expression_weight = 1.0

        if self.periodicity_expression_type is None:
            self.periodicity_expression_type = hp.ExpressionType.skip

        if self.periodicity_expression_weight is None:
            self.periodicity_expression_weight = 1.0

        if self.use_opti_callback is None:
            self.use_opti_callback = False

        if self.acceptable_constraint_violation is None:
            self.acceptable_constraint_violation = 1e-3

        if self.opti_callback_save_costs is None:
            self.opti_callback_save_costs = True

        if self.opti_callback_save_constraint_multipliers is None:
            self.opti_callback_save_constraint_multipliers = True

    def is_valid(self) -> bool:
        ok = True
        logger = logging.getLogger("[hippopt::HumanoidKynodynamic::Settings]")
        number_of_joints = len(self.joints_name_list)
        if self.robot_urdf is None:
            logger.error("robot_urdf is None")
            ok = False
        if self.joints_name_list is None:
            logger.error("joints_name_list is None")
            ok = False
        if self.contact_points is None:
            logger.error("contact_points is None")
            ok = False
        if self.horizon_length is None:
            logger.error("horizon_length is None")
            ok = False
        if self.time_step is None:
            logger.error("time_step is None")
            ok = False
        if self.minimum_com_height is None:
            logger.error("minimum_com_height is None")
            ok = False
        if self.minimum_feet_lateral_distance is None:
            logger.error("minimum_feet_lateral_distance is None")
            ok = False
        if self.maximum_feet_relative_height is None:
            logger.error("maximum_feet_relative_height is None")
            ok = False
        if self.maximum_joint_positions is None:
            logger.error("maximum_joint_positions is None")
            ok = False
        if self.minimum_joint_positions is None:
            logger.error("minimum_joint_positions is None")
            ok = False
        if self.maximum_joint_velocities is None:
            logger.error("maximum_joint_velocities is None")
            ok = False
        if self.minimum_joint_velocities is None:
            logger.error("minimum_joint_velocities is None")
            ok = False
        if len(self.maximum_joint_positions) != number_of_joints:
            logger.error(
                f"len(maximum_joint_positions)={len(self.maximum_joint_positions)} !="
                f" number_of_joints={number_of_joints}"
            )
            ok = False
        if len(self.minimum_joint_positions) != number_of_joints:
            logger.error(
                f"len(minimum_joint_positions)={len(self.minimum_joint_positions)} !="
                f" number_of_joints={number_of_joints}"
            )
            ok = False
        if len(self.maximum_joint_velocities) != number_of_joints:
            logger.error(
                f"len(maximum_joint_velocities)={len(self.maximum_joint_velocities)} !="
                f" number_of_joints={number_of_joints}"
            )
            ok = False
        if len(self.minimum_joint_velocities) != number_of_joints:
            logger.error(
                f"len(minimum_joint_velocities)={len(self.minimum_joint_velocities)} !="
                f" number_of_joints={number_of_joints}"
            )
            ok = False
        if self.contacts_centroid_cost_multiplier is None:
            logger.error("contacts_centroid_cost_multiplier is None")
            ok = False
        if self.com_linear_velocity_cost_weights is None:
            logger.error("com_linear_velocity_cost_weights is None")
            ok = False
        if len(self.com_linear_velocity_cost_weights) != 3:
            logger.error(
                f"len(com_linear_velocity_cost_weights)="
                f"{len(self.com_linear_velocity_cost_weights)} != 3"
            )
            ok = False
        if self.com_linear_velocity_cost_multiplier is None:
            logger.error("com_linear_velocity_cost_multiplier is None")
            ok = False
        if self.desired_frame_quaternion_cost_frame_name is None:
            logger.error("desired_frame_quaternion_cost_frame_name is None")
            ok = False
        if self.desired_frame_quaternion_cost_multiplier is None:
            logger.error("desired_frame_quaternion_cost_multiplier is None")
            ok = False
        if self.base_quaternion_cost_multiplier is None:
            logger.error("base_quaternion_cost_multiplier is None")
            ok = False
        if self.base_quaternion_velocity_cost_multiplier is None:
            logger.error("base_quaternion_velocity_cost_multiplier is None")
            ok = False
        if self.joint_regularization_cost_weights is None:
            logger.error("joint_regularization_cost_weights is None")
            ok = False
        if len(self.joint_regularization_cost_weights) != number_of_joints:
            logger.error(
                f"len(joint_regularization_cost_weights)="
                f"{len(self.joint_regularization_cost_weights)} !="
                f" number_of_joints={number_of_joints}"
            )
            ok = False
        if self.joint_regularization_cost_multiplier is None:
            logger.error("joint_regularization_cost_multiplier is None")
            ok = False
        if self.force_regularization_cost_multiplier is None:
            logger.error("force_regularization_cost_multiplier is None")
            ok = False
        if self.foot_yaw_regularization_cost_multiplier is None:
            logger.error("foot_yaw_regularization_cost_multiplier is None")
            ok = False
        if self.swing_foot_height_cost_multiplier is None:
            logger.error("swing_foot_height_cost_multiplier is None")
            ok = False
        if self.contact_velocity_control_cost_multiplier is None:
            logger.error("contact_velocity_control_cost_multiplier is None")
            ok = False
        if self.contact_force_control_cost_multiplier is None:
            logger.error("contact_force_control_cost_multiplier is None")
            ok = False
        return ok
