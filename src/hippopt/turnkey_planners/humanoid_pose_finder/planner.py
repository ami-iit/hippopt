import copy
import dataclasses
import logging
from typing import Any

import adam.casadi
import adam.model
import adam.numpy
import adam.parametric.casadi
import casadi as cs
import numpy as np
from adam.parametric.model.parametric_factories.parametric_model import (
    URDFParametricModelFactory,
)

import hippopt as hp
import hippopt.robot_planning as hp_rp


@dataclasses.dataclass
class Settings:
    robot_urdf: str = dataclasses.field(default=None)
    joints_name_list: list[str] = dataclasses.field(default=None)
    parametric_link_names: list[str] = dataclasses.field(default=None)
    contact_points: hp_rp.FeetContactPointDescriptors = dataclasses.field(default=None)
    root_link: str = dataclasses.field(default=None)

    gravity: np.array = dataclasses.field(default=None)
    terrain: hp_rp.TerrainDescriptor = dataclasses.field(default=None)
    relaxed_complementarity_epsilon: float = dataclasses.field(default=None)
    static_friction: float = dataclasses.field(default=None)
    maximum_joint_positions: np.ndarray = dataclasses.field(default=None)
    minimum_joint_positions: np.ndarray = dataclasses.field(default=None)

    base_quaternion_cost_multiplier: float = dataclasses.field(default=None)

    desired_frame_quaternion_cost_frame_name: str = dataclasses.field(default=None)

    desired_frame_quaternion_cost_multiplier: float = dataclasses.field(default=None)

    com_position_expression_type: hp.ExpressionType = dataclasses.field(default=None)
    com_regularization_cost_multiplier: float = dataclasses.field(default=None)

    joint_regularization_cost_weights: np.ndarray = dataclasses.field(default=None)
    joint_regularization_cost_multiplier: float = dataclasses.field(default=None)

    force_regularization_cost_multiplier: float = dataclasses.field(default=None)
    average_force_regularization_cost_multiplier: float = dataclasses.field(
        default=None
    )

    left_point_position_expression_type: hp.ExpressionType = dataclasses.field(
        default=None
    )
    right_point_position_expression_type: hp.ExpressionType = dataclasses.field(
        default=None
    )
    point_position_regularization_cost_multiplier: float = dataclasses.field(
        default=None
    )

    left_hand_frame_name: str = dataclasses.field(default=None)
    lef_hand_position_in_frame: np.array = dataclasses.field(default=None)
    left_hand_regularization_cost_multiplier: float = dataclasses.field(default=None)
    left_hand_expression_type: hp.ExpressionType = dataclasses.field(default=None)
    right_hand_frame_name: str = dataclasses.field(default=None)
    right_hand_position_in_frame: np.array = dataclasses.field(default=None)
    right_hand_regularization_cost_multiplier: float = dataclasses.field(default=None)
    right_hand_expression_type: hp.ExpressionType = dataclasses.field(default=None)

    opti_solver: str = dataclasses.field(default="ipopt")
    problem_type: str = dataclasses.field(default="nlp")
    casadi_function_options: dict = dataclasses.field(default_factory=dict)
    casadi_opti_options: dict = dataclasses.field(default_factory=dict)
    casadi_solver_options: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root_link = "root_link"
        self.gravity = np.array([0.0, 0.0, -9.80665, 0.0, 0.0, 0.0])
        self.terrain = hp_rp.PlanarTerrain()
        self.relaxed_complementarity_epsilon = 0.0001
        self.static_friction = 0.3
        self.com_position_expression_type = hp.ExpressionType.minimize
        self.left_point_position_expression_type = hp.ExpressionType.minimize
        self.right_point_position_expression_type = hp.ExpressionType.minimize
        self.lef_hand_position_in_frame = np.array([0.0, 0.0, 0.0])
        self.left_hand_regularization_cost_multiplier = 1.0
        self.left_hand_expression_type = hp.ExpressionType.skip
        self.right_hand_position_in_frame = np.array([0.0, 0.0, 0.0])
        self.right_hand_regularization_cost_multiplier = 1.0
        self.right_hand_expression_type = hp.ExpressionType.skip

    def is_valid(self) -> bool:
        ok = True
        logger = logging.getLogger("[hippopt::HumanoidPoseFInder::Settings]")
        if self.robot_urdf is None:
            logger.error("robot_urdf is None")
            ok = False
        if self.joints_name_list is None:
            logger.error("joints_name_list is None")
            ok = False
        if self.contact_points is None:
            logger.error("contact_points is None")
            ok = False
        if self.root_link is None:
            logger.error("root_link is None")
            ok = False
        if self.gravity is None or len(self.gravity) != 6:
            logger.error("gravity is None")
            ok = False
        if self.terrain is None:
            logger.error("terrain is None")
            ok = False
        if self.relaxed_complementarity_epsilon is None:
            logger.error("relaxed_complementarity_epsilon is None")
            ok = False
        if self.static_friction is None:
            logger.error("static_friction is None")
            ok = False
        if self.maximum_joint_positions is None:
            logger.error("maximum_joint_positions is None")
            ok = False
        if self.minimum_joint_positions is None:
            logger.error("minimum_joint_positions is None")
            ok = False
        if self.com_regularization_cost_multiplier is None:
            logger.error("com_regularization_cost_multiplier is None")
            ok = False
        if self.base_quaternion_cost_multiplier is None:
            logger.error("base_quaternion_cost_multiplier is None")
            ok = False
        if self.desired_frame_quaternion_cost_frame_name is None:
            logger.error("desired_frame_quaternion_cost_frame_name is None")
            ok = False
        if self.desired_frame_quaternion_cost_multiplier is None:
            logger.error("desired_frame_quaternion_cost_multiplier is None")
            ok = False
        if self.joint_regularization_cost_weights is None:
            logger.error("joint_regularization_cost_weights is None")
            ok = False
        if self.joint_regularization_cost_multiplier is None:
            logger.error("joint_regularization_cost_multiplier is None")
            ok = False
        if self.force_regularization_cost_multiplier is None:
            logger.error("force_regularization_cost_multiplier is None")
            ok = False
        if self.average_force_regularization_cost_multiplier is None:
            logger.error("average_force_regularization_cost_multiplier is None")
            ok = False
        if self.point_position_regularization_cost_multiplier is None:
            logger.error("point_position_regularization_cost_multiplier is None")
            ok = False
        if self.com_position_expression_type is None:
            logger.error("com_position_expression_type is None")
            ok = False
        if self.left_point_position_expression_type is None:
            logger.error("left_point_position_expression_type is None")
            ok = False
        if self.right_point_position_expression_type is None:
            logger.error("right_point_position_expression_type is None")
            ok = False
        if self.left_hand_expression_type is None:
            logger.error("left_hand_expression_type is None")
            ok = False
        if (
            self.left_hand_expression_type != hp.ExpressionType.skip
            and self.left_hand_frame_name is None
        ):
            logger.error("left_hand_frame_name is None")
            ok = False
        if self.lef_hand_position_in_frame is None:
            logger.error("lef_hand_position_in_frame is None")
            ok = False
        if self.left_hand_regularization_cost_multiplier is None:
            logger.error("left_hand_regularization_cost_multiplier is None")
            ok = False
        if self.right_hand_expression_type is None:
            logger.error("right_hand_expression_type is None")
            ok = False
        if (
            self.right_hand_expression_type != hp.ExpressionType.skip
            and self.right_hand_frame_name is None
        ):
            logger.error("right_hand_frame_name is None")
            ok = False
        if self.right_hand_position_in_frame is None:
            logger.error("right_hand_position_in_frame is None")
            ok = False
        if self.right_hand_regularization_cost_multiplier is None:
            logger.error("right_hand_regularization_cost_multiplier is None")
            ok = False

        return ok


@dataclasses.dataclass
class References(hp.OptimizationObject):
    state: hp_rp.HumanoidState = hp.default_composite_field(
        cls=hp.Parameter, factory=hp_rp.HumanoidState
    )
    frame_quaternion_xyzw: hp.StorageType = hp.default_storage_field(hp.Parameter)
    left_hand_position: hp.StorageType = hp.default_storage_field(hp.Parameter)
    right_hand_position: hp.StorageType = hp.default_storage_field(hp.Parameter)

    contact_point_descriptors: dataclasses.InitVar[
        hp_rp.FeetContactPointDescriptors
    ] = dataclasses.field(default=None)
    number_of_joints: dataclasses.InitVar[int] = dataclasses.field(default=None)

    def __post_init__(
        self,
        contact_point_descriptors: hp_rp.FeetContactPointDescriptors,
        number_of_joints: int,
    ):
        self.state = hp_rp.HumanoidState(
            contact_point_descriptors=contact_point_descriptors,
            number_of_joints=number_of_joints,
        )
        self.frame_quaternion_xyzw = np.zeros(4)
        self.frame_quaternion_xyzw[3] = 1.0
        self.left_hand_position = np.zeros(3)
        self.right_hand_position = np.zeros(3)


@dataclasses.dataclass
class Variables(hp.OptimizationObject):
    state: hp_rp.HumanoidState = hp.default_composite_field(
        cls=hp.Variable, factory=hp_rp.HumanoidState
    )
    mass: hp.StorageType = hp.default_storage_field(hp.Parameter)
    parametric_link_length_multipliers: hp.StorageType = hp.default_storage_field(
        hp.Parameter
    )
    parametric_link_densities: hp.StorageType = hp.default_storage_field(hp.Parameter)
    gravity: hp.StorageType = hp.default_storage_field(hp.Parameter)
    references: References = hp.default_composite_field(
        cls=hp.Parameter, factory=References
    )

    relaxed_complementarity_epsilon: hp.StorageType = hp.default_storage_field(
        hp.Parameter
    )
    static_friction: hp.StorageType = hp.default_storage_field(hp.Parameter)
    maximum_joint_positions: hp.StorageType = hp.default_storage_field(hp.Parameter)
    minimum_joint_positions: hp.StorageType = hp.default_storage_field(hp.Parameter)

    left_hand_position_in_frame: hp.StorageType = hp.default_storage_field(hp.Parameter)
    right_hand_position_in_frame: hp.StorageType = hp.default_storage_field(
        hp.Parameter
    )

    settings: dataclasses.InitVar[Settings] = dataclasses.field(default=None)
    kin_dyn_object: dataclasses.InitVar[adam.casadi.KinDynComputations] = (
        dataclasses.field(default=None)
    )

    def __post_init__(
        self,
        settings: Settings,
        kin_dyn_object: (
            adam.casadi.KinDynComputations
            | adam.parametric.casadi.KinDynComputationsParametric
        ),
    ) -> None:
        self.state = hp_rp.HumanoidState(
            contact_point_descriptors=settings.contact_points,
            number_of_joints=len(settings.joints_name_list),
        )
        self.references = References(
            contact_point_descriptors=settings.contact_points,
            number_of_joints=len(settings.joints_name_list),
        )
        self.parametric_link_length_multipliers = (
            np.ones(len(settings.parametric_link_names))
            if settings.parametric_link_names is not None
            else 0.0
        )

        if isinstance(
            kin_dyn_object, adam.parametric.casadi.KinDynComputationsParametric
        ):
            self.parametric_link_densities = kin_dyn_object.get_original_densities()
            total_mass_fun = kin_dyn_object.get_total_mass()
            self.mass = float(
                total_mass_fun(
                    self.parametric_link_length_multipliers,
                    self.parametric_link_densities,
                ).full()
            )
        else:
            self.mass = kin_dyn_object.get_total_mass()
            self.parametric_link_densities = 0.0
        self.gravity = settings.gravity
        self.static_friction = settings.static_friction
        self.relaxed_complementarity_epsilon = settings.relaxed_complementarity_epsilon
        self.maximum_joint_positions = settings.maximum_joint_positions
        self.minimum_joint_positions = settings.minimum_joint_positions
        self.left_hand_position_in_frame = settings.lef_hand_position_in_frame
        self.right_hand_position_in_frame = settings.right_hand_position_in_frame


class Planner:
    def __init__(self, settings: Settings):
        if not settings.is_valid():
            raise ValueError("Settings are not valid")

        self.settings = copy.deepcopy(settings)

        if self.settings.parametric_link_names is not None:
            self.parametric_model = True
            self.kin_dyn_object = adam.parametric.casadi.KinDynComputationsParametric(
                urdfstring=self.settings.robot_urdf,
                joints_name_list=self.settings.joints_name_list,
                links_name_list=self.settings.parametric_link_names,
                root_link=self.settings.root_link,
                gravity=self.settings.gravity,
                f_opts=self.settings.casadi_function_options,
            )
        else:
            self.parametric_model = False
            self.kin_dyn_object = adam.casadi.KinDynComputations(
                urdfstring=self.settings.robot_urdf,
                joints_name_list=self.settings.joints_name_list,
                root_link=self.settings.root_link,
                gravity=self.settings.gravity,
                f_opts=self.settings.casadi_function_options,
            )
            self.numeric_mass = self.kin_dyn_object.get_total_mass()

        self.variables = Variables(
            settings=self.settings, kin_dyn_object=self.kin_dyn_object
        )

        self.optimization_solver = hp.OptiSolver(
            inner_solver=self.settings.opti_solver,
            problem_type=self.settings.problem_type,
            options_solver=self.settings.casadi_solver_options,
            options_plugin=self.settings.casadi_opti_options,
        )

        self.op = hp.OptimizationProblem.create(
            input_structure=self.variables, optimization_solver=self.optimization_solver
        )

        sym_variables = self.op.variables  # type: Variables

        function_inputs = self._get_function_inputs_dict()

        # Normalized quaternion computation
        normalized_quaternion_fun = hp_rp.quaternion_xyzw_normalization(
            **function_inputs
        )
        normalized_quaternion = normalized_quaternion_fun(
            q=sym_variables.state.kinematics.base.quaternion_xyzw
        )["quaternion_normalized"]

        # Align names used in the terrain function with those in function_inputs
        self.settings.terrain.change_options(**function_inputs)

        # Definition of contact constraint functions
        relaxed_complementarity_fun = hp_rp.relaxed_complementarity_margin(
            terrain=self.settings.terrain,
            **function_inputs,
        )
        friction_margin_fun = hp_rp.friction_cone_square_margin(
            terrain=self.settings.terrain, **function_inputs
        )
        height_fun = self.settings.terrain.height_function()
        normal_force_fun = hp_rp.normal_force_component(
            terrain=self.settings.terrain, **function_inputs
        )

        point_kinematics_functions = {}
        all_contact_points = (
            sym_variables.state.contact_points.left
            + sym_variables.state.contact_points.right
        )

        for i, point in enumerate(all_contact_points):
            self._add_contact_point_feasibility(
                relaxed_complementarity_fun,
                friction_margin_fun,
                height_fun,
                normal_force_fun,
                point,
            )

            self._add_contact_kinematic_consistency(
                function_inputs,
                normalized_quaternion,
                point,
                point_kinematics_functions,
            )

        self._add_kinematics_constraints(
            function_inputs, normalized_quaternion, all_contact_points
        )
        self._add_kinematics_regularization(
            function_inputs=function_inputs, normalized_quaternion=normalized_quaternion
        )

        expression_type = self.settings.left_point_position_expression_type
        self._add_foot_regularization(
            points=sym_variables.state.contact_points.left,
            references=sym_variables.references.state.contact_points.left,
            point_position_expression_type=expression_type,
        )
        expression_type = self.settings.right_point_position_expression_type
        self._add_foot_regularization(
            points=sym_variables.state.contact_points.right,
            references=sym_variables.references.state.contact_points.right,
            point_position_expression_type=expression_type,
        )

    def _get_function_inputs_dict(self):
        variables = self.op.variables
        function_inputs = {
            "mass_name": variables.mass.name(),
            "com_name": variables.state.com.name(),
            "quaternion_xyzw_name": "q",
            "gravity_name": variables.gravity.name(),
            "point_position_names": [],
            "point_force_names": [],
            "point_position_in_frame_name": "p_parent",
            "base_position_name": "pb",
            "base_quaternion_xyzw_name": "qb",
            "joint_positions_name": "s",
            "point_position_name": "p",
            "point_force_name": "f",
            "height_multiplier_name": "kt",
            "relaxed_complementarity_epsilon_name": "eps",
            "static_friction_name": "mu_s",
            "desired_quaternion_xyzw_name": "qd",
            "first_point_name": "p_0",
            "second_point_name": "p_1",
            "desired_yaw_name": "yd",
            "desired_height_name": "hd",
            "parametric_link_length_multipliers_name": "pi_l",
            "parametric_link_densities_name": "pi_d",
            "options": self.settings.casadi_function_options,
        }
        return function_inputs

    def _add_kinematics_constraints(
        self,
        function_inputs: dict,
        normalized_quaternion: cs.MX,
        all_contact_points: list[hp_rp.ContactPointState],
    ):
        problem = self.op.problem
        variables = self.op.variables  # type: Variables

        # Unitary quaternion
        problem.add_constraint(
            expression=cs.MX(
                cs.sumsqr(variables.state.kinematics.base.quaternion_xyzw) == 1
            ),
            name="unitary_quaternion",
        )

        # Consistency of com position with kinematics
        com_kinematics_fun = hp_rp.center_of_mass_position_from_kinematics(
            kindyn_object=self.kin_dyn_object, **function_inputs
        )
        if self.parametric_model:
            com_kinematics = com_kinematics_fun(
                pb=variables.state.kinematics.base.position,
                qb=normalized_quaternion,
                s=variables.state.kinematics.joints.positions,
                pi_l=variables.parametric_link_length_multipliers,
                pi_d=variables.parametric_link_densities,
            )["com_position"]
        else:
            com_kinematics = com_kinematics_fun(
                pb=variables.state.kinematics.base.position,
                qb=normalized_quaternion,
                s=variables.state.kinematics.joints.positions,
            )["com_position"]
        problem.add_constraint(
            expression=cs.MX(variables.state.com == com_kinematics),
            name="com_kinematics_consistency",
        )

        # Zero momentum derivative
        # 0 = sum_i (p_i x f_i) + mg (centroidal momentum dynamics)
        function_inputs["point_position_names"] = [
            point.p.name() for point in all_contact_points
        ]
        function_inputs["point_force_names"] = [
            point.f.name() for point in all_contact_points
        ]
        centroidal_dynamics_fun = hp_rp.centroidal_dynamics_with_point_forces(
            number_of_points=len(function_inputs["point_position_names"]),
            assume_unitary_mass=True,
            **function_inputs,
        )

        centroidal_inputs = {
            variables.gravity.name(): variables.gravity,  # noqa
            variables.state.com.name(): variables.state.com,
        }
        for point in all_contact_points:
            centroidal_inputs[point.p.name()] = point.p
            centroidal_inputs[point.f.name()] = point.f

        centroidal_dynamics = centroidal_dynamics_fun(**centroidal_inputs)["h_g_dot"]

        problem.add_constraint(
            expression=centroidal_dynamics == np.zeros(6),  # noqa
            name="centroidal_momentum_dynamics",
        )

        # Joint position bounds
        problem.add_constraint(
            expression=cs.Opti_bounded(
                variables.minimum_joint_positions,
                variables.state.kinematics.joints.positions,
                variables.maximum_joint_positions,
            ),
            name="joint_position_bounds",
        )

    def _add_kinematics_regularization(
        self, function_inputs: dict, normalized_quaternion: cs.MX
    ):
        problem = self.op.problem
        variables = self.op.variables  # type: Variables

        # Desired base orientation
        quaternion_error_fun = hp_rp.quaternion_xyzw_error(**function_inputs)
        quaternion_error = quaternion_error_fun(
            q=variables.state.kinematics.base.quaternion_xyzw,
            qd=variables.references.state.kinematics.base.quaternion_xyzw,
        )["quaternion_error"]
        problem.add_cost(
            expression=cs.sumsqr(quaternion_error),
            name="base_quaternion_error",
            scaling=self.settings.base_quaternion_cost_multiplier,
        )

        # Desired frame orientation
        rotation_error_kinematics_fun = hp_rp.rotation_error_from_kinematics(
            kindyn_object=self.kin_dyn_object,
            target_frame=self.settings.desired_frame_quaternion_cost_frame_name,
            **function_inputs,
        )
        if self.parametric_model:
            rotation_error_kinematics = rotation_error_kinematics_fun(
                pb=variables.state.kinematics.base.position,
                qb=normalized_quaternion,
                s=variables.state.kinematics.joints.positions,
                pi_l=variables.parametric_link_length_multipliers,
                pi_d=variables.parametric_link_densities,
                qd=variables.references.frame_quaternion_xyzw,
            )["rotation_error"]
        else:
            rotation_error_kinematics = rotation_error_kinematics_fun(
                pb=variables.state.kinematics.base.position,
                qb=normalized_quaternion,
                s=variables.state.kinematics.joints.positions,
                qd=variables.references.frame_quaternion_xyzw,
            )["rotation_error"]
        problem.add_cost(
            expression=cs.sumsqr(cs.trace(rotation_error_kinematics) - 3),
            name="frame_rotation_error",
            scaling=self.settings.desired_frame_quaternion_cost_multiplier,
        )

        # Desired center of mass position
        com_position_error = variables.state.com - variables.references.state.com
        problem.add_expression(
            expression=cs.sumsqr(com_position_error),
            name="com_position_error",
            scaling=self.settings.com_regularization_cost_multiplier,
            mode=self.settings.com_position_expression_type,
        )

        # Desired joint positions
        joint_positions_error = (
            variables.state.kinematics.joints.positions
            - variables.references.state.kinematics.joints.positions
        )

        weighted_error = (
            joint_positions_error.T
            @ cs.diag(self.settings.joint_regularization_cost_weights)
            @ joint_positions_error
        )

        problem.add_cost(
            expression=weighted_error,
            name="joint_positions_error",
            scaling=self.settings.joint_regularization_cost_multiplier,
        )

        if self.settings.left_hand_expression_type != hp.ExpressionType.skip:
            left_hand_position_fun = hp_rp.point_position_from_kinematics(
                kindyn_object=self.kin_dyn_object,
                frame_name=self.settings.left_hand_frame_name,
                **function_inputs,
            )
            if self.parametric_model:
                left_hand_position = left_hand_position_fun(
                    pb=variables.state.kinematics.base.position,
                    qb=normalized_quaternion,
                    s=variables.state.kinematics.joints.positions,
                    pi_l=variables.parametric_link_length_multipliers,
                    pi_d=variables.parametric_link_densities,
                    p_parent=variables.left_hand_position_in_frame,
                )["point_position"]
            else:
                left_hand_position = left_hand_position_fun(
                    pb=variables.state.kinematics.base.position,
                    qb=normalized_quaternion,
                    s=variables.state.kinematics.joints.positions,
                    p_parent=variables.left_hand_position_in_frame,
                )["point_position"]

            problem.add_expression(
                expression=cs.MX(
                    left_hand_position == variables.references.left_hand_position
                ),
                name="left_hand_position_error",
                scaling=self.settings.left_hand_regularization_cost_multiplier,
                mode=self.settings.left_hand_expression_type,
            )

        if self.settings.right_hand_expression_type != hp.ExpressionType.skip:
            right_hand_position_fun = hp_rp.point_position_from_kinematics(
                kindyn_object=self.kin_dyn_object,
                frame_name=self.settings.right_hand_frame_name,
                **function_inputs,
            )

            if self.parametric_model:
                right_hand_position = right_hand_position_fun(
                    pb=variables.state.kinematics.base.position,
                    qb=normalized_quaternion,
                    s=variables.state.kinematics.joints.positions,
                    pi_l=variables.parametric_link_length_multipliers,
                    pi_d=variables.parametric_link_densities,
                    p_parent=variables.right_hand_position_in_frame,
                )["point_position"]
            else:
                right_hand_position = right_hand_position_fun(
                    pb=variables.state.kinematics.base.position,
                    qb=normalized_quaternion,
                    s=variables.state.kinematics.joints.positions,
                    p_parent=variables.right_hand_position_in_frame,
                )["point_position"]

            problem.add_expression(
                expression=cs.MX(
                    right_hand_position == variables.references.right_hand_position
                ),
                name="right_hand_position_error",
                scaling=self.settings.right_hand_regularization_cost_multiplier,
                mode=self.settings.right_hand_expression_type,
            )

    def _add_contact_kinematic_consistency(
        self,
        function_inputs: dict,
        normalized_quaternion: cs.MX,
        point: hp_rp.ContactPointState,
        point_kinematics_functions: dict,
    ):
        problem = self.op.problem
        variables = self.op.variables  # type: Variables

        # Creation of contact kinematics consistency functions
        descriptor = point.descriptor
        if descriptor.foot_frame not in point_kinematics_functions:
            point_kinematics_functions[descriptor.foot_frame] = (
                hp_rp.point_position_from_kinematics(
                    kindyn_object=self.kin_dyn_object,
                    frame_name=descriptor.foot_frame,
                    **function_inputs,
                )
            )

        # Consistency between the contact position and the kinematics
        if self.parametric_model:
            point_kinematics = point_kinematics_functions[descriptor.foot_frame](
                pb=variables.state.kinematics.base.position,
                qb=normalized_quaternion,
                s=variables.state.kinematics.joints.positions,
                pi_l=variables.parametric_link_length_multipliers,
                pi_d=variables.parametric_link_densities,
                p_parent=descriptor.position_in_foot_frame,
            )["point_position"]
        else:
            point_kinematics = point_kinematics_functions[descriptor.foot_frame](
                pb=variables.state.kinematics.base.position,
                qb=normalized_quaternion,
                s=variables.state.kinematics.joints.positions,
                p_parent=descriptor.position_in_foot_frame,
            )["point_position"]
        problem.add_constraint(
            expression=cs.MX(point.p == point_kinematics),
            name=point.p.name() + "_kinematics_consistency",
        )

    def _add_contact_point_feasibility(
        self,
        complementarity_margin_fun: cs.Function,
        friction_margin_fun: cs.Function,
        height_fun: cs.Function,
        normal_force_fun: cs.Function,
        point: hp_rp.ContactPointState,
    ):
        problem = self.op.problem
        variables = self.op.variables  # type: Variables

        # Normal complementarity
        dcc_margin = complementarity_margin_fun(
            p=point.p,
            f=point.f * variables.mass,
            eps=variables.relaxed_complementarity_epsilon,
        )["relaxed_complementarity_margin"]
        problem.add_constraint(
            expression=cs.MX(dcc_margin >= 0),
            name=point.p.name() + "_complementarity",
        )

        # Point height greater than zero
        point_height = height_fun(p=point.p)["point_height"]
        problem.add_constraint(
            expression=cs.MX(point_height >= 0),
            name=point.p.name() + "_height",
        )

        # Normal force greater than zero
        normal_force = normal_force_fun(p=point.p, f=point.f)["normal_force"]
        problem.add_constraint(
            expression=cs.MX(normal_force >= 0),
            name=point.f.name() + "_normal",
        )

        # Friction
        friction_margin = friction_margin_fun(
            p=point.p,
            f=point.f,
            mu_s=variables.static_friction,
        )["friction_cone_square_margin"]
        problem.add_constraint(
            expression=cs.MX(friction_margin >= 0),
            name=point.f.name() + "_friction",
        )

    def _add_foot_regularization(
        self,
        points: list[hp_rp.ContactPointState],
        references: list[hp_rp.ContactPointState],
        point_position_expression_type: hp.ExpressionType,
    ) -> None:
        problem = self.op.problem

        assert len(points) > 0
        # Force ratio regularization
        sum_of_forces = cs.MX.zeros(3, 1)
        for point in points:
            sum_of_forces += point.f

        multiplier = 1.0 / len(points)

        for point in points:
            force_error_from_average = point.f - multiplier * sum_of_forces

            problem.add_cost(
                expression=cs.sumsqr(force_error_from_average),
                name=point.f.name() + "_average_regularization",
                scaling=self.settings.average_force_regularization_cost_multiplier,
            )

        # Force and position regularization
        for i, point in enumerate(points):
            problem.add_expression(
                expression=cs.sumsqr(point.p - references[i].p),
                name=point.p.name() + "_regularization",
                scaling=self.settings.point_position_regularization_cost_multiplier,
                mode=point_position_expression_type,
            )
            problem.add_cost(
                expression=cs.sumsqr(point.f - references[i].f),
                name=point.f.name() + "_regularization",
                scaling=self.settings.force_regularization_cost_multiplier,
            )

    def _apply_mass_regularization(self, input_var: Variables) -> Variables:
        if self.parametric_model:
            assert isinstance(
                self.kin_dyn_object, adam.parametric.casadi.KinDynComputationsParametric
            )
            numeric_mass_fun = self.kin_dyn_object.get_total_mass()
            numeric_mass = numeric_mass_fun(
                input_var.parametric_link_length_multipliers,
                input_var.parametric_link_densities,
            )
        else:
            numeric_mass = self.numeric_mass

        if isinstance(numeric_mass, float) and numeric_mass == 0:
            raise ValueError("The mass of the robot is zero. This is not supported.")

        output = input_var
        if output.state is not None:
            for point in (
                output.state.contact_points.left + output.state.contact_points.right
            ):
                point.f /= numeric_mass

        if output.references.state is not None:
            for point in (
                output.references.state.contact_points.left
                + output.references.state.contact_points.right
            ):
                point.f /= numeric_mass

        return output

    def _undo_mass_regularization(self, input_var: Variables) -> Variables:
        if self.parametric_model:
            assert isinstance(
                self.kin_dyn_object, adam.parametric.casadi.KinDynComputationsParametric
            )
            numeric_mass_fun = self.kin_dyn_object.get_total_mass()
            numeric_mass = numeric_mass_fun(
                input_var.parametric_link_length_multipliers,
                input_var.parametric_link_densities,
            )
        else:
            numeric_mass = self.numeric_mass

        if isinstance(numeric_mass, float) and numeric_mass == 0:
            raise ValueError("The mass of the robot is zero. This is not supported.")

        output = input_var
        if output.state is not None:
            for point in (
                output.state.contact_points.left + output.state.contact_points.right
            ):
                point.f *= numeric_mass

        if output.references.state is not None:
            for point in (
                output.references.state.contact_points.left
                + output.references.state.contact_points.right
            ):
                point.f *= numeric_mass

        return output

    def set_initial_guess(self, initial_guess: Variables) -> None:
        self.op.problem.set_initial_guess(
            self._apply_mass_regularization(initial_guess)
        )

    def get_initial_guess(self) -> Variables:
        return self._undo_mass_regularization(self.op.problem.get_initial_guess())

    def set_references(self, references: References) -> None:
        guess = (
            self.op.problem.get_initial_guess()
        )  # Avoid the undo of the mass regularization
        guess.references = references
        self.set_initial_guess(guess)

    def solve(self) -> hp.Output[Variables]:
        output = self.op.problem.solve()
        output.values = self._undo_mass_regularization(output.values)
        return output

    def to_function(
        self,
        input_name_prefix: str,
        function_name: str = "opti_function",
        options: dict = None,
    ) -> cs.Function:
        options = {} if options is None else options
        inner_function = self.optimization_solver.to_function(
            input_name_prefix=input_name_prefix,
            function_name=function_name + "_internal",
            options=options,
        )
        variable_names = inner_function.name_in()
        variables_list = []
        variables_sym = {}
        for n in variable_names:
            variables_sym[n] = cs.MX.sym(n, inner_function.size_in(n))
            variables_list.append(variables_sym[n])

        optimization_structure = copy.deepcopy(
            self.optimization_solver.get_optimization_objects()
        )
        optimization_structure.from_dict(
            input_dict=variables_sym, prefix=input_name_prefix
        )
        mass_regularized_vars = self._apply_mass_regularization(optimization_structure)
        output_dict = inner_function(
            **mass_regularized_vars.to_dict(prefix=input_name_prefix)
        )
        optimization_structure = copy.deepcopy(
            self.optimization_solver.get_optimization_objects()
        )
        optimization_structure.from_dict(input_dict=output_dict)
        mass_regularized_output = self._undo_mass_regularization(optimization_structure)
        mass_regularized_output_dict = mass_regularized_output.to_dict()

        output_values = []
        output_names = []
        for n in output_dict:
            output_values.append(mass_regularized_output_dict[n])
            output_names.append(n)

        return cs.Function(
            function_name,
            variables_list,
            output_values,
            variable_names,
            output_names,
            options,
        )

    def change_opti_options(
        self,
        inner_solver: str = None,
        options_solver: dict[str, Any] = None,
        options_plugin: dict[str, Any] = None,
    ) -> None:
        self.optimization_solver.set_opti_options(
            inner_solver, options_solver, options_plugin
        )

    def get_adam_model(self) -> adam.model.Model:
        if self.parametric_model:
            guess = self.get_initial_guess()
            original_length = guess.parametric_link_length_multipliers
            original_density = guess.parametric_link_densities
            factory = URDFParametricModelFactory(
                path=self.settings.robot_urdf,
                math=adam.numpy.numpy_like.SpatialMath(),
                links_name_list=self.settings.parametric_link_names,
                length_multiplier=original_length,
                densities=original_density,
            )
            model = adam.model.Model.build(
                factory=factory, joints_name_list=self.settings.joints_name_list
            )
            return model

        return self.kin_dyn_object.rbdalgos.model

    def get_variables_structure(self) -> Variables:
        return copy.deepcopy(self.variables)
