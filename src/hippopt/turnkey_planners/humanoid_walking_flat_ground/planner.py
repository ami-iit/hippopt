import copy
import dataclasses
import typing

import adam.casadi
import casadi as cs
import liecasadi
import numpy as np

import hippopt as hp
import hippopt.integrators as hp_int
import hippopt.robot_planning as hp_rp


@dataclasses.dataclass
class ExtendedContactPoint(hp_rp.ContactPoint):
    u_v: hp.StorageType = hp.default_storage_field(hp.Variable)

    desired_ratio: hp.StorageType = hp.default_storage_field(hp.Parameter)

    number_of_points: dataclasses.InitVar[int] = dataclasses.field(default=None)

    def __post_init__(
        self, input_descriptor: hp_rp.ContactPointDescriptor, number_of_points: int
    ) -> None:
        super().__post_init__(input_descriptor)
        self.u_v = np.zeros(3)
        self.desired_ratio = 1.0 / number_of_points


@dataclasses.dataclass
class FeetContactPoints(hp.OptimizationObject):
    left: list[ExtendedContactPoint] = hp.default_composite_field()
    right: list[ExtendedContactPoint] = hp.default_composite_field()


@dataclasses.dataclass
class FeetContactPointDescriptors:
    left: list[hp_rp.ContactPointDescriptor] = dataclasses.field(default_factory=list)
    right: list[hp_rp.ContactPointDescriptor] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Settings:
    robot_urdf: str = dataclasses.field(default=None)
    joints_name_list: list[str] = dataclasses.field(default=None)
    contact_points: FeetContactPointDescriptors = dataclasses.field(default=None)
    root_link: str = dataclasses.field(default=None)
    gravity: np.array = dataclasses.field(default=None)
    horizon_length: int = dataclasses.field(default=None)
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

    casadi_function_options: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        self.casadi_function_options = (
            self.casadi_function_options
            if isinstance(self.casadi_function_options, dict)
            else {}
        )
        self.root_link = "root_link"
        self.gravity = np.array([0.0, 0.0, -9.80665, 0.0, 0.0, 0.0])
        self.integrator = hp_int.ImplicitTrapezoid
        self.terrain = hp_rp.PlanarTerrain()
        self.planar_dcc_height_multiplier = 10.0
        self.dcc_gain = 20.0
        self.dcc_epsilon = 0.05
        self.static_friction = 0.3
        self.maximum_velocity_control = np.ndarray([2.0, 2.0, 5.0])
        self.maximum_force_derivative = np.ndarray([100.0, 100.0, 100.0])
        self.maximum_angular_momentum = 10.0

    def is_valid(self) -> bool:
        number_of_joints = len(self.joints_name_list)
        return (
            self.robot_urdf is not None
            and self.joints_name_list is not None
            and self.contact_points is not None
            and self.horizon_length is not None
            and self.minimum_com_height is not None
            and self.minimum_feet_lateral_distance is not None
            and self.maximum_feet_relative_height is not None
            and self.maximum_joint_positions is not None
            and self.minimum_joint_positions is not None
            and self.maximum_joint_velocities is not None
            and self.minimum_joint_velocities is not None
            and len(self.maximum_joint_positions) == number_of_joints
            and len(self.minimum_joint_positions) == number_of_joints
            and len(self.maximum_joint_velocities) == number_of_joints
            and len(self.minimum_joint_velocities) == number_of_joints
            and self.contacts_centroid_cost_multiplier is not None
            and self.com_linear_velocity_cost_weights is not None
            and len(self.com_linear_velocity_cost_weights) == 3
            and self.com_linear_velocity_cost_multiplier is not None
            and self.desired_frame_quaternion_cost_frame_name is not None
            and self.desired_frame_quaternion_cost_multiplier is not None
            and self.base_quaternion_cost_multiplier is not None
            and self.base_quaternion_velocity_cost_multiplier is not None
            and self.joint_regularization_cost_weights is not None
            and len(self.joint_regularization_cost_weights) == number_of_joints
            and self.joint_regularization_cost_multiplier is not None
            and self.force_regularization_cost_multiplier is not None
        )


@dataclasses.dataclass
class References(hp.OptimizationObject):
    contacts_centroid_cost_weights: hp.StorageType = hp.default_storage_field(
        hp.Parameter
    )
    contacts_centroid: hp.StorageType = hp.default_storage_field(hp.Parameter)

    com_linear_velocity: hp.StorageType = hp.default_storage_field(hp.Parameter)

    desired_frame_quaternion_xyzw: hp.StorageType = hp.default_storage_field(
        hp.Parameter
    )

    base_quaternion_xyzw: hp.StorageType = hp.default_storage_field(hp.Parameter)

    base_quaternion_xyzw_velocity: hp.StorageType = hp.default_storage_field(
        hp.Parameter
    )

    joint_regularization_cost: hp.StorageType = hp.default_storage_field(hp.Parameter)

    number_of_joints: dataclasses.InitVar[int] = dataclasses.field(default=None)

    def __post_init__(self, number_of_joints: int) -> None:
        self.contacts_centroid_cost_weights = np.zeros((3, 1))
        self.contacts_centroid = np.zeros((3, 1))
        self.com_linear_velocity = np.zeros((3, 1))
        self.desired_frame_quaternion_xyzw = np.zeros((4, 1))
        self.desired_frame_quaternion_xyzw[3] = 1
        self.base_quaternion_xyzw = np.zeros((4, 1))
        self.base_quaternion_xyzw[3] = 1
        self.base_quaternion_xyzw_velocity = np.zeros((4, 1))
        self.joint_regularization_cost = np.zeros((number_of_joints, 1))


@dataclasses.dataclass
class Variables(hp.OptimizationObject):
    contact_points: hp.CompositeType[FeetContactPoints] = hp.default_composite_field()
    com: hp.StorageType = hp.default_storage_field(hp.Variable)
    centroidal_momentum: hp.StorageType = hp.default_storage_field(hp.Variable)
    mass: hp.StorageType = hp.default_storage_field(hp.Parameter)
    kinematics: hp.CompositeType[
        hp_rp.FloatingBaseSystem
    ] = hp.default_composite_field()

    com_initial: hp.StorageType = hp.default_storage_field(hp.Parameter)
    centroidal_momentum_initial: hp.StorageType = hp.default_storage_field(hp.Parameter)

    dt: hp.StorageType = hp.default_storage_field(hp.Parameter)
    gravity: hp.StorageType = hp.default_storage_field(hp.Parameter)
    planar_dcc_height_multiplier: hp.StorageType = hp.default_storage_field(
        hp.Parameter
    )
    dcc_gain: hp.StorageType = hp.default_storage_field(hp.Parameter)
    dcc_epsilon: hp.StorageType = hp.default_storage_field(hp.Parameter)
    static_friction: hp.StorageType = hp.default_storage_field(hp.Parameter)
    maximum_velocity_control: hp.StorageType = hp.default_storage_field(hp.Parameter)
    maximum_force_derivative: hp.StorageType = hp.default_storage_field(hp.Parameter)
    maximum_angular_momentum: hp.StorageType = hp.default_storage_field(hp.Parameter)
    minimum_com_height: hp.StorageType = hp.default_storage_field(hp.Parameter)
    minimum_feet_lateral_distance: hp.StorageType = hp.default_storage_field(
        hp.Parameter
    )
    maximum_feet_relative_height: hp.StorageType = hp.default_storage_field(
        hp.Parameter
    )
    maximum_joint_positions: hp.StorageType = hp.default_storage_field(hp.Parameter)
    minimum_joint_positions: hp.StorageType = hp.default_storage_field(hp.Parameter)
    maximum_joint_velocities: hp.StorageType = hp.default_storage_field(hp.Parameter)
    minimum_joint_velocities: hp.StorageType = hp.default_storage_field(hp.Parameter)

    references: hp.CompositeType[References] = hp.default_composite_field()

    settings: dataclasses.InitVar[Settings] = dataclasses.field(default=None)
    kin_dyn_object: dataclasses.InitVar[
        adam.casadi.KinDynComputations
    ] = dataclasses.field(default=None)

    def __post_init__(
        self,
        settings: Settings,
        kin_dyn_object: adam.casadi.KinDynComputations,
    ) -> None:
        self.contact_points.left = [
            ExtendedContactPoint(
                descriptor=point, number_of_points=len(settings.contact_points.left)
            )
            for point in settings.contact_points.left
        ]
        self.contact_points.right = [
            ExtendedContactPoint(
                descriptor=point, number_of_points=len(settings.contact_points.right)
            )
            for point in settings.contact_points.right
        ]

        self.com = np.zeros(3)
        self.centroidal_momentum = np.zeros(6)
        self.kinematics = hp_rp.FloatingBaseSystem(kin_dyn_object.NDoF)
        self.dt = 0.1
        self.gravity = kin_dyn_object.g[:, 3]
        self.mass = kin_dyn_object.get_total_mass()

        self.com_initial = np.zeros(3)
        self.centroidal_momentum_initial = np.zeros(6)

        self.planar_dcc_height_multiplier = settings.planar_dcc_height_multiplier
        self.dcc_gain = settings.dcc_gain
        self.dcc_epsilon = settings.dcc_epsilon
        self.static_friction = settings.static_friction
        self.maximum_velocity_control = settings.maximum_velocity_control
        self.maximum_force_derivative = settings.maximum_force_derivative
        self.maximum_angular_momentum = settings.maximum_angular_momentum
        self.minimum_com_height = settings.minimum_com_height
        self.minimum_feet_lateral_distance = settings.minimum_feet_lateral_distance
        self.maximum_feet_relative_height = settings.maximum_feet_relative_height
        self.maximum_joint_positions = settings.maximum_joint_positions
        self.minimum_joint_positions = settings.minimum_joint_positions
        self.maximum_joint_velocities = settings.maximum_joint_velocities
        self.minimum_joint_velocities = settings.minimum_joint_velocities
        self.references = References(number_of_joints=kin_dyn_object.NDoF)


class HumanoidWalkingFlatGround:
    def __init__(self, settings: Settings) -> None:
        if not settings.is_valid():
            raise ValueError("Settings are not valid")
        self.settings = copy.deepcopy(settings)
        self.kin_dyn_object = adam.casadi.KinDynComputations(
            urdfstring=self.settings.robot_urdf,
            joints_name_list=self.settings.joints_name_list,
            root_link=self.settings.root_link,
            gravity=self.settings.gravity,
            f_opts=self.settings.casadi_function_options,
        )

        self.variables = Variables(
            settings=self.settings, kin_dyn_object=self.kin_dyn_object
        )

        self.ocp = hp.OptimalControlProblem.create(
            self.variables, horizon_length=self.settings.horizon_length
        )

        sym = self.ocp.symbolic_structure

        function_inputs = {
            "mass_name": sym.mass.name(),
            "momentum_name": sym.centroidal_momentum.name(),
            "com_name": sym.com.name(),
            "quaternion_xyzw_name": "q",
            "gravity_name": sym.gravity.name(),
            "point_position_names": [],
            "point_force_names": [],
            "point_position_in_frame_name": "p_parent",
            "base_position_name": "pb",
            "base_quaternion_xyzw_name": "qb",
            "joint_positions_name": "s",
            "base_position_derivative_name": "pb_dot",
            "base_quaternion_xyzw_derivative_name": "qb_dot",
            "joint_velocities_name": "s_dot",
            "point_position_name": "p",
            "point_force_name": "f",
            "point_velocity_name": "v",
            "point_force_derivative_name": "f_dot",
            "point_position_control_name": "u_p",
            "height_multiplier_name": "kt",
            "dcc_gain_name": "k_bs",
            "dcc_epsilon_name": "eps",
            "static_friction_name": "mu_s",
            "desired_quaternion_xyzw_name": "qd",
            "options": self.settings.casadi_function_options,
        }

        # Normalized quaternion computation
        normalized_quaternion_fun = hp_rp.quaternion_xyzw_normalization(
            **function_inputs
        )
        normalized_quaternion = normalized_quaternion_fun(
            q=sym.kinematics.base.quaternion_xyzw
        )["quaternion_normalized"]

        # Align names used in the terrain function with those in function_inputs
        self.settings.terrain.change_options(**function_inputs)

        # Definition of contact constraint functions
        dcc_planar_fun = hp_rp.dcc_planar_complementarity(
            terrain=self.settings.terrain,
            **function_inputs,
        )
        dcc_margin_fun = hp_rp.dcc_complementarity_margin(
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
        all_contact_points = sym.contact_points.left + sym.contact_points.right

        for point in all_contact_points:
            self.add_point_dynamics(point)

            self.add_contact_point_feasibility(
                dcc_margin_fun,
                dcc_planar_fun,
                friction_margin_fun,
                height_fun,
                normal_force_fun,
                point,
            )

            self.add_contact_kinematic_consistency(
                function_inputs,
                normalized_quaternion,
                point,
                point_kinematics_functions,
            )

        self.add_robot_dynamics(all_contact_points, function_inputs)

        self.add_kinematics_constraints(
            function_inputs, height_fun, normalized_quaternion
        )
        self.add_kinematics_regularization(function_inputs=function_inputs)

        self.add_contact_centroids_expressions(function_inputs)

        self.add_foot_regularization(sym.contact_points.left)
        self.add_foot_regularization(sym.contact_points.right)

    def add_contact_centroids_expressions(self, function_inputs):
        problem = self.ocp.problem
        sym = self.ocp.symbolic_structure

        # Maximum feet relative height
        def get_centroid(
            points: list[ExtendedContactPoint], function_inputs_dict: dict
        ) -> cs.MX:
            function_inputs_dict["point_position_names"] = [
                pt.p.name() for pt in points
            ]
            point_position_dict = {pt.p.name(): pt.p for pt in points}
            centroid_fun = hp_rp.contact_points_centroid(
                number_of_points=len(function_inputs_dict["point_position_names"]),
                **function_inputs_dict,
            )
            return centroid_fun(**point_position_dict)["centroid"]

        left_centroid = get_centroid(
            points=sym.contact_points.left, function_inputs_dict=function_inputs
        )
        right_centroid = get_centroid(
            points=sym.contact_points.right, function_inputs_dict=function_inputs
        )
        problem.add_expression_to_horizon(
            expression=cs.Opti_bounded(
                -sym.maximum_feet_relative_height,
                (left_centroid[2] - right_centroid[2]),
                sym.maximum_feet_relative_height,
            ),
            apply_to_first_elements=False,
            name="maximum_feet_relative_height",
        )

        # Contact centroid position cost
        centroid_error = sym.contacts_centroid_references - 0.5 * (
            left_centroid + right_centroid
        )
        weighted_centroid_squared_error = (
            centroid_error.T()
            @ cs.diag(sym.contacts_centroid_cost_weights)
            @ centroid_error
        )
        problem.add_expression_to_horizon(
            expression=weighted_centroid_squared_error,
            apply_to_first_elements=False,
            name="contacts_centroid_cost",
            mode=hp.ExpressionType.minimize,
            scaling=self.settings.contacts_centroid_cost_multiplier,
        )

    def add_kinematics_constraints(
        self,
        function_inputs: dict,
        height_fun: cs.Function,
        normalized_quaternion: cs.MX,
    ):
        problem = self.ocp.problem
        sym = self.ocp.symbolic_structure

        # Unitary quaternion
        problem.add_expression_to_horizon(
            expression=cs.MX(cs.sumsqr(sym.kinematics.base.quaternion_xyzw) == 1),
            apply_to_first_elements=False,
            name="unitary_quaternion",
        )

        # Consistency of com position with kinematics
        com_kinematics_fun = hp_rp.center_of_mass_position_from_kinematics(
            kindyn_object=self.kin_dyn_object, **function_inputs
        )
        com_kinematics = com_kinematics_fun(
            pb=sym.kinematics.base.position,
            qb=normalized_quaternion,
            s=sym.kinematics.joints.positions,
        )["com_position"]
        problem.add_expression_to_horizon(
            expression=cs.MX(sym.com == com_kinematics),
            apply_to_first_elements=False,
            name="com_kinematics_consistency",
        )

        # Consistency of centroidal momentum (angular part only) with kinematics
        centroidal_kinematics_fun = hp_rp.centroidal_momentum_from_kinematics(
            kindyn_object=self.kin_dyn_object, **function_inputs
        )
        centroidal_kinematics = centroidal_kinematics_fun(
            pb=sym.kinematics.base.position,
            qb=normalized_quaternion,
            s=sym.kinematics.joints.positions,
            pb_dot=sym.kinematics.base.linear_velocity,
            qb_dot=sym.kinematics.base.quaternion_velocity_xyzw,
            s_dot=sym.kinematics.joints.velocities,
        )["h_g"]
        problem.add_expression_to_horizon(
            expression=cs.MX(sym.centroidal_momentum[3:] == centroidal_kinematics[3:]),
            apply_to_first_elements=True,
            name="centroidal_momentum_kinematics_consistency",
        )

        # Bounds on angular momentum
        problem.add_expression_to_horizon(
            expression=cs.Opti_bounded(
                -sym.maximum_angular_momentum,
                sym.centroidal_momentum[3:],
                sym.maximum_angular_momentum,
            ),
            apply_to_first_elements=True,
            name="angular_momentum_bounds",
        )

        # Minimum com height
        com_height = height_fun(p=sym.com)["point_height"]
        problem.add_expression_to_horizon(
            expression=cs.MX(com_height >= sym.minimum_com_height),
            apply_to_first_elements=False,
            name="minimum_com_height",
        )

        # Minimum feet lateral distance
        left_frame = sym.contact_points.left[0].descriptor.foot_frame
        right_frame = sym.contact_points.right[0].descriptor.foot_frame
        relative_position_fun = hp_rp.frames_relative_position(
            kindyn_object=self.kin_dyn_object,
            reference_frame=right_frame,
            target_frame=left_frame,
            **function_inputs,
        )
        relative_position = relative_position_fun(s=sym.kinematics.joints.positions)[
            "relative_position"
        ]
        problem.add_expression_to_horizon(
            expression=cs.MX(
                relative_position[:2] >= sym.minimum_feet_lateral_distance
            ),
            apply_to_first_elements=False,
            name="minimum_feet_distance",
        )

        # Joint position bounds
        problem.add_expression_to_horizon(
            expression=cs.Opti_bounded(
                sym.minimum_joint_positions,
                sym.kinematics.joints.positions,
                sym.maximum_joint_positions,
            ),
            apply_to_first_elements=False,
            name="joint_position_bounds",
        )

        # Joint velocity bounds
        problem.add_expression_to_horizon(
            expression=cs.Opti_bounded(
                sym.minimum_joint_velocities,
                sym.kinematics.joints.velocities,
                sym.maximum_joint_velocities,
            ),
            apply_to_first_elements=True,
            name="joint_velocity_bounds",
        )

    def add_kinematics_regularization(self, function_inputs: dict):
        problem = self.ocp.problem
        sym = self.ocp.symbolic_structure
        # Desired com velocity
        com_velocity_error = (
            sym.centroidal_momentum[:3] - sym.references.com_linear_velocity * sym.mass
        )
        com_velocity_weighted_error = (
            com_velocity_error.T()
            * cs.diag(cs.DM(self.settings.com_linear_velocity_cost_weights))
            * com_velocity_error
        )
        problem.add_expression_to_horizon(
            expression=com_velocity_weighted_error,
            apply_to_first_elements=True,
            name="com_velocity_error",
            mode=hp.ExpressionType.minimize,
            scaling=self.settings.com_linear_velocity_cost_multiplier,
        )

        # Desired frame orientation
        quaternion_error_fun = hp_rp.quaternion_error(
            kindyn_object=self.kin_dyn_object,
            target_frame=self.settings.desired_frame_quaternion_cost_frame_name,
            **function_inputs,
        )
        quaternion_error = quaternion_error_fun(
            pb=sym.kinematics.base.position,
            qb=sym.kinematics.base.quaternion_xyzw,
            s=sym.kinematics.joints.positions,
            qd=sym.references.desired_frame_quaternion_xyzw,
        )["quaternion_error"]
        problem.add_expression_to_horizon(
            expression=cs.sumsqr(quaternion_error),
            apply_to_first_elements=False,
            name="frame_quaternion_error",
            mode=hp.ExpressionType.minimize,
            scaling=self.settings.desired_frame_quaternion_cost_multiplier,
        )

        # Desired base orientation
        # TODO: use the actual reference
        identity = liecasadi.SO3.Identity()
        problem.add_expression_to_horizon(
            expression=cs.sumsqr(sym.kinematics.base.quaternion_xyzw - identity),
            apply_to_first_elements=False,
            name="base_quaternion_error",
            mode=hp.ExpressionType.minimize,
            scaling=self.settings.base_quaternion_cost_multiplier,
        )

        # Desired base angular velocity
        problem.add_expression_to_horizon(
            expression=cs.sumsqr(
                sym.kinematics.base.quaternion_velocity_xyzw
                - sym.references.base_quaternion_xyzw_velocity
            ),
            apply_to_first_elements=True,
            name="base_quaternion_velocity_error",
            mode=hp.ExpressionType.minimize,
            scaling=self.settings.base_quaternion_velocity_cost_multiplier,
        )

        # Desired joint positions
        joint_positions_error = (
            sym.kinematics.joints.positions - sym.references.joint_regularization
        )
        joint_velocity_error = (
            sym.kinematics.joints.velocities
            + cs.diag(cs.DM(self.settings.joint_regularization_cost_weights))
            * joint_positions_error
        )
        problem.add_expression_to_horizon(
            expression=cs.sumsqr(joint_velocity_error),
            apply_to_first_elements=False,
            name="joint_positions_error",
            mode=hp.ExpressionType.minimize,
            scaling=self.settings.joint_regularization_cost_multiplier,
        )

    def add_robot_dynamics(self, all_contact_points: list, function_inputs: dict):
        problem = self.ocp.problem
        sym = self.ocp.symbolic_structure
        default_integrator = self.settings.integrator

        # dot(pb) = pb_dot (base position dynamics)
        problem.add_dynamics(
            hp.dot(sym.kinematics.base.position) == sym.kinematics.base.linear_velocity,
            x0=problem.initial(sym.kinematics.base.initial_position),
            integrator=default_integrator,
            name="base_position_dynamics",
        )

        # dot(q) = q_dot (base quaternion dynamics)
        problem.add_dynamics(
            hp.dot(sym.kinematics.base.quaternion_xyzw)
            == sym.kinematics.base.quaternion_velocity_xyzw,
            x0=problem.initial(sym.kinematics.base.initial_quaternion_xyzw),
            integrator=default_integrator,
            name="base_quaternion_dynamics",
        )

        # dot(s) = s_dot (joint position dynamics)
        problem.add_dynamics(
            hp.dot(sym.kinematics.joints.positions) == sym.kinematics.joints.velocities,
            x0=problem.initial(sym.kinematics.joints.initial_positions),
            integrator=default_integrator,
            name="joint_position_dynamics",
        )

        # dot(com) = h_g[:3]/m (center of mass dynamics)
        com_dynamics = hp_rp.com_dynamics_from_momentum(**function_inputs)
        problem.add_dynamics(
            hp.dot(sym.com) == com_dynamics,
            x0=problem.initial(sym.com_initial),
            integrator=default_integrator,
            name="com_dynamics",
        )

        # dot(h) = sum_i (p_i x f_i) + mg (centroidal momentum dynamics)
        function_inputs["point_position_names"] = [
            point.p.name() for point in all_contact_points
        ]
        function_inputs["point_force_names"] = [
            point.f.name() for point in all_contact_points
        ]
        centroidal_dynamics = hp_rp.centroidal_dynamics_with_point_forces(
            number_of_points=len(function_inputs["point_position_names"]),
            **function_inputs,
        )
        problem.add_dynamics(
            hp.dot(sym.centroidal_momentum) == centroidal_dynamics,
            x0=problem.initial(sym.centroidal_momentum_initial),
            integrator=default_integrator,
            name="centroidal_momentum_dynamics",
        )

    def add_contact_kinematic_consistency(
        self,
        function_inputs: dict,
        normalized_quaternion: cs.MX,
        point: ExtendedContactPoint,
        point_kinematics_functions: dict,
    ):
        problem = self.ocp.problem
        sym = self.ocp.symbolic_structure

        # Creation of contact kinematics consistency functions
        descriptor = point.descriptor
        if descriptor.foot_frame not in point_kinematics_functions:
            point_kinematics_functions[
                descriptor.foot_frame
            ] = hp_rp.point_position_from_kinematics(
                kindyn_object=self.kin_dyn_object,
                frame_name=descriptor.foot_frame,
                **function_inputs,
            )

        # Consistency between the contact position and the kinematics
        point_kinematics = point_kinematics_functions[descriptor.foot_frame](
            pb=sym.kinematics.base.position,
            qb=normalized_quaternion,
            s=sym.kinematics.joints.positions,
            p_parent=descriptor.position_in_foot_frame,
        )["point_position"]
        problem.add_expression_to_horizon(
            expression=cs.MX(point.p == point_kinematics),
            apply_to_first_elements=False,
            name=point.p.name() + "_kinematics_consistency",
        )

    def add_contact_point_feasibility(
        self,
        dcc_margin_fun: cs.Function,
        dcc_planar_fun: cs.Function,
        friction_margin_fun: cs.Function,
        height_fun: cs.Function,
        normal_force_fun: cs.Function,
        point: ExtendedContactPoint,
    ):
        problem = self.ocp.problem
        sym = self.ocp.symbolic_structure

        # Planar complementarity
        dcc_planar = dcc_planar_fun(
            p=point.p, kt=sym.planar_dcc_height_multiplier, u_p=point.u_v
        )["planar_complementarity"]
        problem.add_expression_to_horizon(
            expression=cs.MX(point.v == dcc_planar),
            apply_to_first_elements=True,
            name=point.p.name() + "_planar_complementarity",
        )

        # Normal complementarity
        dcc_margin = dcc_margin_fun(
            p=point.p,
            f=point.f,
            v=point.v,
            f_dot=point.f_dot,
            k_bs=sym.dcc_gain,
            eps=sym.dcc_epsilon,
        )["dcc_complementarity_margin"]
        problem.add_expression_to_horizon(
            expression=cs.MX(dcc_margin >= 0),
            apply_to_first_elements=True,
            name=point.p.name() + "_dcc",
        )

        # Point height greater than zero
        point_height = height_fun(p=point.p)["point_height"]
        problem.add_expression_to_horizon(
            expression=cs.MX(point_height >= 0),
            apply_to_first_elements=False,
            name=point.p.name() + "_height",
        )

        # Normal force greater than zero
        normal_force = normal_force_fun(p=point.p, f=point.f)["normal_force"]
        problem.add_expression_to_horizon(
            expression=cs.MX(normal_force >= 0),
            apply_to_first_elements=False,
            name=point.f.name() + "_normal",
        )

        # Friction
        friction_margin = friction_margin_fun(
            p=point.p,
            f=point.f,
            mu_s=sym.static_friction,
        )["friction_cone_square_margin"]
        problem.add_expression_to_horizon(
            expression=cs.MX(friction_margin >= 0),
            apply_to_first_elements=False,
            name=point.f.name() + "_friction",
        )

        # Bounds on contact velocity control inputs
        problem.add_expression_to_horizon(
            expression=cs.Opti_bounded(
                -sym.maximum_velocity_control,
                point.u_v,
                sym.maximum_velocity_control,
            ),
            apply_to_first_elements=True,
            name=point.u_v.name() + "_bounds",  # noqa
        )

        # Bounds on contact force control inputs
        problem.add_expression_to_horizon(
            expression=cs.Opti_bounded(
                -sym.maximum_force_control,
                point.u_f,  # noqa
                sym.maximum_force_control,
            ),
            apply_to_first_elements=True,
            name=point.u_f.name() + "_bounds",  # noqa
        )

    def add_point_dynamics(self, point: ExtendedContactPoint) -> None:
        default_integrator = self.settings.integrator
        problem = self.ocp.problem

        # dot(f) = f_dot
        problem.add_dynamics(
            hp.dot(point.f) == point.f_dot,
            x0=problem.initial(point.f0),
            integrator=default_integrator,
            name=point.f.name() + "_dynamics",
        )

        # dot(p) = v
        problem.add_dynamics(
            hp.dot(point.p) == point.v,
            x0=problem.initial(point.p0),
            integrator=default_integrator,
            name=point.p.name() + "_dynamics",
        )

    def add_foot_regularization(self, points: list[ExtendedContactPoint]) -> None:
        problem = self.ocp.problem

        # Force ratio regularization
        def sum_of_other_forces(
            input_points: list[ExtendedContactPoint], point_index: int
        ) -> cs.MX:
            output_force = cs.MX.zeros(3, 1)
            for i in range(len(input_points)):
                if i != point_index:
                    output_force += input_points[i].f
            return output_force

        for point in points:
            force_error = point.f - point.desired_ratio * sum_of_other_forces(
                points, points.index(point)
            )

            problem.add_expression_to_horizon(
                expression=cs.sumsqr(force_error),
                apply_to_first_elements=False,
                name=point.f.name() + "_regularization",
                mode=hp.ExpressionType.minimize,
                scaling=self.settings.force_regularization_cost_multiplier,
            )

    def set_initial_conditions(self) -> None:  # TODO: fill
        pass

    def set_references(self, references: References) -> None:  # TODO: fill
        pass
