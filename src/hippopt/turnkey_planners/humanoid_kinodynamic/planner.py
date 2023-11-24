import copy
import dataclasses
import typing

import adam.casadi
import casadi as cs
import numpy as np

import hippopt as hp
import hippopt.integrators as hp_int
import hippopt.robot_planning as hp_rp


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

    casadi_function_options: dict = dataclasses.field(default_factory=dict)

    casadi_opti_options: dict = dataclasses.field(default_factory=dict)

    casadi_solver_options: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root_link = "root_link"
        self.gravity = np.array([0.0, 0.0, -9.80665, 0.0, 0.0, 0.0])
        self.integrator = hp_int.ImplicitTrapezoid
        self.terrain = hp_rp.PlanarTerrain()
        self.planar_dcc_height_multiplier = 10.0
        self.dcc_gain = 20.0
        self.dcc_epsilon = 0.05
        self.static_friction = 0.3
        self.maximum_velocity_control = np.array([2.0, 2.0, 5.0])
        self.maximum_force_derivative = np.array([100.0, 100.0, 100.0])
        self.maximum_angular_momentum = 10.0

    def is_valid(self) -> bool:
        number_of_joints = len(self.joints_name_list)
        return (
            self.robot_urdf is not None
            and self.joints_name_list is not None
            and self.contact_points is not None
            and self.horizon_length is not None
            and self.time_step is not None
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
            and self.foot_yaw_regularization_cost_multiplier is not None
            and self.swing_foot_height_cost_multiplier is not None
            and self.contact_velocity_control_cost_multiplier is not None
            and self.contact_force_control_cost_multiplier is not None
        )


@dataclasses.dataclass
class ContactReferences(hp.OptimizationObject):
    desired_force_ratio: hp.StorageType = hp.default_storage_field(hp.Parameter)

    number_of_points: dataclasses.InitVar[int] = dataclasses.field(default=None)

    def __post_init__(self, number_of_points: int) -> None:
        self.desired_force_ratio = (
            1.0 / number_of_points
            if number_of_points is not None and number_of_points > 0
            else 0.0
        )


@dataclasses.dataclass
class FootReferences(hp.OptimizationObject):
    points: hp.CompositeType[list[ContactReferences]] = hp.default_composite_field(
        factory=list, time_varying=False
    )
    yaw: hp.StorageType = hp.default_storage_field(hp.Parameter)

    number_of_points: dataclasses.InitVar[int] = dataclasses.field(default=0)

    def __post_init__(self, number_of_points: int) -> None:
        number_of_points = number_of_points if number_of_points is not None else 0
        self.points = [
            ContactReferences(number_of_points=number_of_points)
            for _ in range(number_of_points)
        ]

        self.yaw = 0.0


@dataclasses.dataclass
class FeetReferences(hp.OptimizationObject):
    left: hp.CompositeType[FootReferences] = hp.default_composite_field(
        factory=FootReferences, time_varying=False
    )
    right: hp.CompositeType[FootReferences] = hp.default_composite_field(
        factory=FootReferences, time_varying=False
    )

    desired_swing_height: hp.StorageType = hp.default_storage_field(hp.Parameter)

    number_of_points_left: dataclasses.InitVar[int] = dataclasses.field(default=0)
    number_of_points_right: dataclasses.InitVar[int] = dataclasses.field(default=0)

    def __post_init__(
        self, number_of_points_left: int, number_of_points_right: int
    ) -> None:
        self.left = FootReferences(number_of_points=number_of_points_left)
        self.right = FootReferences(number_of_points=number_of_points_right)
        self.desired_swing_height = 0.02


@dataclasses.dataclass
class References(hp.OptimizationObject):
    feet: hp.CompositeType[FeetReferences] = hp.default_composite_field(
        factory=FeetReferences, time_varying=False
    )

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

    joint_regularization: hp.StorageType = hp.default_storage_field(hp.Parameter)

    number_of_joints: dataclasses.InitVar[int] = dataclasses.field(default=0)
    number_of_points_left: dataclasses.InitVar[int] = dataclasses.field(default=0)
    number_of_points_right: dataclasses.InitVar[int] = dataclasses.field(default=0)

    def __post_init__(
        self,
        number_of_joints: int,
        number_of_points_left: int,
        number_of_points_right: int,
    ) -> None:
        self.feet = FeetReferences(
            number_of_points_left=number_of_points_left,
            number_of_points_right=number_of_points_right,
        )
        self.contacts_centroid_cost_weights = np.zeros((3, 1))
        self.contacts_centroid = np.zeros((3, 1))
        self.com_linear_velocity = np.zeros((3, 1))
        self.desired_frame_quaternion_xyzw = np.zeros((4, 1))
        self.desired_frame_quaternion_xyzw[3] = 1
        self.base_quaternion_xyzw = np.zeros((4, 1))
        self.base_quaternion_xyzw[3] = 1
        self.base_quaternion_xyzw_velocity = np.zeros((4, 1))
        self.joint_regularization = np.zeros((number_of_joints, 1))


@dataclasses.dataclass
class ExtendedContactPoint(
    hp_rp.ContactPointState,
    hp_rp.ContactPointStateDerivative,
):
    u_v: hp.StorageType = hp.default_storage_field(hp.Variable)

    def __post_init__(self, input_descriptor: hp_rp.ContactPointDescriptor) -> None:
        hp_rp.ContactPointState.__post_init__(self, input_descriptor)
        hp_rp.ContactPointStateDerivative.__post_init__(self)
        self.u_v = np.zeros(3)

    def to_contact_point_state(self) -> hp_rp.ContactPointState:
        output = hp_rp.ContactPointState()
        output.p = self.p
        output.f = self.f
        output.descriptor = self.descriptor
        return output


@dataclasses.dataclass
class FeetContactPointsExtended(hp.OptimizationObject):
    left: list[ExtendedContactPoint] = hp.default_composite_field(factory=list)
    right: list[ExtendedContactPoint] = hp.default_composite_field(factory=list)

    def to_feet_contact_points(self) -> hp_rp.FeetContactPoints:
        output = hp_rp.FeetContactPoints()
        output.left = hp_rp.FootContactState.from_list(
            [point.to_contact_point_state() for point in self.left]
        )
        output.right = hp_rp.FootContactState.from_list(
            [point.to_contact_point_state() for point in self.right]
        )
        return output


@dataclasses.dataclass
class ExtendedHumanoid(hp.OptimizationObject):
    contact_points: hp.CompositeType[
        FeetContactPointsExtended
    ] = hp.default_composite_field(factory=FeetContactPointsExtended)

    kinematics: hp.CompositeType[hp_rp.FloatingBaseSystem] = hp.default_composite_field(
        cls=hp.Variable, factory=hp_rp.FloatingBaseSystem
    )

    com: hp.StorageType = hp.default_storage_field(hp.Variable)
    centroidal_momentum: hp.StorageType = hp.default_storage_field(hp.Variable)

    contact_point_descriptors: dataclasses.InitVar[
        hp_rp.FeetContactPointDescriptors
    ] = dataclasses.field(default=None)
    number_of_joints: dataclasses.InitVar[int] = dataclasses.field(default=None)

    def __post_init__(
        self,
        contact_point_descriptors: hp_rp.FeetContactPointDescriptors,
        number_of_joints: int,
    ) -> None:
        if contact_point_descriptors is not None:
            self.contact_points.left = [
                ExtendedContactPoint(input_descriptor=point)
                for point in contact_point_descriptors.left
            ]
            self.contact_points.right = [
                ExtendedContactPoint(input_descriptor=point)
                for point in contact_point_descriptors.right
            ]

        self.com = np.zeros(3)
        self.centroidal_momentum = np.zeros(6)
        self.kinematics = hp_rp.FloatingBaseSystem(number_of_joints=number_of_joints)

    def to_humanoid_state(self) -> hp_rp.HumanoidState:
        output = hp_rp.HumanoidState()
        output.kinematics = self.kinematics.to_floating_base_system_state()
        output.contact_points = self.contact_points.to_feet_contact_points()
        output.com = self.com
        return output


@dataclasses.dataclass
class ExtendedHumanoidState(hp_rp.HumanoidState):
    centroidal_momentum: hp.StorageType = hp.default_storage_field(hp.Variable)

    def __post_init__(
        self,
        contact_point_descriptors: hp_rp.FeetContactPointDescriptors,
        number_of_joints: int,
    ) -> None:
        hp_rp.HumanoidState.__post_init__(
            self,
            contact_point_descriptors=contact_point_descriptors,
            number_of_joints=number_of_joints,
        )
        self.centroidal_momentum = np.zeros(6)


@dataclasses.dataclass
class Variables(hp.OptimizationObject):
    system: hp.CompositeType[ExtendedHumanoid] = hp.default_composite_field(
        cls=hp.Variable, factory=ExtendedHumanoid
    )

    mass: hp.StorageType = hp.default_storage_field(hp.Parameter)

    initial_state: hp.CompositeType[ExtendedHumanoidState] = hp.default_composite_field(
        cls=hp.Parameter, factory=ExtendedHumanoidState, time_varying=False
    )

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

    references: hp.CompositeType[References] = hp.default_composite_field(
        factory=References, time_varying=True
    )

    settings: dataclasses.InitVar[Settings] = dataclasses.field(default=None)
    kin_dyn_object: dataclasses.InitVar[
        adam.casadi.KinDynComputations
    ] = dataclasses.field(default=None)

    def __post_init__(
        self,
        settings: Settings,
        kin_dyn_object: adam.casadi.KinDynComputations,
    ) -> None:
        self.system = ExtendedHumanoid(
            contact_point_descriptors=settings.contact_points,
            number_of_joints=kin_dyn_object.NDoF,
        )

        self.initial_state = ExtendedHumanoidState(
            contact_point_descriptors=settings.contact_points,
            number_of_joints=kin_dyn_object.NDoF,
        )

        self.dt = settings.time_step
        self.gravity = kin_dyn_object.g
        self.mass = kin_dyn_object.get_total_mass()

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
        self.references = References(
            number_of_joints=kin_dyn_object.NDoF,
            number_of_points_left=len(settings.contact_points.left),
            number_of_points_right=len(settings.contact_points.right),
        )


class Planner:
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
        self.numeric_mass = self.kin_dyn_object.get_total_mass()

        self.variables = Variables(
            settings=self.settings, kin_dyn_object=self.kin_dyn_object
        )

        optimization_solver = hp.OptiSolver(
            inner_solver=self.settings.opti_solver,
            problem_type=self.settings.problem_type,
            options_solver=self.settings.casadi_solver_options,
            options_plugin=self.settings.casadi_opti_options,
        )
        ocp_solver = hp.MultipleShootingSolver(optimization_solver=optimization_solver)

        self.ocp = hp.OptimalControlProblem.create(
            input_structure=self.variables,
            optimal_control_solver=ocp_solver,
            horizon=self.settings.horizon_length,
        )

        sym = self.ocp.symbolic_structure  # type: Variables

        function_inputs = self.get_function_inputs_dict()

        # Normalized quaternion computation
        normalized_quaternion_fun = hp_rp.quaternion_xyzw_normalization(
            **function_inputs
        )
        normalized_quaternion = normalized_quaternion_fun(
            q=sym.system.kinematics.base.quaternion_xyzw
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
        all_contact_points = (
            sym.system.contact_points.left + sym.system.contact_points.right
        )
        all_contact_initial_state = (
            sym.initial_state.contact_points.left
            + sym.initial_state.contact_points.right
        )

        for i, point in enumerate(all_contact_points):
            self.add_point_dynamics(point, initial_state=all_contact_initial_state[i])

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

            self.add_contact_point_regularization(
                point=point,
                desired_swing_height=sym.references.feet.desired_swing_height,
                function_inputs=function_inputs,
            )

        self.add_robot_dynamics(all_contact_points, function_inputs)

        self.add_kinematics_constraints(
            function_inputs, height_fun, normalized_quaternion
        )
        self.add_kinematics_regularization(
            function_inputs=function_inputs,
            base_quaternion_normalized=normalized_quaternion,
        )

        self.add_contact_centroids_expressions(function_inputs)

        self.add_foot_regularization(
            points=sym.system.contact_points.left,
            descriptors=self.settings.contact_points.left,
            references=sym.references.feet.left,
            function_inputs=function_inputs,
            foot_name="left",
        )
        self.add_foot_regularization(
            points=sym.system.contact_points.right,
            descriptors=self.settings.contact_points.right,
            references=sym.references.feet.right,
            function_inputs=function_inputs,
            foot_name="right",
        )

    def get_function_inputs_dict(self):
        sym = self.ocp.symbolic_structure
        function_inputs = {
            "mass_name": sym.mass.name(),
            "momentum_name": sym.system.centroidal_momentum.name(),
            "com_name": sym.system.com.name(),
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
            "first_point_name": "p_0",
            "second_point_name": "p_1",
            "desired_yaw_name": "yd",
            "desired_height_name": "hd",
            "options": self.settings.casadi_function_options,
        }
        return function_inputs

    def add_contact_centroids_expressions(self, function_inputs):
        problem = self.ocp.problem
        sym = self.ocp.symbolic_structure  # type: Variables

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
            points=sym.system.contact_points.left, function_inputs_dict=function_inputs
        )
        right_centroid = get_centroid(
            points=sym.system.contact_points.right, function_inputs_dict=function_inputs
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
        centroid_error = sym.references.contacts_centroid - 0.5 * (
            left_centroid + right_centroid
        )
        weighted_centroid_squared_error = (
            centroid_error.T
            @ cs.diag(sym.references.contacts_centroid_cost_weights)
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
        sym = self.ocp.symbolic_structure  # type: Variables

        # Unitary quaternion
        problem.add_expression_to_horizon(
            expression=cs.MX(
                cs.sumsqr(sym.system.kinematics.base.quaternion_xyzw) == 1
            ),
            apply_to_first_elements=False,
            name="unitary_quaternion",
        )

        # Consistency of com position with kinematics
        com_kinematics_fun = hp_rp.center_of_mass_position_from_kinematics(
            kindyn_object=self.kin_dyn_object, **function_inputs
        )
        com_kinematics = com_kinematics_fun(
            pb=sym.system.kinematics.base.position,
            qb=normalized_quaternion,
            s=sym.system.kinematics.joints.positions,
        )["com_position"]
        problem.add_expression_to_horizon(
            expression=cs.MX(sym.system.com == com_kinematics),
            apply_to_first_elements=False,
            name="com_kinematics_consistency",
        )

        # Consistency of centroidal momentum (angular part only) with kinematics
        centroidal_kinematics_fun = hp_rp.centroidal_momentum_from_kinematics(
            kindyn_object=self.kin_dyn_object, **function_inputs
        )
        centroidal_kinematics = centroidal_kinematics_fun(
            pb=sym.system.kinematics.base.position,
            qb=normalized_quaternion,
            s=sym.system.kinematics.joints.positions,
            pb_dot=sym.system.kinematics.base.linear_velocity,
            qb_dot=sym.system.kinematics.base.quaternion_velocity_xyzw,
            s_dot=sym.system.kinematics.joints.velocities,
        )["h_g"]
        problem.add_expression_to_horizon(
            expression=cs.MX(
                sym.system.centroidal_momentum[3:]
                == centroidal_kinematics[3:] / sym.mass
            ),
            apply_to_first_elements=True,
            name="centroidal_momentum_kinematics_consistency",
        )

        # Bounds on angular momentum
        problem.add_expression_to_horizon(
            expression=cs.Opti_bounded(
                -sym.maximum_angular_momentum,
                sym.system.centroidal_momentum[3:] * sym.mass,
                sym.maximum_angular_momentum,
            ),
            apply_to_first_elements=True,
            name="angular_momentum_bounds",
        )

        # Minimum com height
        com_height = height_fun(p=sym.system.com)["point_height"]
        problem.add_expression_to_horizon(
            expression=cs.MX(com_height >= sym.minimum_com_height),
            apply_to_first_elements=False,
            name="minimum_com_height",
        )

        # Minimum feet lateral distance
        left_frame = sym.system.contact_points.left[0].descriptor.foot_frame
        right_frame = sym.system.contact_points.right[0].descriptor.foot_frame
        relative_position_fun = hp_rp.frames_relative_position(
            kindyn_object=self.kin_dyn_object,
            reference_frame=right_frame,
            target_frame=left_frame,
            **function_inputs,
        )
        relative_position = relative_position_fun(
            s=sym.system.kinematics.joints.positions
        )["relative_position"]
        problem.add_expression_to_horizon(
            expression=cs.MX(relative_position[1] >= sym.minimum_feet_lateral_distance),
            apply_to_first_elements=False,
            name="minimum_feet_distance",
        )

        # Joint position bounds
        problem.add_expression_to_horizon(
            expression=cs.Opti_bounded(
                sym.minimum_joint_positions,
                sym.system.kinematics.joints.positions,
                sym.maximum_joint_positions,
            ),
            apply_to_first_elements=False,
            name="joint_position_bounds",
        )

        # Joint velocity bounds
        problem.add_expression_to_horizon(
            expression=cs.Opti_bounded(
                sym.minimum_joint_velocities,
                sym.system.kinematics.joints.velocities,
                sym.maximum_joint_velocities,
            ),
            apply_to_first_elements=True,
            name="joint_velocity_bounds",
        )

    def add_kinematics_regularization(
        self, function_inputs: dict, base_quaternion_normalized: cs.MX
    ):
        problem = self.ocp.problem
        sym = self.ocp.symbolic_structure  # type: Variables
        # Desired com velocity
        com_velocity_error = (
            sym.system.centroidal_momentum[:3] - sym.references.com_linear_velocity
        )
        com_velocity_weighted_error = (
            com_velocity_error.T
            @ cs.diag(cs.DM(self.settings.com_linear_velocity_cost_weights))
            @ com_velocity_error
        )
        problem.add_expression_to_horizon(
            expression=com_velocity_weighted_error,
            apply_to_first_elements=True,
            name="com_velocity_error",
            mode=hp.ExpressionType.minimize,
            scaling=self.settings.com_linear_velocity_cost_multiplier,
        )

        # Desired frame orientation
        rotation_error_kinematics_fun = hp_rp.rotation_error_from_kinematics(
            kindyn_object=self.kin_dyn_object,
            target_frame=self.settings.desired_frame_quaternion_cost_frame_name,
            **function_inputs,
        )
        rotation_error_kinematics = rotation_error_kinematics_fun(
            pb=sym.system.kinematics.base.position,
            qb=base_quaternion_normalized,
            s=sym.system.kinematics.joints.positions,
            qd=sym.references.desired_frame_quaternion_xyzw,
        )["rotation_error"]
        problem.add_expression_to_horizon(
            expression=cs.sumsqr(cs.trace(rotation_error_kinematics) - 3),
            apply_to_first_elements=False,
            name="frame_quaternion_error",
            mode=hp.ExpressionType.minimize,
            scaling=self.settings.desired_frame_quaternion_cost_multiplier,
        )

        # Desired base orientation
        quaternion_error_fun = hp_rp.quaternion_xyzw_error(**function_inputs)
        quaternion_error = quaternion_error_fun(
            q=sym.system.kinematics.base.quaternion_xyzw,
            qd=sym.references.base_quaternion_xyzw,
        )["quaternion_error"]
        problem.add_expression_to_horizon(
            expression=cs.sumsqr(quaternion_error),
            apply_to_first_elements=False,
            name="base_quaternion_error",
            mode=hp.ExpressionType.minimize,
            scaling=self.settings.base_quaternion_cost_multiplier,
        )

        # Desired base angular velocity
        problem.add_expression_to_horizon(
            expression=cs.sumsqr(
                sym.system.kinematics.base.quaternion_velocity_xyzw
                - sym.references.base_quaternion_xyzw_velocity
            ),
            apply_to_first_elements=True,
            name="base_quaternion_velocity_error",
            mode=hp.ExpressionType.minimize,
            scaling=self.settings.base_quaternion_velocity_cost_multiplier,
        )

        # Desired joint positions
        joint_positions_error = (
            sym.system.kinematics.joints.positions - sym.references.joint_regularization
        )
        joint_velocity_error = (
            sym.system.kinematics.joints.velocities
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
        sym = self.ocp.symbolic_structure  # type: Variables
        default_integrator = self.settings.integrator

        # dot(pb) = pb_dot (base position dynamics)
        problem.add_dynamics(
            hp.dot(sym.system.kinematics.base.position)
            == sym.system.kinematics.base.linear_velocity,
            x0=problem.initial(sym.initial_state.kinematics.base.position),
            dt=sym.dt,
            integrator=default_integrator,
            name="base_position_dynamics",
        )

        # dot(q) = q_dot (base quaternion dynamics)
        problem.add_dynamics(
            hp.dot(sym.system.kinematics.base.quaternion_xyzw)
            == sym.system.kinematics.base.quaternion_velocity_xyzw,
            x0=problem.initial(sym.initial_state.kinematics.base.quaternion_xyzw),
            dt=sym.dt,
            integrator=default_integrator,
            name="base_quaternion_dynamics",
        )

        # dot(s) = s_dot (joint position dynamics)
        problem.add_dynamics(
            hp.dot(sym.system.kinematics.joints.positions)
            == sym.system.kinematics.joints.velocities,
            x0=problem.initial(sym.initial_state.kinematics.joints.positions),
            dt=sym.dt,
            integrator=default_integrator,
            name="joint_position_dynamics",
        )

        # dot(com) = h_g[:3] (center of mass dynamics, regularized by the mass)
        problem.add_dynamics(
            hp.dot(sym.system.com) == sym.system.centroidal_momentum[:3],  # noqa
            x0=problem.initial(sym.initial_state.com),  # noqa
            dt=sym.dt,
            integrator=default_integrator,
            name="com_dynamics",
        )

        # dot(h) = sum_i (p_i x f_i) + g (centroidal momentum dynamics,mass regularized)
        function_inputs["point_position_names"] = [
            point.p.name() for point in all_contact_points
        ]
        function_inputs["point_force_names"] = [
            point.f.name() for point in all_contact_points
        ]
        centroidal_dynamics = hp_rp.centroidal_dynamics_with_point_forces(
            number_of_points=len(function_inputs["point_position_names"]),
            assume_unitary_mass=True,
            **function_inputs,
        )
        problem.add_dynamics(
            hp.dot(sym.system.centroidal_momentum) == centroidal_dynamics,  # noqa
            x0=problem.initial(sym.initial_state.centroidal_momentum),  # noqa
            dt=sym.dt,
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
        sym = self.ocp.symbolic_structure  # type: Variables

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
            pb=sym.system.kinematics.base.position,
            qb=normalized_quaternion,
            s=sym.system.kinematics.joints.positions,
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
        sym = self.ocp.symbolic_structure  # type: Variables

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
                -sym.maximum_force_derivative,
                point.f_dot * sym.mass,
                sym.maximum_force_derivative,
            ),
            apply_to_first_elements=True,
            name=point.f_dot.name() + "_bounds",  # noqa
        )

    def add_point_dynamics(
        self, point: ExtendedContactPoint, initial_state: hp_rp.ContactPointState
    ) -> None:
        default_integrator = self.settings.integrator
        problem = self.ocp.problem
        sym = self.ocp.symbolic_structure  # type: Variables

        # dot(f) = f_dot
        problem.add_dynamics(
            hp.dot(point.f) == point.f_dot,
            x0=problem.initial(initial_state.f),
            dt=sym.dt,
            integrator=default_integrator,
            name=point.f.name() + "_dynamics",
        )

        # dot(p) = v
        problem.add_dynamics(
            hp.dot(point.p) == point.v,
            x0=problem.initial(initial_state.p),
            dt=sym.dt,
            integrator=default_integrator,
            name=point.p.name() + "_dynamics",
        )

    def add_foot_regularization(
        self,
        points: list[ExtendedContactPoint],
        descriptors: list[hp_rp.ContactPointDescriptor],
        references: FootReferences,
        function_inputs: dict,
        foot_name: str,
    ) -> None:
        problem = self.ocp.problem

        # Force ratio regularization
        sum_of_forces = cs.MX.zeros(3, 1)
        for point in points:
            sum_of_forces += point.f

        for i, point in enumerate(points):
            alpha = references.points[i].desired_force_ratio
            force_error = point.f - alpha * sum_of_forces

            problem.add_expression_to_horizon(
                expression=cs.sumsqr(force_error),
                apply_to_first_elements=False,
                name=point.f.name() + "_regularization",
                mode=hp.ExpressionType.minimize,
                scaling=self.settings.force_regularization_cost_multiplier,
            )

        # Foot yaw task
        centroid_position = np.zeros((3, 1))
        for descriptor in descriptors:
            position_in_foot_frame = descriptor.position_in_foot_frame.reshape((3, 1))
            centroid_position += position_in_foot_frame
        centroid_position /= len(descriptors)

        bottom_right_index = None
        top_right_index = None
        top_left_index = None

        # The values below are useful to get the outermost points of the foot
        bottom_right_value = 0
        top_right_value = 0
        top_left_value = 0

        for i, descriptor in enumerate(descriptors):
            relative_position = (
                descriptor.position_in_foot_frame.reshape((3, 1)) - centroid_position
            )

            if (
                relative_position[1] < 0
                and relative_position[0] < 0
                and (
                    bottom_right_index is None
                    or relative_position[0] * relative_position[1] > bottom_right_value
                )
            ):
                bottom_right_value = relative_position[0] * relative_position[1]
                bottom_right_index = i
            elif relative_position[1] < 0 < relative_position[0] and (
                top_right_index is None
                or relative_position[0] * relative_position[1] > top_right_value
            ):
                top_right_value = relative_position[0] * relative_position[1]
                top_right_index = i
            elif (
                relative_position[1] > 0
                and relative_position[0] > 0
                and (
                    top_left_index is None
                    or relative_position[0] * relative_position[1] > top_left_value
                )
            ):
                top_left_value = relative_position[0] * relative_position[1]
                top_left_index = i

        assert bottom_right_index is not None
        assert top_right_index is not None
        assert top_left_index is not None
        assert (
            bottom_right_index != top_right_index
            and top_right_index != top_left_index
            and top_left_index != bottom_right_index
        )

        yaw_alignment_fun = hp_rp.contact_points_yaw_alignment_error(**function_inputs)
        yaw_alignment_forward = yaw_alignment_fun(
            p_0=points[bottom_right_index].p,
            p_1=points[top_right_index].p,
            yd=references.yaw,
        )["yaw_alignment_error"]

        yaw_alignment_sideways = yaw_alignment_fun(
            p_0=points[top_right_index].p,
            p_1=points[top_left_index].p,
            yd=references.yaw + np.pi / 2,
        )["yaw_alignment_error"]

        yaw_error = 0.5 * (
            cs.sumsqr(yaw_alignment_forward) + cs.sumsqr(yaw_alignment_sideways)
        )

        problem.add_expression_to_horizon(
            expression=yaw_error,
            apply_to_first_elements=False,
            name=foot_name + "_yaw_regularization",
            mode=hp.ExpressionType.minimize,
            scaling=self.settings.foot_yaw_regularization_cost_multiplier,
        )

    def add_contact_point_regularization(
        self,
        point: ExtendedContactPoint,
        desired_swing_height: hp.StorageType,
        function_inputs: dict,
    ) -> None:
        problem = self.ocp.problem

        # Swing height regularization
        swing_heuristic_fun = hp_rp.swing_height_heuristic(
            self.settings.terrain, **function_inputs
        )
        heuristic = swing_heuristic_fun(p=point.p, v=point.v, hd=desired_swing_height)[
            "heuristic"
        ]

        problem.add_expression_to_horizon(
            expression=heuristic,
            apply_to_first_elements=False,
            name=point.p.name() + "_swing_height_regularization",
            mode=hp.ExpressionType.minimize,
            scaling=self.settings.swing_foot_height_cost_multiplier,
        )

        # Contact velocity control regularization
        problem.add_expression_to_horizon(
            expression=cs.sumsqr(point.u_v),
            apply_to_first_elements=False,
            name=point.u_v.name() + "_regularization",  # noqa
            mode=hp.ExpressionType.minimize,
            scaling=self.settings.contact_velocity_control_cost_multiplier,
        )

        # Contact force control regularization
        problem.add_expression_to_horizon(
            expression=cs.sumsqr(point.f_dot),
            apply_to_first_elements=False,
            name=point.f_dot.name() + "_regularization",  # noqa
            mode=hp.ExpressionType.minimize,
            scaling=self.settings.contact_force_control_cost_multiplier,
        )

    def set_initial_guess(self, initial_guess: Variables) -> None:
        self.ocp.problem.set_initial_guess(initial_guess)

    def get_initial_guess(self) -> Variables:
        return self.ocp.problem.get_initial_guess()

    def set_references(self, references: References | list[References]) -> None:
        guess = self.get_initial_guess()

        assert isinstance(guess.references, list)
        assert not isinstance(references, list) or len(references) == len(
            guess.references
        )
        for i in range(len(guess.references)):
            guess.references[i] = (
                references[i] if isinstance(references, list) else references
            )
        self.ocp.problem.set_initial_guess(guess)

    def set_initial_state(self, initial_state: ExtendedHumanoidState) -> None:
        guess = self.get_initial_guess()
        guess.initial_state = initial_state
        guess.initial_state.centroidal_momentum /= self.numeric_mass
        for point in (
            guess.initial_state.contact_points.left
            + guess.initial_state.contact_points.right
        ):
            point.f /= self.numeric_mass
        self.ocp.problem.set_initial_guess(guess)

    def solve(self) -> hp.Output[Variables]:
        output = self.ocp.problem.solve()

        values = output.values

        for s in values.system:
            for point in s.contact_points.left + s.contact_points.right:
                point.f *= values.mass

        return output