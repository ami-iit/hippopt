import copy
import dataclasses
from typing import TypeVar

import adam.casadi
import adam.parametric.casadi
import numpy as np

import hippopt as hp
from hippopt import robot_planning as hp_rp
from hippopt.turnkey_planners.humanoid_kinodynamic.settings import Settings


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


TExtendedContactPoint = TypeVar("TExtendedContactPoint", bound="ExtendedContactPoint")


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

    @staticmethod
    def from_contact_point_state(
        input_state: hp_rp.ContactPointState,
    ) -> TExtendedContactPoint:
        output = ExtendedContactPoint(input_descriptor=input_state.descriptor)
        output.p = input_state.p
        output.f = input_state.f
        output.u_v = None
        output.f_dot = None
        output.v = None
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

    def from_feet_contact_points(self, input_points: hp_rp.FeetContactPoints) -> None:
        self.left = [
            ExtendedContactPoint.from_contact_point_state(point)
            for point in input_points.left
        ]
        self.right = [
            ExtendedContactPoint.from_contact_point_state(point)
            for point in input_points.right
        ]


TExtendedHumanoid = TypeVar("TExtendedHumanoid", bound="ExtendedHumanoid")


@dataclasses.dataclass
class ExtendedHumanoid(hp.OptimizationObject):
    contact_points: hp.CompositeType[FeetContactPointsExtended] = (
        hp.default_composite_field(factory=FeetContactPointsExtended)
    )

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

    @staticmethod
    def from_humanoid_state(input_state: hp_rp.HumanoidState) -> TExtendedHumanoid:
        output = ExtendedHumanoid()
        output.contact_points.from_feet_contact_points(input_state.contact_points)
        output.kinematics = hp_rp.FloatingBaseSystem.from_floating_base_system_state(
            input_state.kinematics
        )
        output.com = input_state.com
        output.centroidal_momentum = None
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

    parametric_link_length_multipliers: hp.StorageType = hp.default_storage_field(
        hp.Parameter
    )
    parametric_link_densities: hp.StorageType = hp.default_storage_field(hp.Parameter)

    initial_state: hp.CompositeType[ExtendedHumanoidState] = hp.default_composite_field(
        cls=hp.Parameter, factory=ExtendedHumanoidState, time_varying=False
    )

    final_state: hp.CompositeType[hp_rp.HumanoidState] = hp.default_composite_field(
        cls=hp.Parameter, factory=hp_rp.HumanoidState, time_varying=False
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
        | adam.parametric.casadi.KinDynComputationsParametric
    ] = dataclasses.field(default=None)

    def __post_init__(
        self,
        settings: Settings,
        kin_dyn_object: (
            adam.casadi.KinDynComputations
            | adam.parametric.casadi.KinDynComputationsParametric
        ),
    ) -> None:
        self.system = ExtendedHumanoid(
            contact_point_descriptors=settings.contact_points,
            number_of_joints=kin_dyn_object.NDoF,
        )

        self.initial_state = ExtendedHumanoidState(
            contact_point_descriptors=settings.contact_points,
            number_of_joints=kin_dyn_object.NDoF,
        )

        self.final_state = hp_rp.HumanoidState(
            contact_point_descriptors=settings.contact_points,
            number_of_joints=kin_dyn_object.NDoF,
        )

        self.dt = settings.time_step
        self.gravity = kin_dyn_object.g

        self.parametric_link_length_multipliers = (
            np.ones(len(settings.parametric_link_names))
            if settings.parametric_link_names is not None
            else np.array([])
        )
        self.parametric_link_densities = (
            copy.deepcopy(settings.initial_densities)
            if settings.initial_densities is not None
            else np.array([])
        )

        if isinstance(
            kin_dyn_object, adam.parametric.casadi.KinDynComputationsParametric
        ):
            total_mass_fun = kin_dyn_object.get_total_mass()
            self.mass = float(
                total_mass_fun(
                    self.parametric_link_length_multipliers,
                    self.parametric_link_densities,
                ).full()
            )
        else:
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
