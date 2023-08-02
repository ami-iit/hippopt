import dataclasses

import adam.casadi
import numpy as np

import hippopt as hp
import hippopt.robot_planning as hp_rp


@dataclasses.dataclass
class FeetContactPoints(hp.OptimizationObject):
    left: list[hp_rp.ContactPoint] = hp.default_storage_field(hp.Variable)
    right: list[hp_rp.ContactPoint] = hp.default_storage_field(hp.Variable)


@dataclasses.dataclass
class FeetContactPointDescriptors:
    left: list[hp_rp.ContactPointDescriptor] = dataclasses.field(default_factory=list)
    right: list[hp_rp.ContactPointDescriptor] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Variables(hp.OptimizationObject):
    contact_points: FeetContactPoints = dataclasses.field(
        default_factory=FeetContactPoints
    )
    com: hp.StorageType = hp.default_storage_field(hp.Variable)
    centroidal_momentum: hp.StorageType = hp.default_storage_field(hp.Variable)
    base_position: hp.StorageType = hp.default_storage_field(hp.Variable)
    base_linear_velocity: hp.StorageType = hp.default_storage_field(hp.Variable)
    base_quaternion_xyzw: hp.StorageType = hp.default_storage_field(hp.Variable)
    base_quaternion_velocity_xyzw: hp.StorageType = hp.default_storage_field(
        hp.Variable
    )
    joint_positions: hp.StorageType = hp.default_storage_field(hp.Variable)
    joint_velocities: hp.StorageType = hp.default_storage_field(hp.Variable)

    dt: hp.StorageType = hp.default_storage_field(hp.Parameter)
    gravity: hp.StorageType = hp.default_storage_field(hp.Parameter)

    number_of_joints: dataclasses.InitVar[int] = dataclasses.field(default=None)
    contact_point_descriptors: dataclasses.InitVar[
        FeetContactPointDescriptors
    ] = dataclasses.field(default=None)

    def __post_init__(
        self,
        number_of_joints: int,
        contact_point_descriptors: FeetContactPointDescriptors,
    ):
        self.contact_points.left = [
            hp_rp.ContactPoint(descriptor=point)
            for point in contact_point_descriptors.left
        ]
        self.contact_points.right = [
            hp_rp.ContactPoint(descriptor=point)
            for point in contact_point_descriptors.right
        ]

        self.com = np.zeros(3)
        self.centroidal_momentum = np.zeros(3)
        self.base_position = np.zeros(3)
        self.base_linear_velocity = np.zeros(3)
        self.base_quaternion_xyzw = np.zeros(4)
        self.base_quaternion_velocity_xyzw = np.zeros(4)
        self.joint_positions = np.zeros(number_of_joints)
        self.joint_velocities = np.zeros(number_of_joints)
        self.dt = 0.1 * np.ones(1)
        self.gravity = np.zeros(3)
        self.gravity[2] = -9.81


@dataclasses.dataclass
class Settings:
    robot_urdf: str = dataclasses.field(default=None)
    joints_name_list: list[str] = dataclasses.field(default=None)
    contact_points: list[hp_rp.ContactPointDescriptor] = dataclasses.field(default=None)
    root_link: str = dataclasses.field(default=None)
    gravity: np.array = dataclasses.field(default=None)
    casadi_function_options: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self.casadi_function_options = (
            self.casadi_function_options
            if isinstance(self.casadi_function_options, dict)
            else {}
        )
        self.root_link = "root_link"
        self.gravity = np.array([0.0, 0.0, -9.80665, 0.0, 0.0, 0.0])

    def is_valid(self):
        return (
            self.robot_urdf is not None
            and self.joints_name_list is not None
            and self.contact_points is not None
        )


class HumanoidWalkingFlatGround:
    def __init__(self, settings: Settings) -> None:
        if not settings.is_valid():
            raise ValueError("Settings are not valid")
        self.settings = settings
        self.kin_dyn_object = adam.casadi.KinDynComputations(
            urdfstring=self.settings.robot_urdf,
            joints_name_list=self.settings.joints_name_list,
            root_link=self.settings.root_link,
            gravity=self.settings.gravity,
            f_opts=self.settings.casadi_function_options,
        )
