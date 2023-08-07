import dataclasses

import numpy as np

from hippopt import (
    OptimizationObject,
    Parameter,
    StorageType,
    Variable,
    default_composite_field,
    default_storage_field,
)


@dataclasses.dataclass
class FreeFloatingObject(OptimizationObject):
    position: StorageType = default_storage_field(Variable)
    linear_velocity: StorageType = default_storage_field(Variable)
    quaternion_xyzw: StorageType = default_storage_field(Variable)
    quaternion_velocity_xyzw: StorageType = default_storage_field(Variable)

    # Initial conditions
    initial_position: StorageType = default_storage_field(Parameter)
    initial_quaternion_xyzw: StorageType = default_storage_field(Parameter)

    def __post_init__(self):
        self.position = np.zeros(3)
        self.linear_velocity = np.zeros(3)
        self.quaternion_xyzw = np.zeros(4)
        self.quaternion_xyzw[3] = 1.0
        self.quaternion_velocity_xyzw = np.zeros(4)

        self.initial_position = np.zeros(3)
        self.initial_quaternion_xyzw = np.zeros(4)
        self.initial_quaternion_xyzw[3] = 1.0


@dataclasses.dataclass
class KinematicTree(OptimizationObject):
    positions: StorageType = default_storage_field(Variable)
    velocities: StorageType = default_storage_field(Variable)

    # Initial conditions
    initial_positions: StorageType = default_storage_field(Parameter)

    number_of_joints: dataclasses.InitVar[int] = dataclasses.field(default=None)

    def __post_init__(self, number_of_joints: int):
        self.positions = np.zeros(number_of_joints)
        self.velocities = np.zeros(number_of_joints)

        self.initial_positions = np.zeros(number_of_joints)


@dataclasses.dataclass
class FloatingBaseSystem(OptimizationObject):
    base: FreeFloatingObject = default_composite_field(factory=FreeFloatingObject)
    joints: KinematicTree = default_composite_field()

    number_of_joints: dataclasses.InitVar[int] = dataclasses.field(default=None)

    def __post_init__(self, number_of_joints: int):
        self.joints = KinematicTree(number_of_joints=number_of_joints)
