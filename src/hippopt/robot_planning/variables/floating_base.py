import dataclasses

import numpy as np

from hippopt import (
    OptimizationObject,
    StorageType,
    Variable,
    default_composite_field,
    default_storage_field,
)


@dataclasses.dataclass
class FreeFloatingObjectState(OptimizationObject):
    position: StorageType = default_storage_field(Variable)
    quaternion_xyzw: StorageType = default_storage_field(Variable)

    def __post_init__(self):
        self.position = np.zeros(3)
        self.quaternion_xyzw = np.zeros(4)
        self.quaternion_xyzw[3] = 1.0


@dataclasses.dataclass
class FreeFloatingObjectStateDerivative(OptimizationObject):
    linear_velocity: StorageType = default_storage_field(Variable)
    quaternion_velocity_xyzw: StorageType = default_storage_field(Variable)

    def __post_init__(self):
        self.linear_velocity = np.zeros(3)
        self.quaternion_velocity_xyzw = np.zeros(4)


@dataclasses.dataclass
class FreeFloatingObject(FreeFloatingObjectState, FreeFloatingObjectStateDerivative):
    pass


@dataclasses.dataclass
class KinematicTreeState(OptimizationObject):
    positions: StorageType = default_storage_field(Variable)

    number_of_joints: dataclasses.InitVar[int] = dataclasses.field(default=None)

    def __post_init__(self, number_of_joints: int):
        self.positions = np.zeros(number_of_joints)


@dataclasses.dataclass
class KinematicTreeStateDerivative(OptimizationObject):
    velocities: StorageType = default_storage_field(Variable)

    number_of_joints: dataclasses.InitVar[int] = dataclasses.field(default=None)

    def __post_init__(self, number_of_joints: int):
        self.velocities = np.zeros(number_of_joints)


@dataclasses.dataclass
class KinematicTree(KinematicTreeState, KinematicTreeStateDerivative):
    pass


@dataclasses.dataclass
class FloatingBaseSystemState(OptimizationObject):
    base: FreeFloatingObjectState = default_composite_field(
        factory=FreeFloatingObjectState
    )
    joints: KinematicTreeState = default_composite_field()

    number_of_joints: dataclasses.InitVar[int] = dataclasses.field(default=None)

    def __post_init__(self, number_of_joints: int):
        self.joints = KinematicTreeState(number_of_joints=number_of_joints)


@dataclasses.dataclass
class FloatingBaseSystemStateDerivative(OptimizationObject):
    base: FreeFloatingObjectStateDerivative = default_composite_field(
        factory=FreeFloatingObjectStateDerivative
    )
    joints: KinematicTreeStateDerivative = default_composite_field()

    number_of_joints: dataclasses.InitVar[int] = dataclasses.field(default=None)

    def __post_init__(self, number_of_joints: int):
        self.joints = KinematicTreeStateDerivative(number_of_joints=number_of_joints)


@dataclasses.dataclass
class FloatingBaseSystem(OptimizationObject):
    base: FreeFloatingObject = default_composite_field(factory=FreeFloatingObject)
    joints: KinematicTree = default_composite_field()

    number_of_joints: dataclasses.InitVar[int] = dataclasses.field(default=None)

    def __post_init__(self, number_of_joints: int):
        self.joints = KinematicTree(number_of_joints=number_of_joints)
