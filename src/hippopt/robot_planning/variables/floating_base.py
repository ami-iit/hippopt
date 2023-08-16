import dataclasses

import numpy as np

from hippopt import (
    CompositeType,
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
    def __post_init__(self):
        FreeFloatingObjectState.__post_init__(self)
        FreeFloatingObjectStateDerivative.__post_init__(self)


@dataclasses.dataclass
class KinematicTreeState(OptimizationObject):
    positions: StorageType = default_storage_field(Variable)

    number_of_joints_state: dataclasses.InitVar[int] = dataclasses.field(default=0)

    def __post_init__(self, number_of_joints_state: int):
        self.positions = np.zeros(number_of_joints_state)


@dataclasses.dataclass
class KinematicTreeStateDerivative(OptimizationObject):
    velocities: StorageType = default_storage_field(Variable)

    number_of_joints_derivative: dataclasses.InitVar[int] = dataclasses.field(
        default=None
    )

    def __post_init__(self, number_of_joints_derivative: int):
        self.velocities = np.zeros(number_of_joints_derivative)


@dataclasses.dataclass
class KinematicTree(KinematicTreeState, KinematicTreeStateDerivative):
    def __post_init__(
        self,
        number_of_joints_derivative: int = None,
        number_of_joints_state: int = None,
    ):
        if number_of_joints_derivative is None and number_of_joints_state is None:
            number_of_joints_state = 0
            number_of_joints_derivative = 0

        number_of_joints_state = (
            number_of_joints_derivative
            if number_of_joints_state is None
            else number_of_joints_state
        )
        number_of_joints_derivative = (
            number_of_joints_state
            if number_of_joints_derivative is None
            else number_of_joints_derivative
        )
        KinematicTreeState.__post_init__(
            self, number_of_joints_state=number_of_joints_state
        )
        KinematicTreeStateDerivative.__post_init__(
            self, number_of_joints_derivative=number_of_joints_derivative
        )


@dataclasses.dataclass
class FloatingBaseSystemState(OptimizationObject):
    base: CompositeType[FreeFloatingObjectState] = default_composite_field(
        factory=FreeFloatingObjectState
    )
    joints: CompositeType[KinematicTreeState] = default_composite_field(
        factory=KinematicTreeState
    )

    number_of_joints_state: dataclasses.InitVar[int] = dataclasses.field(default=None)

    def __post_init__(self, number_of_joints_state: int):
        self.joints = KinematicTreeState(number_of_joints_state=number_of_joints_state)


@dataclasses.dataclass
class FloatingBaseSystemStateDerivative(OptimizationObject):
    base: CompositeType[FreeFloatingObjectStateDerivative] = default_composite_field(
        factory=FreeFloatingObjectStateDerivative
    )
    joints: CompositeType[KinematicTreeStateDerivative] = default_composite_field(
        factory=KinematicTreeStateDerivative
    )

    number_of_joints_derivative: dataclasses.InitVar[int] = dataclasses.field(
        default=None
    )

    def __post_init__(self, number_of_joints_derivative: int):
        self.joints = KinematicTreeStateDerivative(
            number_of_joints_derivative=number_of_joints_derivative
        )


@dataclasses.dataclass
class FloatingBaseSystem(OptimizationObject):
    base: CompositeType[FreeFloatingObject] = default_composite_field(
        factory=FreeFloatingObject
    )
    joints: CompositeType[KinematicTree] = default_composite_field(
        factory=KinematicTree
    )

    number_of_joints: dataclasses.InitVar[int] = dataclasses.field(default=None)

    def __post_init__(self, number_of_joints: int):
        self.joints = KinematicTree(number_of_joints_state=number_of_joints)
