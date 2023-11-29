import dataclasses
from typing import TypeVar

import numpy as np

from hippopt import (
    CompositeType,
    OptimizationObject,
    OverridableVariable,
    StorageType,
    default_composite_field,
    default_storage_field,
)


@dataclasses.dataclass
class FreeFloatingObjectState(OptimizationObject):
    position: StorageType = default_storage_field(OverridableVariable)
    quaternion_xyzw: StorageType = default_storage_field(OverridableVariable)

    def __post_init__(self):
        if self.position is None:
            self.position = np.zeros(3)
        if self.quaternion_xyzw is None:
            self.quaternion_xyzw = np.zeros(4)
            self.quaternion_xyzw[3] = 1.0


@dataclasses.dataclass
class FreeFloatingObjectStateDerivative(OptimizationObject):
    linear_velocity: StorageType = default_storage_field(OverridableVariable)
    quaternion_velocity_xyzw: StorageType = default_storage_field(OverridableVariable)

    def __post_init__(self):
        if self.linear_velocity is None:
            self.linear_velocity = np.zeros(3)

        if self.quaternion_velocity_xyzw is None:
            self.quaternion_velocity_xyzw = np.zeros(4)


@dataclasses.dataclass
class FreeFloatingObject(FreeFloatingObjectState, FreeFloatingObjectStateDerivative):
    def __post_init__(self):
        FreeFloatingObjectState.__post_init__(self)
        FreeFloatingObjectStateDerivative.__post_init__(self)


@dataclasses.dataclass
class KinematicTreeState(OptimizationObject):
    positions: StorageType = default_storage_field(OverridableVariable)

    number_of_joints_state: dataclasses.InitVar[int] = dataclasses.field(default=0)

    def __post_init__(self, number_of_joints_state: int):
        if number_of_joints_state is not None:
            self.positions = np.zeros(number_of_joints_state)


@dataclasses.dataclass
class KinematicTreeStateDerivative(OptimizationObject):
    velocities: StorageType = default_storage_field(OverridableVariable)

    number_of_joints_derivative: dataclasses.InitVar[int] = dataclasses.field(
        default=None
    )

    def __post_init__(self, number_of_joints_derivative: int):
        if number_of_joints_derivative is not None:
            self.velocities = np.zeros(number_of_joints_derivative)


@dataclasses.dataclass
class KinematicTree(KinematicTreeState, KinematicTreeStateDerivative):
    def __post_init__(
        self,
        number_of_joints_derivative: int = None,
        number_of_joints_state: int = None,
    ):
        if (
            number_of_joints_derivative is not None
            or number_of_joints_state is not None
        ):
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
        if number_of_joints_state is not None:
            self.joints = KinematicTreeState(
                number_of_joints_state=number_of_joints_state
            )


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
        if number_of_joints_derivative is not None:
            self.joints = KinematicTreeStateDerivative(
                number_of_joints_derivative=number_of_joints_derivative
            )


TFloatingBaseSystem = TypeVar("TFloatingBaseSystem", bound="FloatingBaseSystem")


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
        if number_of_joints is not None:
            self.joints = KinematicTree(number_of_joints_state=number_of_joints)

    def to_floating_base_system_state(self):
        output = FloatingBaseSystemState()
        output.base.position = self.base.position
        output.base.quaternion_xyzw = self.base.quaternion_xyzw
        output.joints.positions = self.joints.positions
        return output

    @staticmethod
    def from_floating_base_system_state(
        state: FloatingBaseSystemState,
    ) -> TFloatingBaseSystem:
        output = FloatingBaseSystem(number_of_joints=len(state.joints.positions))
        output.base.position = state.base.position
        output.base.quaternion_xyzw = state.base.quaternion_xyzw
        output.base.linear_velocity = None
        output.base.quaternion_velocity_xyzw = None
        output.joints.positions = state.joints.positions
        output.joints.velocities = None
        return output
