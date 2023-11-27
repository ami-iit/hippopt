import dataclasses
from typing import TypeVar

import casadi as cs
import liecasadi
import numpy as np

from hippopt import (
    CompositeType,
    OptimizationObject,
    OverridableVariable,
    Parameter,
    StorageType,
    default_composite_field,
    default_storage_field,
)


@dataclasses.dataclass
class ContactPointDescriptor(OptimizationObject):
    position_in_foot_frame: StorageType = default_storage_field(Parameter)
    foot_frame: str = dataclasses.field(default=None)

    input_foot_frame: dataclasses.InitVar[str] = dataclasses.field(default=None)
    input_position_in_foot_frame: dataclasses.InitVar[np.ndarray] = dataclasses.field(
        default=None
    )

    def __post_init__(
        self, input_foot_frame: str, input_position_in_foot_frame: np.ndarray
    ) -> None:
        self.foot_frame = input_foot_frame
        self.position_in_foot_frame = input_position_in_foot_frame

    @staticmethod
    def rectangular_foot(
        foot_frame: str,
        x_length: float,
        y_length: float,
        top_left_point_position: np.array,
    ):
        return [
            ContactPointDescriptor(
                input_foot_frame=foot_frame,
                input_position_in_foot_frame=top_left_point_position,
            ),
            ContactPointDescriptor(
                input_foot_frame=foot_frame,
                input_position_in_foot_frame=top_left_point_position
                + np.array([-x_length, 0.0, 0.0]),
            ),
            ContactPointDescriptor(
                input_foot_frame=foot_frame,
                input_position_in_foot_frame=top_left_point_position
                + np.array([-x_length, -y_length, 0.0]),
            ),
            ContactPointDescriptor(
                input_foot_frame=foot_frame,
                input_position_in_foot_frame=top_left_point_position
                + np.array([0.0, -y_length, 0.0]),
            ),
        ]


@dataclasses.dataclass
class ContactPointState(OptimizationObject):
    p: StorageType = default_storage_field(OverridableVariable)
    f: StorageType = default_storage_field(OverridableVariable)

    descriptor: CompositeType[ContactPointDescriptor] = default_composite_field(
        factory=ContactPointDescriptor, time_varying=False
    )

    input_descriptor: dataclasses.InitVar[ContactPointDescriptor] = dataclasses.field(
        default=None
    )

    def __post_init__(self, input_descriptor: ContactPointDescriptor) -> None:
        self.p = np.zeros(3)
        self.f = np.zeros(3)

        self.descriptor = input_descriptor


@dataclasses.dataclass
class ContactPointStateDerivative(OptimizationObject):
    v: StorageType = default_storage_field(OverridableVariable)
    f_dot: StorageType = default_storage_field(OverridableVariable)

    def __post_init__(self) -> None:
        self.v = np.zeros(3)
        self.f_dot = np.zeros(3)


TFootContactState = TypeVar("TFootContactState", bound="FootContactState")


@dataclasses.dataclass
class FootContactState(list[ContactPointState], OptimizationObject):
    def set_from_parent_frame_transform(self, transform: liecasadi.SE3):
        for contact_point in self:
            contact_point.p = transform.translation() + transform.rotation().act(
                contact_point.descriptor.position_in_foot_frame
            )

    @staticmethod
    def from_list(input_list: list[ContactPointState]) -> TFootContactState:
        output = FootContactState()
        for contact_point in input_list:
            output.append(contact_point)
        return output

    @staticmethod
    def from_parent_frame_transform(
        descriptor: list[ContactPointDescriptor], transform: liecasadi.SE3
    ) -> TFootContactState:
        foot_contact_state = FootContactState()
        for contact_point_descriptor in descriptor:
            foot_contact_state.append(
                ContactPointState(input_descriptor=contact_point_descriptor)
            )

        foot_contact_state.set_from_parent_frame_transform(transform)
        return foot_contact_state


@dataclasses.dataclass
class FeetContactPointDescriptors:
    left: list[ContactPointDescriptor] = dataclasses.field(default_factory=list)
    right: list[ContactPointDescriptor] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class FeetContactPoints(OptimizationObject):
    left: FootContactState = default_composite_field(factory=FootContactState)
    right: FootContactState = default_composite_field(factory=FootContactState)


@dataclasses.dataclass
class FootContactPhaseDescriptor:
    transform: liecasadi.SE3 = dataclasses.field(default_factory=liecasadi.SE3)
    mid_swing_transform: liecasadi.SE3 = dataclasses.field(
        default_factory=liecasadi.SE3
    )
    force: float = dataclasses.field(default=100.0)
    activation_time: float = dataclasses.field(default=None)
    deactivation_time: float = dataclasses.field(default=None)

    def __post_init__(self) -> None:
        self.transform = liecasadi.SE3.from_translation_and_rotation(
            cs.DM.zeros(3), liecasadi.SO3.Identity()
        )
        self.mid_swing_transform = liecasadi.SE3.from_translation_and_rotation(
            cs.DM.zeros(3), liecasadi.SO3.Identity()
        )
