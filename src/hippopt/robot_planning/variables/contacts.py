import copy
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
        if input_foot_frame is not None:
            self.foot_frame = input_foot_frame
        if input_position_in_foot_frame is not None:
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
        self.p = np.zeros(3) if self.p is None else self.p
        self.f = np.zeros(3) if self.f is None else self.f
        if input_descriptor is not None:
            self.descriptor = copy.deepcopy(input_descriptor)


@dataclasses.dataclass
class ContactPointStateDerivative(OptimizationObject):
    v: StorageType = default_storage_field(OverridableVariable)
    f_dot: StorageType = default_storage_field(OverridableVariable)

    def __post_init__(self) -> None:
        self.v = np.zeros(3) if self.v is None else self.v
        self.f_dot = np.zeros(3) if self.f_dot is None else self.f_dot


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
    transform: liecasadi.SE3 = dataclasses.field(default=None)
    mid_swing_transform: liecasadi.SE3 | None = dataclasses.field(default=None)
    force: np.ndarray = dataclasses.field(default=None)
    activation_time: float | None = dataclasses.field(default=None)
    deactivation_time: float | None = dataclasses.field(default=None)

    def __post_init__(self) -> None:
        self.transform = (
            liecasadi.SE3.from_translation_and_rotation(
                cs.DM.zeros(3), liecasadi.SO3.Identity()
            )
            if self.transform is None
            else self.transform
        )
        if self.force is None:
            self.force = np.zeros(3)
            self.force[2] = 100


@dataclasses.dataclass
class FeetContactPhasesDescriptor:
    left: list[FootContactPhaseDescriptor] = dataclasses.field(default_factory=list)
    right: list[FootContactPhaseDescriptor] = dataclasses.field(default_factory=list)
