import dataclasses

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


@dataclasses.dataclass
class FeetContactPointDescriptors:
    left: list[ContactPointDescriptor] = dataclasses.field(default_factory=list)
    right: list[ContactPointDescriptor] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class FeetContactPoints(OptimizationObject):
    left: list[ContactPointState] = default_composite_field(factory=list)
    right: list[ContactPointState] = default_composite_field(factory=list)
