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


@dataclasses.dataclass
class ContactPoint(OptimizationObject):
    p: StorageType = default_storage_field(Variable)
    v: StorageType = default_storage_field(Variable)
    f: StorageType = default_storage_field(Variable)
    f_dot: StorageType = default_storage_field(Variable)

    # Initial conditions
    p0: StorageType = default_storage_field(Parameter)
    v0: StorageType = default_storage_field(Parameter)
    f0: StorageType = default_storage_field(Parameter)

    descriptor: ContactPointDescriptor = default_composite_field(time_varying=False)

    input_descriptor: dataclasses.InitVar[ContactPointDescriptor] = dataclasses.field(
        default=None
    )

    def __post_init__(self, input_descriptor: ContactPointDescriptor) -> None:
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.f = np.zeros(3)
        self.f_dot = np.zeros(3)

        self.p0 = np.zeros(3)
        self.v0 = np.zeros(3)
        self.f0 = np.zeros(3)

        self.descriptor = input_descriptor
