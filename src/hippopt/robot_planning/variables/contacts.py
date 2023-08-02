import dataclasses

import numpy as np

from hippopt import (
    OptimizationObject,
    Parameter,
    StorageType,
    Variable,
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
    ):
        self.foot_frame = input_foot_frame
        self.position_in_foot_frame = input_position_in_foot_frame


@dataclasses.dataclass
class ContactPoint(OptimizationObject):
    p: StorageType = default_storage_field(Variable)
    v: StorageType = default_storage_field(Variable)
    u_v: StorageType = default_storage_field(Variable)
    f: StorageType = default_storage_field(Variable)
    f_dot: StorageType = default_storage_field(Variable)

    descriptor: ContactPointDescriptor = dataclasses.field(default=None)

    input_descriptor: dataclasses.InitVar[ContactPointDescriptor] = dataclasses.field(
        default=None
    )

    def __post_init__(self, input_descriptor: ContactPointDescriptor):
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.u_v = np.zeros(3)
        self.f = np.zeros(3)
        self.f_dot = np.zeros(3)

        self.descriptor = input_descriptor
