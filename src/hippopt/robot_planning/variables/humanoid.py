import dataclasses

import numpy as np

from hippopt import (
    CompositeType,
    OptimizationObject,
    OverridableVariable,
    StorageType,
    default_composite_field,
    default_storage_field,
)
from hippopt.robot_planning.variables.contacts import (
    ContactPointState,
    FeetContactPointDescriptors,
    FeetContactPoints,
)
from hippopt.robot_planning.variables.floating_base import FloatingBaseSystemState


@dataclasses.dataclass
class HumanoidState(OptimizationObject):
    contact_points: CompositeType[FeetContactPoints] = default_composite_field(
        factory=FeetContactPoints, time_varying=False
    )

    kinematics: CompositeType[FloatingBaseSystemState] = default_composite_field(
        factory=FloatingBaseSystemState, time_varying=False
    )

    com: StorageType = default_storage_field(OverridableVariable)

    contact_point_descriptors: dataclasses.InitVar[
        FeetContactPointDescriptors
    ] = dataclasses.field(default=None)
    number_of_joints: dataclasses.InitVar[int] = dataclasses.field(default=None)

    def __post_init__(
        self,
        contact_point_descriptors: FeetContactPointDescriptors,
        number_of_joints: int,
    ) -> None:
        if contact_point_descriptors is not None:
            self.contact_points.left = [
                ContactPointState(input_descriptor=point)
                for point in contact_point_descriptors.left
            ]
            self.contact_points.right = [
                ContactPointState(input_descriptor=point)
                for point in contact_point_descriptors.right
            ]
        if number_of_joints is not None:
            self.kinematics = FloatingBaseSystemState(
                number_of_joints_state=number_of_joints
            )
        if self.com is None:
            self.com = np.zeros(3)
