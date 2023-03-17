import dataclasses
from typing import Any, ClassVar, TypeVar

from hippopt.base.optimization_object import OptimizationObject

TContinuousVariable = TypeVar("TContinuousVariable", bound="ContinuousVariable")


@dataclasses.dataclass
class ContinuousVariable(OptimizationObject):
    """"""

    StorageType: ClassVar[str] = "continuous_variable"
    StorageTypeMetadata: ClassVar[dict[str, Any]] = dict(StorageType=StorageType)
