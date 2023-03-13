import dataclasses
from typing import Any, ClassVar, TypeVar

from hippopt.base.optimization_object import OptimizationObject

TParameter = TypeVar("TParameter", bound="Parameter")


@dataclasses.dataclass
class Parameter(OptimizationObject):
    """"""

    StorageType: ClassVar[str] = "parameter"
    StorageTypeMetadata: ClassVar[dict[str, Any]] = dict(StorageType=StorageType)
