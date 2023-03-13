import dataclasses
from typing import Any, ClassVar, TypeVar

from hippopt.base.optimization_object import OptimizationObject

TVariable = TypeVar("TVariable", bound="Variable")


@dataclasses.dataclass
class Variable(OptimizationObject):
    """"""

    StorageType: ClassVar[str] = "variable"
    StorageTypeMetadata: ClassVar[dict[str, Any]] = dict(StorageType=StorageType)
