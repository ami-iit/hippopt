from hippopt.base.optimization_object import OptimizationObject
from hippopt.common import Any, ClassVar, TypeVar, dataclasses

TVariable = TypeVar("TVariable", bound="Variable")


@dataclasses.dataclass
class Variable(OptimizationObject):
    """"""

    StorageType: ClassVar[str] = "variable"
    StorageTypeMetadata: ClassVar[dict[str, Any]] = dict(StorageType=StorageType)
