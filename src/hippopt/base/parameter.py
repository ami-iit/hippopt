from hippopt.base.optimization_object import OptimizationObject
from hippopt.common import Any, ClassVar, TypeVar, dataclasses

TParameter = TypeVar("TParameter", bound="Parameter")


@dataclasses.dataclass
class Parameter(OptimizationObject):
    """"""

    StorageType: ClassVar[str] = "parameter"
    StorageTypeMetadata: ClassVar[dict[str, Any]] = dict(StorageType=StorageType)
