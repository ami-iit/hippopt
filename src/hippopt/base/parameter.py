import dataclasses
from typing import Any, ClassVar, Type, TypeVar

from hippopt.base.optimization_object import OptimizationObject

TParameter = TypeVar("TParameter", bound="Parameter")


@dataclasses.dataclass
class Parameter(OptimizationObject):
    """"""

    StorageType: ClassVar[str] = "parameter"
    StorageTypeMetadata: ClassVar[dict[str, Any]] = dict(
        StorageType=StorageType, TimeDependent=False
    )

    @classmethod
    def default_storage_field(cls, time_dependent: bool = False):
        cls_dict = cls.StorageTypeMetadata
        cls_dict[OptimizationObject.TimeDependentField] = time_dependent

        return dataclasses.field(
            default=None,
            metadata=cls_dict,
        )
