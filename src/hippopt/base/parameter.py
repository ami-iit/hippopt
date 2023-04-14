import dataclasses
from typing import Any, ClassVar, TypeVar

from hippopt.base.optimization_object import OptimizationObject, TimeExpansion

TParameter = TypeVar("TParameter", bound="Parameter")


@dataclasses.dataclass
class Parameter(OptimizationObject):
    """"""

    StorageTypeValue: ClassVar[str] = "parameter"
    StorageTypeMetadata: ClassVar[dict[str, Any]] = dict(
        StorageType=StorageTypeValue,
        TimeDependent=False,
        TimeExpansion=TimeExpansion.List,
    )

    @classmethod
    def default_storage_field(
        cls,
        time_dependent: bool = False,
        time_expansion: TimeExpansion = TimeExpansion.List,
    ):
        cls_dict = cls.StorageTypeMetadata
        cls_dict[OptimizationObject.TimeDependentField] = time_dependent
        cls_dict[OptimizationObject.TimeExpansionField] = time_expansion

        return dataclasses.field(
            default=None,
            metadata=cls_dict,
        )
