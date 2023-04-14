import dataclasses
from enum import Enum
from typing import Any, ClassVar, TypeVar

from hippopt.base.optimization_object import OptimizationObject, TimeExpansion

TVariable = TypeVar("TVariable", bound="Variable")


class VariableType(Enum):
    continuous = 0


@dataclasses.dataclass
class Variable(OptimizationObject):
    """"""

    StorageTypeValue: ClassVar[str] = "variable"
    VariableTypeField: ClassVar[str] = "VariableType"
    StorageTypeMetadata: ClassVar[dict[str, Any]] = dict(
        StorageType=StorageTypeValue,
        TimeDependent=True,
        TimeExpansion=TimeExpansion.List,
        VariableType=VariableType.continuous,
    )

    @classmethod
    def default_storage_field(
        cls,
        time_dependent: bool = True,
        time_expansion: TimeExpansion = TimeExpansion.List,
        variable_type: VariableType = VariableType.continuous,
    ):
        cls_dict = cls.StorageTypeMetadata
        cls_dict[OptimizationObject.TimeDependentField] = time_dependent
        cls_dict[OptimizationObject.TimeExpansionField] = time_expansion
        cls_dict[cls.VariableTypeField] = variable_type

        return dataclasses.field(
            default=None,
            metadata=cls_dict,
        )
