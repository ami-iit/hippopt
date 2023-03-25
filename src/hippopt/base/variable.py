import dataclasses
from enum import Enum
from typing import Any, ClassVar, TypeVar

from hippopt.base.optimization_object import OptimizationObject

TVariable = TypeVar("TVariable", bound="Variable")


class VariableType(Enum):
    continuous = 0


@dataclasses.dataclass
class Variable(OptimizationObject):
    """"""

    StorageType: ClassVar[str] = "variable"
    VariableTypeField: ClassVar[str] = "VariableType"
    StorageTypeMetadata: ClassVar[dict[str, Any]] = dict(
        StorageType=StorageType,
        TimeDependent=False,
        VariableType=VariableType.continuous,
    )

    @classmethod
    def default_storage_field(
        cls,
        time_dependent: bool = True,
        variable_type: VariableType = VariableType.continuous,
    ):
        cls_dict = cls.StorageTypeMetadata
        cls_dict[OptimizationObject.TimeDependentField] = time_dependent
        cls_dict[cls.VariableTypeField] = variable_type

        return dataclasses.field(
            default=None,
            metadata=cls_dict,
        )
