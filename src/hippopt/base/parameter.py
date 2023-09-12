import copy
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
        OverrideIfComposite=False,
    )

    @classmethod
    def default_storage_metadata(
        cls,
        time_dependent: bool = False,
        time_expansion: TimeExpansion = TimeExpansion.List,
    ):
        cls_dict = copy.deepcopy(cls.StorageTypeMetadata)
        cls_dict[OptimizationObject.TimeDependentField] = time_dependent
        cls_dict[OptimizationObject.TimeExpansionField] = time_expansion

        return cls_dict


@dataclasses.dataclass
class OverridableParameter(Parameter):
    """"""

    StorageTypeMetadata: ClassVar[dict[str, Any]] = dict(
        StorageType=Parameter.StorageTypeValue,
        TimeDependent=False,
        TimeExpansion=TimeExpansion.List,
        OverrideIfComposite=True,
    )
