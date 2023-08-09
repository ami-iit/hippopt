import abc
import dataclasses
from enum import Enum
from typing import Any, ClassVar, Type, TypeVar

import casadi as cs
import numpy as np

TOptimizationObject = TypeVar("TOptimizationObject", bound="OptimizationObject")
TGenericCompositeObject = TypeVar("TGenericCompositeObject")
CompositeType = TGenericCompositeObject | list[TGenericCompositeObject]
StorageType = CompositeType[cs.MX] | CompositeType[np.ndarray] | CompositeType[float]


class TimeExpansion(Enum):
    List = 0
    Matrix = 1


@dataclasses.dataclass
class OptimizationObject(abc.ABC):
    StorageTypeValue: ClassVar[str] = "generic"
    StorageTypeField: ClassVar[str] = "StorageType"
    TimeDependentField: ClassVar[str] = "TimeDependent"
    TimeExpansionField: ClassVar[str] = "TimeExpansion"
    StorageTypeMetadata: ClassVar[dict[str, Any]] = dict(
        StorageType=StorageType, TimeDependent=False, TimeExpansion=TimeExpansion.List
    )

    @classmethod
    def default_storage_metadata(cls, **kwargs) -> dict:
        pass


def default_storage_metadata(cls: Type[OptimizationObject], **kwargs) -> dict:
    return cls.default_storage_metadata(**kwargs)


def default_storage_field(cls: Type[OptimizationObject], **kwargs):
    return dataclasses.field(
        default=None, metadata=default_storage_metadata(cls, **kwargs)
    )


def time_varying_metadata(time_varying: bool = True):
    return {OptimizationObject.TimeDependentField: time_varying}


def default_composite_field(factory=None, time_varying: bool = True):
    return dataclasses.field(
        default_factory=factory,
        metadata=time_varying_metadata(time_varying),
    )
