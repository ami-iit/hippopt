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
    OverrideIfCompositeField: ClassVar[str] = "OverrideIfComposite"
    CompositeTypeField: ClassVar[str] = "CompositeType"
    StorageTypeMetadata: ClassVar[dict[str, Any]] = dict(
        StorageType=StorageTypeValue,
        TimeDependent=False,
        TimeExpansion=TimeExpansion.List,
        OverrideIfComposite=False,
    )

    def to_list(self) -> list:
        output_list = []
        for field in dataclasses.fields(self):
            composite_value = self.__getattribute__(field.name)

            is_list = isinstance(composite_value, list)
            list_of_optimization_objects = is_list and all(
                isinstance(elem, OptimizationObject) or isinstance(elem, list)
                for elem in composite_value
            )
            list_of_float = is_list and all(
                isinstance(elem, float) for elem in composite_value
            )
            if list_of_float:
                is_list = False

            if list_of_optimization_objects:
                for elem in composite_value:
                    output_list += elem.to_list()
                continue

            if isinstance(composite_value, OptimizationObject):
                output_list += composite_value.to_list()
                continue

            if OptimizationObject.StorageTypeField in field.metadata:
                value_list = composite_value if is_list else [composite_value]
                for value in value_list:
                    output_list.append(value)
                continue

        return output_list

    def to_mx(self) -> cs.MX:
        return cs.vertcat(*self.to_list())

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


def default_composite_field(
    cls: Type[OptimizationObject] = None, factory=None, time_varying: bool = True
):
    cls_dict = time_varying_metadata(time_varying)
    cls_dict[OptimizationObject.CompositeTypeField] = (
        cls.StorageTypeMetadata if cls is not None else None
    )

    return dataclasses.field(
        default_factory=factory,
        metadata=cls_dict,
    )
