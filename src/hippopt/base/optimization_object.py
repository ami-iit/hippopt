import abc
import copy
import dataclasses
from typing import Any, ClassVar, Type, TypeVar

import casadi as cs
import numpy as np

TOptimizationObject = TypeVar("TOptimizationObject", bound="OptimizationObject")
StorageType = cs.MX | np.ndarray


@dataclasses.dataclass
class OptimizationObject(abc.ABC):
    StorageType: ClassVar[str] = "generic"
    StorageTypeField: ClassVar[str] = "StorageType"
    TimeDependentField: ClassVar[str] = "TimeDependent"
    StorageTypeMetadata: ClassVar[dict[str, Any]] = dict(
        StorageType=StorageType, TimeDependent=False
    )

    @classmethod
    def default_storage_field(cls, **kwargs):
        pass

    def get_default_initialization(
        self: TOptimizationObject, field_name: str
    ) -> np.ndarray:
        """
        Get the default initialization of a given field
        It is supposed to be called only for the fields having the StorageType metadata
        """
        return np.zeros(dataclasses.asdict(self)[field_name].shape)

    def get_default_initialized_object(
        self: TOptimizationObject,
    ) -> TOptimizationObject:
        """
        :return: A copy of the object with its initial values
        """

        output = copy.deepcopy(self)

        for field in dataclasses.fields(output):
            if self.StorageTypeField in field.metadata:
                output.__setattr__(
                    field.name, output.get_default_initialization(field.name)
                )
                continue

            if isinstance(output.__getattribute__(field.name), OptimizationObject):
                output.__setattr__(
                    field.name,
                    output.__getattribute__(
                        field.name
                    ).get_default_initialized_object(),
                )

        return output


def default_storage_field(cls: Type[OptimizationObject], **kwargs):
    return cls.default_storage_field(**kwargs)
