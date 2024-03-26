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

    @staticmethod
    def _scan(
        input_object: TOptimizationObject | list[TOptimizationObject],
        name_prefix: str = "",
        parent_metadata: dict | None = None,
        input_dict: dict | None = None,
    ) -> (dict, dict):
        output_dict = {}
        metadata_dict = {}
        if isinstance(input_object, list):
            assert all(
                isinstance(elem, OptimizationObject) or isinstance(elem, list)
                for elem in input_object
            )
            for i, elem in enumerate(input_object):
                inner_dict, inner_metadata = OptimizationObject._scan(
                    input_object=elem,
                    name_prefix=name_prefix + f"[{str(i)}].",
                    parent_metadata=parent_metadata,
                    input_dict=input_dict,
                )
                output_dict.update(inner_dict)
                metadata_dict.update(inner_metadata)
            return output_dict, metadata_dict

        assert isinstance(input_object, OptimizationObject)
        for field in dataclasses.fields(input_object):
            composite_value = input_object.__getattribute__(field.name)

            list_of_optimization_objects = (
                isinstance(composite_value, list)
                and len(composite_value) > 0
                and all(
                    isinstance(elem, OptimizationObject) or isinstance(elem, list)
                    for elem in composite_value
                )
            )

            if (
                isinstance(composite_value, OptimizationObject)
                or list_of_optimization_objects
            ):
                new_parent_metadata = parent_metadata
                has_composite_metadata = (
                    OptimizationObject.CompositeTypeField in field.metadata
                    and field.metadata[OptimizationObject.CompositeTypeField]
                    is not None
                )
                if has_composite_metadata:
                    composite_metadata = field.metadata[
                        OptimizationObject.CompositeTypeField
                    ]
                    use_old_metadata = (
                        parent_metadata is not None
                        and OptimizationObject.OverrideIfCompositeField
                        in composite_metadata
                        and composite_metadata[
                            OptimizationObject.OverrideIfCompositeField
                        ]
                    )

                    if not use_old_metadata:
                        new_parent_metadata = composite_metadata

                separator = "" if list_of_optimization_objects else "."
                inner_dict, inner_metadata = OptimizationObject._scan(
                    input_object=composite_value,
                    name_prefix=name_prefix + field.name + separator,
                    parent_metadata=new_parent_metadata,
                    input_dict=input_dict,
                )
                output_dict.update(inner_dict)
                metadata_dict.update(inner_metadata)
                continue

            if OptimizationObject.StorageTypeField in field.metadata:
                should_override = (
                    OptimizationObject.OverrideIfCompositeField in field.metadata
                    and field.metadata[OptimizationObject.OverrideIfCompositeField]
                )
                parent_can_override = (
                    parent_metadata is not None
                    and OptimizationObject.StorageTypeField in parent_metadata
                )
                value_metadata = field.metadata.copy()

                if should_override and parent_can_override:
                    value_metadata[OptimizationObject.StorageTypeField] = (
                        parent_metadata[OptimizationObject.StorageTypeField]
                    )

                value_is_list = isinstance(composite_value, list)
                list_of_float = value_is_list and (
                    len(composite_value) == 0
                    or all(isinstance(elem, float) for elem in composite_value)
                )
                if list_of_float:
                    composite_value = np.array(composite_value)
                    value_is_list = False

                value_list = composite_value if value_is_list else [composite_value]
                name_radix = name_prefix + field.name
                value_from_dict = []
                for i, val in enumerate(value_list):
                    postfix = f"[{i}]" if value_is_list else ""
                    full_name = name_radix + postfix

                    if input_dict is not None and full_name in input_dict:
                        value_from_dict.append(input_dict[full_name])

                    metadata_dict[full_name] = value_metadata
                    output_dict[full_name] = (
                        composite_value[i] if value_is_list else composite_value
                    )

                if len(value_from_dict) > 0:
                    input_object.__setattr__(
                        field.name,
                        value_from_dict if value_is_list else value_from_dict[0],
                    )

                continue

        return output_dict, metadata_dict

    def to_dict(self, prefix: str = "") -> (dict, dict):
        output_dict, metadata_dict = OptimizationObject._scan(
            input_object=self, name_prefix=prefix
        )
        return output_dict, metadata_dict

    def from_dict(self, input_dict: dict, prefix: str = "") -> None:
        OptimizationObject._scan(
            input_object=self, name_prefix=prefix, input_dict=input_dict
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
