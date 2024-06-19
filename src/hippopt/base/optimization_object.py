import abc
import dataclasses
from enum import Enum
from typing import Any, Callable, ClassVar, Type, TypeVar

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
    IsValueFilter: ClassVar[Callable[[str, Any, dict], bool]] = (
        lambda _, value, __: isinstance(value, np.ndarray)
        or isinstance(value, cs.DM)
        or isinstance(value, cs.MX)
    )
    DMConversion: ClassVar[Callable[[str, Any], Any]] = lambda _, value: (
        value.full().flatten() if isinstance(value, cs.DM) else value
    )

    @staticmethod
    def _convert_to_np_array(value: Any) -> Any | np.ndarray:
        output_value = value
        list_of_numbers = isinstance(output_value, list) and (
            len(output_value) == 0
            or all(isinstance(elem, float) for elem in output_value)
            or all(isinstance(elem, int) for elem in output_value)
        )
        if list_of_numbers:
            output_value = np.array(output_value, dtype=float)

        if isinstance(output_value, np.ndarray):
            if output_value.ndim < 2:
                output_value = np.expand_dims(output_value, axis=1)

        if isinstance(output_value, float) or isinstance(output_value, int):
            output_value = output_value * np.ones((1, 1), dtype=float)

        return output_value

    @staticmethod
    def _scan(
        input_object: TOptimizationObject | list[TOptimizationObject],
        name_prefix: str = "",
        parent_metadata: dict | None = None,
        input_dict: dict | None = None,
        output_filter: Callable[[str, Any, dict], bool] | None = None,
        input_conversion: Callable[[str, Any], Any] | None = None,
        output_flat: bool = True,
    ) -> tuple[dict, dict] | tuple[list, list]:
        output_dict = {}
        metadata_dict = {}
        if isinstance(input_object, list):
            if not all(
                isinstance(elem, OptimizationObject) or isinstance(elem, list)
                for elem in input_object
            ):
                raise ValueError(
                    "The input object is a list, but not all elements are"
                    " OptimizationObject instances."
                )
            output_list = []
            output_metadata_list = []
            if not output_flat and name_prefix != "":
                output_dict[name_prefix] = output_list
                metadata_dict[name_prefix] = output_metadata_list

            for i, elem in enumerate(input_object):
                inner_dict, inner_metadata = OptimizationObject._scan(
                    input_object=elem,
                    name_prefix=name_prefix + f"[{str(i)}]." if output_flat else "",
                    parent_metadata=parent_metadata,
                    input_dict=input_dict,
                    output_filter=output_filter,
                    input_conversion=input_conversion,
                    output_flat=output_flat,
                )
                output_dict.update(inner_dict)
                output_list.append(inner_dict)
                metadata_dict.update(inner_metadata)
                output_metadata_list.append(inner_metadata)

            if not output_flat and name_prefix == "":
                return output_list, output_metadata_list
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
                    name_prefix=(
                        name_prefix + field.name + separator if output_flat else ""
                    ),
                    parent_metadata=new_parent_metadata,
                    input_dict=input_dict,
                    output_filter=output_filter,
                    input_conversion=input_conversion,
                    output_flat=output_flat,
                )

                if output_flat:
                    output_dict.update(inner_dict)
                    metadata_dict.update(inner_metadata)
                else:
                    output_dict[field.name] = inner_dict
                    metadata_dict[field.name] = inner_metadata

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

                composite_value = OptimizationObject._convert_to_np_array(
                    composite_value
                )
                value_is_list = isinstance(composite_value, list)
                value_list = composite_value if value_is_list else [composite_value]
                name_radix = name_prefix + field.name if output_flat else field.name
                value_from_dict = []

                if not output_flat and value_is_list:
                    output_dict[field.name] = []
                    metadata_dict[field.name] = []

                for i, val in enumerate(value_list):
                    postfix = f"[{i}]" if value_is_list and output_flat else ""
                    full_name = name_radix + postfix

                    if input_dict is not None and full_name in input_dict:
                        converted_input = (
                            input_conversion(full_name, input_dict[full_name])
                            if input_conversion is not None
                            else input_dict[full_name]
                        )
                        value_from_dict.append(converted_input)

                    output_value = (
                        OptimizationObject._convert_to_np_array(composite_value[i])
                        if value_is_list
                        else composite_value
                    )

                    if output_filter is not None:
                        if not output_filter(full_name, output_value, value_metadata):
                            continue

                    if not output_flat and value_is_list:
                        output_dict[full_name].append(output_value)
                        metadata_dict[full_name].append(value_metadata)
                    else:
                        output_dict[full_name] = output_value
                        metadata_dict[full_name] = value_metadata

                if len(value_from_dict) > 0:
                    input_object.__setattr__(
                        field.name,
                        value_from_dict if value_is_list else value_from_dict[0],
                    )

                continue

        if not output_flat and name_prefix != "":
            nested_output = {name_prefix: output_dict}
            nested_metadata = {name_prefix: metadata_dict}
            return nested_output, nested_metadata

        return output_dict, metadata_dict

    def to_dict(
        self,
        prefix: str = "",
        output_filter: Callable[[str, Any, dict], bool] | None = None,
        flatten: bool = True,
    ) -> dict:
        output_dict, _ = OptimizationObject._scan(
            input_object=self,
            name_prefix=prefix,
            output_filter=output_filter,
            output_flat=flatten,
        )
        return output_dict

    def to_dicts(
        self,
        prefix: str = "",
        output_filter: Callable[[str, Any, dict], bool] | None = None,
        flatten: bool = True,
    ) -> (dict, dict):
        output_dict, metadata_dict = OptimizationObject._scan(
            input_object=self,
            name_prefix=prefix,
            output_filter=output_filter,
            output_flat=flatten,
        )
        return output_dict, metadata_dict

    def from_dict(
        self,
        input_dict: dict,
        prefix: str = "",
        input_conversion: Callable[[str, Any], Any] | None = None,
    ) -> None:
        OptimizationObject._scan(
            input_object=self,
            name_prefix=prefix,
            input_dict=input_dict,
            input_conversion=input_conversion,
        )

    def to_list(
        self,
        output_filter: Callable[[str, Any, dict], bool] | None = None,
    ) -> list:
        output_list = []
        as_dict = self.to_dict(output_filter=output_filter)
        for key in sorted(as_dict.keys()):
            output_list.append(as_dict[key])

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
