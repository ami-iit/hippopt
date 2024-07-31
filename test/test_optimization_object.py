import copy
import dataclasses

import numpy as np

from hippopt import (
    CompositeType,
    OptimizationObject,
    OverridableVariable,
    Parameter,
    StorageType,
    Variable,
    default_composite_field,
    default_storage_field,
)


@dataclasses.dataclass
class CustomVariable(OptimizationObject):
    variable: StorageType = default_storage_field(cls=Variable)
    parameter: StorageType = default_storage_field(cls=Parameter)
    scalar: StorageType = default_storage_field(cls=Variable)

    def __post_init__(self):
        self.variable = np.ones(shape=3)
        self.parameter = np.ones(shape=3)
        self.scalar = 1.0


@dataclasses.dataclass
class AggregateClass(OptimizationObject):
    aggregated: CompositeType[CustomVariable] = default_composite_field(
        factory=CustomVariable
    )
    aggregated_list: CompositeType[list[CustomVariable]] = default_composite_field(
        factory=list
    )
    other_parameter: StorageType = default_storage_field(cls=Parameter)
    other: str = ""

    def __post_init__(self):
        self.other_parameter = np.ones(3)
        self.other = "untouched"
        for _ in range(3):
            self.aggregated_list.append(CustomVariable())


@dataclasses.dataclass
class CustomOverridableVariable(OptimizationObject):
    overridable: StorageType = default_storage_field(cls=OverridableVariable)
    not_overridable: StorageType = default_storage_field(cls=Variable)

    def __post_init__(self):
        self.overridable = 0.0
        self.not_overridable = 0.0


@dataclasses.dataclass
class CustomCompositeOverridableVariable(OptimizationObject):
    composite: CompositeType[CustomOverridableVariable] = default_composite_field(
        cls=Parameter, factory=CustomOverridableVariable
    )


def test_to_dict_flat():
    test_var = AggregateClass()
    test_var_as_dict = test_var.to_dict()
    assert (
        len(test_var_as_dict) == 3 + 3 * 3 + 1
    )  # 3 for aggregated, 3*3 for aggregated_list, 1 for other_parameter

    assert all(
        expected in test_var_as_dict
        for expected in [
            "aggregated.variable",
            "aggregated.parameter",
            "aggregated.scalar",
            "other_parameter",
            "aggregated_list[0].variable",
            "aggregated_list[0].parameter",
            "aggregated_list[0].scalar",
            "aggregated_list[1].variable",
            "aggregated_list[1].parameter",
            "aggregated_list[1].scalar",
            "aggregated_list[2].variable",
            "aggregated_list[2].parameter",
            "aggregated_list[2].scalar",
        ]
    )
    assert "other" not in test_var_as_dict
    dict_copy = copy.deepcopy(test_var_as_dict)
    dict_copy["aggregated.scalar"] = 7.0


def test_to_dict_metadata():
    test_var = CustomCompositeOverridableVariable()
    _, metadata_dict = test_var.to_dicts()
    assert (
        len(metadata_dict) == 2
    )  # 1 for composite.overridable, 1 for composite.not_overridable
    assert all(
        expected in metadata_dict
        for expected in [
            "composite.overridable",
            "composite.not_overridable",
        ]
    )
    assert (
        metadata_dict["composite.overridable"][OptimizationObject.StorageTypeField]
        == Parameter.StorageTypeValue
    )
    assert (
        metadata_dict["composite.not_overridable"][OptimizationObject.StorageTypeField]
        == Variable.StorageTypeValue
    )


def test_to_dict_not_flat():
    test_var = AggregateClass()
    test_var_as_dict = test_var.to_dict(flatten=False)
    assert (
        len(test_var_as_dict) == 3
    )  # 1 for aggregated, 1 for aggregated_list, 1 for other_parameter
    assert all(
        expected in test_var_as_dict
        for expected in [
            "aggregated",
            "aggregated_list",
            "other_parameter",
        ]
    )
    assert all(
        expected in test_var_as_dict["aggregated"]
        for expected in [
            "variable",
            "parameter",
            "scalar",
        ]
    )

    assert test_var_as_dict["aggregated"]["scalar"] == 1.0
    assert len(test_var_as_dict["aggregated_list"]) == 3

    test_var_as_dict = test_var.to_dict(flatten=False, prefix="test")
    assert len(test_var_as_dict) == 1
    assert "test" in test_var_as_dict
    assert all(
        expected in test_var_as_dict["test"]
        for expected in [
            "aggregated",
            "aggregated_list",
            "other_parameter",
        ]
    )


def test_to_dict_metadata_not_flat():
    test_var = CustomCompositeOverridableVariable()
    _, metadata_dict = test_var.to_dicts(flatten=False)
    assert (
        len(metadata_dict) == 1
    )  # 1 for composite.overridable, 1 for composite.not_overridable
    assert "composite" in metadata_dict
    assert len(metadata_dict["composite"]) == 2
    assert all(
        expected in metadata_dict["composite"]
        for expected in [
            "overridable",
            "not_overridable",
        ]
    )
    assert (
        metadata_dict["composite"]["overridable"][OptimizationObject.StorageTypeField]
        == Parameter.StorageTypeValue
    )
    assert (
        metadata_dict["composite"]["not_overridable"][
            OptimizationObject.StorageTypeField
        ]
        == Variable.StorageTypeValue
    )


def test_to_list():
    test_var = AggregateClass()
    test_var_as_list = test_var.to_list()
    assert (len(test_var_as_list)) == 3 + 3 * 3 + 1

    test_var_as_dict = test_var.to_dict()
    key_to_index = {}
    for i, key in enumerate(sorted(test_var_as_dict.keys())):
        key_to_index[key] = i

    assert test_var_as_list[key_to_index["aggregated.variable"]].shape == (3, 1)
    assert test_var_as_list[key_to_index["aggregated.parameter"]].shape == (3, 1)
    assert test_var_as_list[key_to_index["aggregated.scalar"]].shape == (1, 1)


def test_from_dict():
    test_var = AggregateClass()
    test_var_as_dict = test_var.to_dict()
    test_var_as_dict["aggregated.scalar"] = 7.0
    test_var.from_dict(test_var_as_dict)
    assert test_var.aggregated.scalar == 7.0


def test_to_dict_filtered():
    test_var = AggregateClass()
    test_var.aggregated.scalar = None
    test_var_as_dict = test_var.to_dict(output_filter=OptimizationObject.IsValueFilter)
    assert "aggregated.scalar" not in test_var_as_dict


def test_to_dict_converted():
    test_var = AggregateClass()
    test_var_as_dict = test_var.to_dict(
        output_conversion=lambda _, value: (42 if isinstance(value, float) else value),
    )
    assert test_var_as_dict["aggregated.scalar"] == 42


def test_from_dict_converted():
    test_var = AggregateClass()
    test_var_as_dict = test_var.to_dict()
    test_var_as_dict["aggregated.scalar"] = 7.0
    test_var.from_dict(
        test_var_as_dict,
        input_conversion=lambda _, value: (42 if isinstance(value, float) else value),
    )
    assert test_var.aggregated.scalar == 42
