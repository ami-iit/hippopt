import dataclasses

import numpy as np

from hippopt import (
    OptimizationObject,
    Parameter,
    StorageType,
    TOptimizationObject,
    Variable,
    default_storage_field,
)


@dataclasses.dataclass
class MyTestVariable(OptimizationObject):
    storage: StorageType = default_storage_field(cls=Variable)

    def __post_init__(self):
        self.storage = np.ones(shape=3)


@dataclasses.dataclass
class MyTestParameter(OptimizationObject):
    storage: StorageType = default_storage_field(cls=Parameter)

    def __post_init__(self):
        self.storage = np.ones(shape=3)


def test_zero_variable():
    test_var = MyTestVariable()
    test_var_zero = test_var.get_default_initialized_object()
    assert test_var_zero.storage.shape == (3,)
    assert np.all(test_var_zero.storage == 0)


def test_zero_parameter():
    test_par = MyTestParameter()
    test_par_zero = test_par.get_default_initialized_object()
    assert test_par_zero.storage.shape == (3,)
    assert np.all(test_par_zero.storage == 0)


@dataclasses.dataclass
class CustomInitializationVariable(OptimizationObject):
    variable: StorageType = default_storage_field(cls=Variable)
    parameter: StorageType = default_storage_field(cls=Parameter)

    def __post_init__(self):
        self.variable = np.ones(shape=3)
        self.parameter = np.ones(shape=3)

    def get_default_initialization(
        self: TOptimizationObject, field_name: str
    ) -> np.ndarray:
        if field_name == "variable":
            return 2 * np.ones(2)

        return OptimizationObject.get_default_initialization(self, field_name)


@dataclasses.dataclass
class AggregateClass(OptimizationObject):
    aggregated: CustomInitializationVariable = dataclasses.field(
        default_factory=CustomInitializationVariable
    )
    other_parameter: StorageType = default_storage_field(cls=Parameter)
    other: str = ""

    def __post_init__(self):
        self.other_parameter = np.ones(3)
        self.other = "untouched"


def test_custom_initialization():
    test_var = CustomInitializationVariable()
    test_var_init = test_var.get_default_initialized_object()
    assert test_var_init.parameter.shape == (3,)
    assert np.all(test_var_init.parameter == 0)
    assert test_var_init.variable.shape == (2,)
    assert np.all(test_var_init.variable == 2)


def test_aggregated():
    test_var = AggregateClass()
    test_var_init = test_var.get_default_initialized_object()
    assert test_var_init.aggregated.parameter.shape == (3,)
    assert np.all(test_var_init.aggregated.parameter == 0)
    assert test_var_init.aggregated.variable.shape == (2,)
    assert np.all(test_var_init.aggregated.variable == 2)
    assert test_var_init.other_parameter.shape == (3,)
    assert np.all(test_var_init.other_parameter == 0)
    assert test_var_init.other == "untouched"
