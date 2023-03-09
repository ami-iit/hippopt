import dataclasses

import numpy as np

from hippopt import (
    OptimizationObject,
    Parameter,
    StorageType,
    TOptimizationObject,
    Variable,
    default_storage_type,
)


@dataclasses.dataclass
class TestVariable(OptimizationObject):
    storage: StorageType = default_storage_type(Variable)

    def __post_init__(self):
        self.storage = np.ones(shape=3)


@dataclasses.dataclass
class TestParameter(OptimizationObject):
    storage: StorageType = default_storage_type(Parameter)

    def __post_init__(self):
        self.storage = np.ones(shape=3)


def test_zero_variable():
    test_var = TestVariable()
    test_var_zero = test_var.get_default_initialized_object()
    assert test_var_zero.storage.shape == (3,)
    assert np.all(test_var_zero.storage == 0)


def test_zero_parameter():
    test_par = TestParameter()
    test_par_zero = test_par.get_default_initialized_object()
    assert test_par_zero.storage.shape == (3,)
    assert np.all(test_par_zero.storage == 0)


@dataclasses.dataclass
class CustomInitializationVariable(OptimizationObject):
    variable: StorageType = default_storage_type(Variable)
    parameter: StorageType = default_storage_type(Parameter)

    def __post_init__(self):
        self.variable = np.ones(shape=3)
        self.parameter = np.ones(shape=3)

    def get_default_initialization(
        self: TOptimizationObject, field_name: str
    ) -> np.ndarray:
        if field_name == "variable":
            return 2 * np.ones(2)

        return OptimizationObject.get_default_initialization(self, field_name)


def test_custom_initialization():
    test_var = CustomInitializationVariable()
    test_var_init = test_var.get_default_initialized_object()
    assert test_var_init.parameter.shape == (3,)
    assert np.all(test_var_init.parameter == 0)
    assert test_var_init.variable.shape == (2,)
    assert np.all(test_var_init.variable == 2)


@dataclasses.dataclass
class AggregateClass(OptimizationObject):
    aggregated: CustomInitializationVariable
    other_parameter: StorageType = default_storage_type(Parameter)
    other: str = ""

    def __post_init__(self):
        self.aggregated = CustomInitializationVariable()
        self.other_parameter = np.ones(3)
        self.other = "untouched"


def test_aggregated():
    test_var = AggregateClass(aggregated=CustomInitializationVariable())
    test_var_init = test_var.get_default_initialized_object()
    assert test_var_init.aggregated.parameter.shape == (3,)
    assert np.all(test_var_init.aggregated.parameter == 0)
    assert test_var_init.aggregated.variable.shape == (2,)
    assert np.all(test_var_init.aggregated.variable == 2)
    assert test_var_init.other_parameter.shape == (3,)
    assert np.all(test_var_init.other_parameter == 0)
    assert test_var_init.other == "untouched"
