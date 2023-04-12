import dataclasses

import casadi as cs
import numpy as np

from hippopt import (
    OptimizationObject,
    OptiSolver,
    Parameter,
    StorageType,
    TOptimizationObject,
    Variable,
    default_storage_field,
)


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


def test_generate_objects():
    test_var = AggregateClass()
    solver = OptiSolver()
    opti_var = solver.generate_optimization_objects(test_var)
    assert isinstance(opti_var.aggregated.parameter, cs.MX)
    assert opti_var.aggregated.parameter.shape == (3, 1)
    assert isinstance(opti_var.aggregated.variable, cs.MX)
    assert opti_var.aggregated.variable.shape == (3, 1)
    assert isinstance(opti_var.other_parameter, cs.MX)
    assert opti_var.other_parameter.shape == (3, 1)
    assert opti_var.other == "untouched"
    assert solver.get_optimization_objects() is opti_var


def test_generate_objects_list():
    test_var_list = []
    for _ in range(2):
        test_var_list.append(AggregateClass())
    solver = OptiSolver()
    opti_var_list = solver.generate_optimization_objects(test_var_list)
    assert len(opti_var_list) == 2
    for opti_var in opti_var_list:
        assert isinstance(opti_var.aggregated.parameter, cs.MX)
        assert opti_var.aggregated.parameter.shape == (3, 1)
        assert isinstance(opti_var.aggregated.variable, cs.MX)
        assert opti_var.aggregated.variable.shape == (3, 1)
        assert isinstance(opti_var.other_parameter, cs.MX)
        assert opti_var.other_parameter.shape == (3, 1)
        assert opti_var.other == "untouched"
    assert solver.get_optimization_objects() is opti_var_list