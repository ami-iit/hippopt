import dataclasses

import casadi as cs
import numpy as np

from hippopt import (
    CompositeType,
    OptimizationObject,
    OptiSolver,
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
    assert isinstance(opti_var.aggregated.scalar, cs.MX)
    assert opti_var.aggregated.scalar.shape == (1, 1)
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
        assert isinstance(opti_var.aggregated.scalar, cs.MX)
        assert opti_var.aggregated.scalar.shape == (1, 1)
        assert isinstance(opti_var.other_parameter, cs.MX)
        assert opti_var.other_parameter.shape == (3, 1)
        assert opti_var.other == "untouched"
    assert solver.get_optimization_objects() is opti_var_list
