import dataclasses
from typing import List

import casadi as cs
import numpy as np

from hippopt import (
    MultipleShootingSolver,
    OptimalControlProblem,
    OptimizationObject,
    Parameter,
    StorageType,
    Variable,
    default_storage_field,
    time_varying_metadata,
)


@dataclasses.dataclass
class MyTestVar(OptimizationObject):
    variable: StorageType = default_storage_field(Variable)
    parameter: StorageType = default_storage_field(Parameter)
    string: str = "test"

    def __post_init__(self):
        self.variable = np.zeros(3)
        self.parameter = np.zeros(3)


@dataclasses.dataclass
class MyCompositeTestVar(OptimizationObject):
    composite: MyTestVar | List[MyTestVar] = dataclasses.field(
        default_factory=MyTestVar, metadata=time_varying_metadata()
    )


def test_simple_variables_to_horizon():
    horizon_len = 10
    solver = MultipleShootingSolver()
    var = solver.generate_optimization_objects(MyTestVar(), horizon=horizon_len)
    assert var.string == "test"
    assert len(var.variable) == horizon_len
    assert all(v.shape == (3, 1) for v in var.variable)
    assert isinstance(var.parameter, cs.MX)
    assert var.parameter.shape == (3, 1)


def test_composite_variables_to_horizon():
    horizon_len = 10
    solver = MultipleShootingSolver()
    var = solver.generate_optimization_objects(
        MyCompositeTestVar(), horizon=horizon_len
    )
    assert all(comp.string == "test" for comp in var.composite)
    assert len(var.composite) == horizon_len
    assert all(c.variable.shape == (3, 1) for c in var.composite)


def test_flattened_variables_simple():
    horizon_len = 10

    problem, var = OptimalControlProblem.create(
        input_structure=MyTestVar(), horizon=horizon_len
    )

    var_flat = problem.solver().get_flattened_optimization_objects()
    assert "string" not in var_flat
    assert var_flat[0]["variable"][0] == horizon_len
    assert var_flat[0]["parameter"][0] == 1
    assert next(var_flat[0]["parameter"][1]()) is var.parameter
    assert (
        next(var_flat[0]["parameter"][1]()) is var.parameter
    )  # check that we can use the generator twice
    variable_gen = var_flat[0]["variable"][1]()
    assert all(next(variable_gen) is v for v in var.variable)
    variable_gen = var_flat[0]["variable"][1]()
    assert all(
        next(variable_gen) is v for v in var.variable
    )  # check that we can use the generator twice


def test_flattened_variables_composite():
    horizon_len = 10

    problem, var = OptimalControlProblem.create(
        input_structure=MyCompositeTestVar(), horizon=horizon_len
    )

    var_flat = problem.solver().get_flattened_optimization_objects()
    assert var_flat[0]["composite.variable"][0] == horizon_len
    assert var_flat[0]["composite.parameter"][0] == horizon_len
    par_gen = var_flat[0]["composite.parameter"][1]()
    assert all(next(par_gen) is c.parameter for c in var.composite)
    variable_gen = var_flat[0]["composite.variable"][1]()
    assert all(next(variable_gen) is c.variable for c in var.composite)


# TODO Stefano: add test with expand_storage and selecting different horizons
# TODO Stefano: add test with composite variables and with lists
# TODO Stefano: add test on multiple shooting add_dynamics
