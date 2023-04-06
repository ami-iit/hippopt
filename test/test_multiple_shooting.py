import dataclasses

import casadi as cs
import numpy as np

from hippopt import (
    MultipleShootingSolver,
    OptimizationObject,
    Parameter,
    StorageType,
    Variable,
    default_storage_field,
)


@dataclasses.dataclass
class MyTestVar(OptimizationObject):
    variable: StorageType = default_storage_field(Variable)
    parameter: StorageType = default_storage_field(Parameter)
    string: str = "test"

    def __post_init__(self):
        self.variable = np.zeros(3)
        self.parameter = np.zeros(3)


def test_variables_to_horizon():
    horizon_len = 10
    solver = MultipleShootingSolver()
    var = solver.generate_optimization_objects(MyTestVar(), horizon=horizon_len)
    assert var.string == "test"
    assert len(var.variable) == horizon_len
    assert all(v.shape == (3, 1) for v in var.variable)
    assert isinstance(var.parameter, cs.MX)
    assert var.parameter.shape == (3, 1)


# TODO Stefano: add test with expand_storage and selecting different horizons
