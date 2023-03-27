import dataclasses

import casadi as cs
import numpy as np
import pytest

from hippopt import (
    MultipleShootingSolver,
    OptimizationObject,
    Parameter,
    StorageType,
    Variable,
    default_storage_field,
)


@dataclasses.dataclass
class TestVar(OptimizationObject):
    variable: StorageType = default_storage_field(Variable)
    parameter: StorageType = default_storage_field(Parameter)

    def __post_init__(self):
        self.variable = np.zeros(3)
        self.parameter = np.zeros(3)


def test_variables_to_horizon():
    solver = MultipleShootingSolver()
    vars = solver.generate_optimization_objects(TestVar(), horizon=10)
