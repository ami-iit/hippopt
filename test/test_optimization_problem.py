import dataclasses

import casadi as cs
import numpy as np
import pytest

from hippopt import (
    ExpressionType,
    OptimizationObject,
    OptimizationProblem,
    StorageType,
    Variable,
    default_storage_field,
)


@dataclasses.dataclass
class TestVar(OptimizationObject):
    variable: StorageType = default_storage_field(Variable)

    def __post_init__(self):
        self.variable = np.zeros(3)


def test_opti_solver():
    problem = OptimizationProblem()
    var = problem.generate_optimization_objects(input_structure=TestVar())
    np.random.seed(123)
    a = 10.0 * np.random.rand(3) + 0.01
    b = 20.0 * np.random.rand(3) - 10.0
    c = 20.0 * np.random.rand(3) - 10.0

    problem.add_expression(
        mode=ExpressionType.minimize,
        expression=(
            a[k] * cs.power(var.variable[k], 2) + b[k] * var.variable[k]
            for k in range(0, 3)
        ),
    )

    problem.add_expression(
        mode=ExpressionType.subject_to,
        expression=(var.variable[k] >= c[k] for k in range(3)),  # noqa
    )

    output = problem.solver().solve()
    # TODO: Stefano The output should not be opti specific

    expected_x = np.zeros(3)
    expected_cost = 0
    for i in range(3):
        expected = -b[i] / (2 * a[i])
        expected_x[i] = expected if expected >= c[i] else c[i]
        expected_cost += (
            -b[i] ** 2 / (4 * a[i])
            if expected >= c[i]
            else a[i] * (c[i] ** 2) + b[i] * c[i]
        )

    assert output.value(var.variable) == pytest.approx(expected_x)
    assert output.value(problem.solver().cost()) == pytest.approx(expected_cost)


# TODO: Stefano test setting of initial condition and of parameters
