import dataclasses

import casadi as cs
import numpy as np
import pytest

from hippopt import (
    ContinuousVariable,
    ExpressionType,
    OptimizationObject,
    OptimizationProblem,
    Parameter,
    StorageType,
    default_storage_field,
)


@dataclasses.dataclass
class TestVar(OptimizationObject):
    variable: StorageType = default_storage_field(ContinuousVariable)

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

    output, cost_value = problem.solver().solve()

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

    assert output.variable == pytest.approx(expected_x)
    assert cost_value == pytest.approx(expected_cost)

    assert problem.solver().get_solution().variable == pytest.approx(expected_x)
    assert problem.solver().get_cost_value() == pytest.approx(expected_cost)


@dataclasses.dataclass
class TestVarAndPar(OptimizationObject):
    composite: TestVar = dataclasses.field(default_factory=TestVar)
    parameter: StorageType = default_storage_field(Parameter)

    def __post_init__(self):
        self.parameter = np.zeros(3)


def test_opti_solver_with_parameters():
    problem = OptimizationProblem()
    initial_guess = TestVarAndPar()
    var = problem.generate_optimization_objects(input_structure=TestVarAndPar())
    np.random.seed(123)
    a = 10.0 * np.random.rand(3) + 0.01
    b = 20.0 * np.random.rand(3) - 10.0
    c = 20.0 * np.random.rand(3) - 10.0

    initial_guess.parameter = c

    problem.add_expression(
        mode=ExpressionType.minimize,
        expression=(
            a[k] * cs.power(var.composite.variable[k], 2)
            + b[k] * var.composite.variable[k]
            for k in range(0, 3)
        ),
    )

    problem.add_expression(
        mode=ExpressionType.subject_to,
        expression=(  # noqa
            var.composite.variable[k] >= var.parameter[k] for k in range(3)
        ),
    )

    problem.solver().set_initial_guess(initial_guess=initial_guess)

    output, cost_value = problem.solver().solve()

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

    assert output.composite.variable == pytest.approx(expected_x)
    assert cost_value == pytest.approx(expected_cost)
    assert output.parameter == pytest.approx(c)

    assert problem.solver().get_solution().composite.variable == pytest.approx(
        expected_x
    )
    assert problem.solver().get_cost_value() == pytest.approx(expected_cost)


def test_opti_solver_with_parameters_and_lists():
    problem = OptimizationProblem()
    initial_guess = []
    for _ in range(3):
        initial_guess.append(TestVarAndPar())

    var = problem.generate_optimization_objects(input_structure=initial_guess)
    np.random.seed(123)

    a = []
    b = []
    c = []

    for j in range(len(initial_guess)):
        a.append(10.0 * np.random.rand(3) + 0.01)
        b.append(20.0 * np.random.rand(3) - 10.0)
        c.append(20.0 * np.random.rand(3) - 10.0)
        initial_guess[j].parameter = c[j]

    problem.add_expression(
        mode=ExpressionType.minimize,
        expression=(
            a[j][k] * cs.power(var[j].composite.variable[k], 2)
            + b[j][k] * var[j].composite.variable[k]
            for j in range(len(initial_guess))
            for k in range(0, 3)
        ),
    )

    problem.add_expression(
        mode=ExpressionType.subject_to,
        expression=(  # noqa
            var[j].composite.variable[k] >= c[j][k]
            for j in range(len(initial_guess))
            for k in range(3)
        ),
    )

    problem.solver().set_initial_guess(initial_guess=initial_guess)

    output, cost_value = problem.solver().solve()

    expected_x = np.zeros(3)
    expected_cost = 0
    for i in range(len(initial_guess)):
        for j in range(3):
            expected = -b[i][j] / (2 * a[i][j])
            expected_x[j] = expected if expected >= c[i][j] else c[i][j]
            expected_cost += (
                -b[i][j] ** 2 / (4 * a[i][j])
                if expected >= c[i][j]
                else a[i][j] * (c[i][j] ** 2) + b[i][j] * c[i][j]
            )

        assert output[i].composite.variable == pytest.approx(expected_x)
        assert output[i].parameter == pytest.approx(c[i])

    assert cost_value == pytest.approx(expected_cost)
    assert problem.solver().get_cost_value() == pytest.approx(expected_cost)


@dataclasses.dataclass
class SwitchVar(OptimizationObject):
    x: StorageType = default_storage_field(ContinuousVariable)
    y: StorageType = default_storage_field(ContinuousVariable)

    def __post_init__(self):
        self.x = np.zeros(1)
        self.y = np.zeros(1)


def test_switch_costs():
    initial_problem = OptimizationProblem()
    variables = initial_problem.generate_optimization_objects(SwitchVar())
    a = 10
    initial_problem.add_expression(ExpressionType.minimize, variables.x * variables.x)
    initial_problem.add_expression(
        ExpressionType.minimize, a * variables.y * variables.y
    )
    initial_problem.add_expression(
        ExpressionType.subject_to, variables.x + variables.y == a - 1
    )  # noqa
    output, cost_value = initial_problem.solver().solve()
    expected_cost = a + (a - 2) ** 2
    assert cost_value == pytest.approx(expected=expected_cost, rel=0.1)
    assert output.x == pytest.approx(a - 2, rel=0.1)

    new_problem = OptimizationProblem()
    new_variables = new_problem.generate_optimization_objects(SwitchVar())
    new_problem.add_expression(
        ExpressionType.minimize, a * new_variables.y * new_variables.y
    )
    new_problem.add_expression(
        ExpressionType.subject_to, new_variables.x + new_variables.y == a - 1
    )  # noqa
    new_problem.add_expression(
        ExpressionType.subject_to, new_variables.x * new_variables.x
    )
    output, cost_value = new_problem.solver().solve()
    expected_cost = a * (a - 1) ** 2
    assert cost_value == pytest.approx(expected=expected_cost, rel=0.1)
    assert output.x == pytest.approx(0, abs=1e-4)


def test_switch_constraints():
    initial_problem = OptimizationProblem()
    variables = initial_problem.generate_optimization_objects(SwitchVar())
    a = 10
    initial_problem.add_expression(ExpressionType.minimize, (variables.x - 5) ** 2)
    initial_problem.add_expression(
        ExpressionType.minimize, a * variables.y * variables.y
    )
    initial_problem.add_expression(
        ExpressionType.subject_to, variables.x + variables.y == a - 1
    )  # noqa
    initial_output, initial_cost_value = initial_problem.solver().solve()

    new_problem = OptimizationProblem()
    new_variables = new_problem.generate_optimization_objects(SwitchVar())
    new_problem.add_expression(
        ExpressionType.minimize, a * new_variables.y * new_variables.y
    )
    new_problem.add_expression(
        ExpressionType.subject_to, new_variables.x + new_variables.y == a - 1
    )  # noqa
    new_problem.add_expression(ExpressionType.minimize, new_variables.x == 5)
    output, cost_value = new_problem.solver().solve()
    assert cost_value == pytest.approx(expected=initial_cost_value, rel=0.1)
    assert output.x == pytest.approx(initial_output.x)
