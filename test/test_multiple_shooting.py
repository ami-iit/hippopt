import dataclasses

import casadi as cs
import numpy as np
import pytest

from hippopt import (
    CompositeType,
    ExpressionType,
    MultipleShootingSolver,
    OptimalControlProblem,
    OptimizationObject,
    Parameter,
    StorageType,
    TimeExpansion,
    Variable,
    default_composite_field,
    default_storage_field,
    dot,
    integrators,
    time_varying_metadata,
)


@dataclasses.dataclass
class MyTestVarMS(OptimizationObject):
    variable: StorageType = default_storage_field(Variable)
    parameter: StorageType = default_storage_field(Parameter)
    string: str = "test"

    def __post_init__(self):
        self.variable = np.zeros(3)
        self.parameter = np.zeros(3)


@dataclasses.dataclass
class MyCompositeTestVar(OptimizationObject):
    composite: CompositeType[MyTestVarMS] = default_composite_field(factory=MyTestVarMS)
    fixed: CompositeType[MyTestVarMS] = default_composite_field(
        factory=MyTestVarMS, time_varying=False
    )
    extended: StorageType = default_storage_field(
        cls=Variable, time_expansion=TimeExpansion.Matrix
    )

    composite_list: CompositeType[list[MyTestVarMS]] = default_composite_field(
        factory=list[MyTestVarMS]
    )

    fixed_list: list[MyTestVarMS] = dataclasses.field(default=None)

    def __post_init__(self):
        self.extended = np.zeros((3, 1))
        self.composite_list = []
        self.fixed_list = []
        for _ in range(3):
            self.composite_list.append(MyTestVarMS())
            self.fixed_list.append(MyTestVarMS())


def test_simple_variables_to_horizon():
    horizon_len = 10
    solver = MultipleShootingSolver()
    structure = MyTestVarMS()

    var = solver.generate_optimization_objects(structure, horizon=horizon_len)
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
    assert isinstance(var.fixed, MyTestVarMS)  # not a list
    assert var.extended.shape == (3, 10)


def test_composite_variables_custom_horizon():
    structure = []
    for _ in range(3):
        structure.append(MyCompositeTestVar())

    horizon_len = 10
    solver = MultipleShootingSolver()
    var = solver.generate_optimization_objects(
        structure, horizon=horizon_len, horizons={"fixed": horizon_len}
    )

    assert len(var) == 3

    for i in range(3):
        assert all(comp.string == "test" for comp in var[i].composite)
        assert len(var[i].composite) == horizon_len
        assert all(c.variable.shape == (3, 1) for c in var[i].composite)
        assert len(var[i].fixed) == horizon_len
        assert len(var[i].composite_list) == 3
        assert all(len(el) == horizon_len for el in var[i].composite_list)  # noqa


def test_flattened_variables_simple():
    horizon_len = 10

    problem, var, _ = OptimalControlProblem.create(
        input_structure=MyTestVarMS(), horizon=horizon_len
    )

    var_flat = problem.solver().get_flattened_optimization_objects()
    assert "string" not in var_flat
    assert var_flat["variable"][0] == horizon_len
    assert var_flat["parameter"][0] == 1
    assert next(var_flat["parameter"][1]()) is var.parameter
    assert (
        next(var_flat["parameter"][1]()) is var.parameter
    )  # check that we can use the generator twice
    variable_gen = var_flat["variable"][1]()
    assert all(next(variable_gen) is v for v in var.variable)
    variable_gen = var_flat["variable"][1]()
    assert all(
        next(variable_gen) is v for v in var.variable
    )  # check that we can use the generator twice


def test_flattened_variables_composite():
    horizon_len = 10

    structure = []
    for _ in range(3):
        structure.append(MyCompositeTestVar())

    problem, var, _ = OptimalControlProblem.create(
        input_structure=structure, horizon=horizon_len
    )

    var_flat = problem.solver().get_flattened_optimization_objects()

    assert len(var) == 3

    for j in range(3):
        assert var_flat["[" + str(j) + "].composite.variable"][0] == horizon_len
        assert var_flat["[" + str(j) + "].composite.parameter"][0] == horizon_len
        par_gen = var_flat["[" + str(j) + "].composite.parameter"][1]()
        assert all(next(par_gen) is c.parameter for c in var[j].composite)
        variable_gen = var_flat["[" + str(j) + "].composite.variable"][1]()
        assert all(next(variable_gen) is c.variable for c in var[j].composite)
        assert (
            next(var_flat["[" + str(j) + "].fixed.variable"][1]())
            is var[j].fixed.variable
        )
        assert (
            next(var_flat["[" + str(j) + "].fixed.parameter"][1]())
            is var[j].fixed.parameter
        )
        for i in range(3):
            assert all(isinstance(c.variable, cs.MX) for c in var[j].composite_list[i])
            variable_gen = var_flat[
                "[" + str(j) + "].composite_list[" + str(i) + "].variable"
            ][1]()
            assert (
                var_flat["[" + str(j) + "].composite_list[" + str(i) + "].variable"][0]
                == horizon_len
            )
            assert all(
                next(variable_gen) is c.variable for c in var[j].composite_list[i]
            )
            assert all(isinstance(c.parameter, cs.MX) for c in var[j].composite_list[i])
            parameter_gen = var_flat[
                "[" + str(j) + "].composite_list[" + str(i) + "].parameter"
            ][1]()
            assert all(
                next(parameter_gen) is c.parameter for c in var[j].composite_list[i]
            )
            assert (
                var_flat["[" + str(j) + "].composite_list[" + str(i) + "].parameter"][0]
                == horizon_len
            )
            assert (
                next(
                    var_flat["[" + str(j) + "].fixed_list[" + str(i) + "].variable"][
                        1
                    ]()
                )
                is var[j].fixed_list[i].variable
            )
            assert (
                var_flat["[" + str(j) + "].fixed_list[" + str(i) + "].variable"][0] == 1
            )
            assert (
                next(
                    var_flat["[" + str(j) + "].fixed_list[" + str(i) + "].parameter"][
                        1
                    ]()
                )
                is var[j].fixed_list[i].parameter
            )
            assert (
                var_flat["[" + str(j) + "].fixed_list[" + str(i) + "].parameter"][0]
                == 1
            )


@dataclasses.dataclass
class MassFallingState(OptimizationObject):
    x: StorageType = default_storage_field(Variable)
    v: StorageType = default_storage_field(Variable)

    def __post_init__(self):
        self.x = np.zeros(1)
        self.v = np.zeros(1)

    @staticmethod
    def get_dynamics():
        _x = cs.MX.sym("x", 1)
        _v = cs.MX.sym("v", 1)
        _g = cs.MX.sym("g", 1)

        x_dot = _v
        v_dot = _g

        return cs.Function(
            "dynamics",
            [_x, _v, _g],
            [x_dot, v_dot],
            ["x", "v", "g"],
            ["x_dot", "v_dot"],
        )


@dataclasses.dataclass
class MassFallingTestVariables(OptimizationObject):
    masses: CompositeType[list[MassFallingState]] = dataclasses.field(
        metadata=time_varying_metadata(), default=None
    )
    g: StorageType = default_storage_field(Parameter)
    foo: StorageType = default_storage_field(Variable)

    def __post_init__(self):
        self.g = -9.81
        self.masses = []
        for _ in range(3):
            self.masses.append(MassFallingState())
        self.foo = np.zeros((3, 1))


def test_multiple_shooting():
    guess = MassFallingTestVariables()
    guess.masses = None
    guess.foo = None

    horizon = 100
    dt = 0.01
    initial_position = 1.0
    initial_velocity = 0

    problem, var, symbolic = OptimalControlProblem.create(
        input_structure=MassFallingTestVariables(),
        horizon=horizon,
    )

    problem.add_dynamics(
        dot(["masses[0].x", "masses[0].v"])
        == (MassFallingState.get_dynamics(), {"masses[0].x": "x", "masses[0].v": "v"}),
        dt=dt,
        integrator=integrators.ForwardEuler,
    )

    initial_position_constraint = var.masses[0][0].x == initial_position

    problem.add_constraint(initial_position_constraint, name="initial_position")
    problem.add_constraint(var.masses[0][0].v == initial_velocity)

    problem.add_dynamics(
        dot(["masses[1].x", "masses[1].v"])
        == (MassFallingState.get_dynamics(), {"masses[1].x": "x", "masses[1].v": "v"}),
        dt=dt,
        x0={"masses[1].x": initial_position, "masses[1].v": initial_velocity},
        integrator=integrators.ForwardEuler,
        mode=ExpressionType.minimize,
        x0_name="initial_condition",
    )

    problem.add_dynamics(
        dot(symbolic.masses[2].x) == symbolic.masses[2].v,
        dt=dt,
        x0=initial_position,
        integrator=integrators.ForwardEuler,
        x0_name="initial_condition_simple_x",
    )

    problem.add_dynamics(
        dot(symbolic.masses[2].v) == ["g"],
        dt=dt,
        x0={symbolic.masses[2].v: initial_velocity},
        integrator=integrators.ForwardEuler,
        x0_name="initial_condition_simple_v",
    )

    problem.add_expression_to_horizon(
        expression=(symbolic.foo >= 5), apply_to_first_elements=False
    )

    problem.add_constraint(expression=problem.initial(symbolic.foo) == 0)

    problem.add_expression_to_horizon(
        expression=cs.sumsqr(symbolic.foo),
        apply_to_first_elements=True,
        mode=ExpressionType.minimize,
    )

    problem.set_initial_guess(guess)

    sol = problem.solve()

    assert (
        problem.get_constraint_expressions()["initial_position"]
        is initial_position_constraint
    )

    assert "initial_condition{0}" in problem.get_cost_expressions()
    assert "initial_condition{1}" in problem.get_cost_expressions()
    assert "initial_position" in sol.constraint_multipliers
    assert "initial_condition_simple_x{0}" in sol.constraint_multipliers
    assert "initial_condition_simple_v{0}" in sol.constraint_multipliers

    expected_position = initial_position
    expected_velocity = initial_velocity

    for i in range(horizon):
        assert float(sol.values.masses[0][i].x) == pytest.approx(expected_position)
        assert float(sol.values.masses[0][i].v) == pytest.approx(expected_velocity)
        assert float(sol.values.masses[1][i].x) == pytest.approx(expected_position)
        assert float(sol.values.masses[1][i].v) == pytest.approx(expected_velocity)
        assert float(sol.values.masses[2][i].x) == pytest.approx(expected_position)
        assert float(sol.values.masses[2][i].v) == pytest.approx(expected_velocity)
        if i == 0:
            assert sol.values.foo[i] == pytest.approx(0)
        else:
            assert sol.values.foo[i] == pytest.approx(5)
        expected_position += dt * expected_velocity
        expected_velocity += dt * guess.g
