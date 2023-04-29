import dataclasses

import casadi as cs
import numpy as np

from hippopt import (
    ForwardEuler,
    MultipleShootingSolver,
    OptimalControlProblem,
    OptimizationObject,
    Parameter,
    StorageType,
    TimeExpansion,
    Variable,
    default_storage_field,
    dot,
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
    composite: MyTestVarMS | list[MyTestVarMS] = dataclasses.field(
        default_factory=MyTestVarMS, metadata=time_varying_metadata()
    )
    fixed: MyTestVarMS | list[MyTestVarMS] = dataclasses.field(
        default_factory=MyTestVarMS
    )
    extended: StorageType = default_storage_field(
        cls=Variable, time_expansion=TimeExpansion.Matrix
    )

    composite_list: list[MyTestVarMS] | list[list[MyTestVarMS]] = dataclasses.field(
        default=None, metadata=time_varying_metadata()
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

    problem, var = OptimalControlProblem.create(
        input_structure=MyTestVarMS(), horizon=horizon_len
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

    structure = []
    for _ in range(3):
        structure.append(MyCompositeTestVar())

    problem, var = OptimalControlProblem.create(
        input_structure=structure, horizon=horizon_len
    )

    var_flat = problem.solver().get_flattened_optimization_objects()

    assert len(var_flat) == 3
    assert len(var) == 3

    for j in range(3):
        assert var_flat[j]["composite.variable"][0] == horizon_len
        assert var_flat[j]["composite.parameter"][0] == horizon_len
        par_gen = var_flat[j]["composite.parameter"][1]()
        assert all(next(par_gen) is c.parameter for c in var[j].composite)
        variable_gen = var_flat[j]["composite.variable"][1]()
        assert all(next(variable_gen) is c.variable for c in var[j].composite)
        assert next(var_flat[j]["fixed.variable"][1]()) is var[j].fixed.variable
        assert next(var_flat[j]["fixed.parameter"][1]()) is var[j].fixed.parameter
        for i in range(3):
            assert all(isinstance(c.variable, cs.MX) for c in var[j].composite_list[i])
            variable_gen = var_flat[j]["composite_list[" + str(i) + "].variable"][1]()
            assert (
                var_flat[j]["composite_list[" + str(i) + "].variable"][0] == horizon_len
            )
            assert all(
                next(variable_gen) is c.variable for c in var[j].composite_list[i]
            )
            assert all(isinstance(c.parameter, cs.MX) for c in var[j].composite_list[i])
            parameter_gen = var_flat[j]["composite_list[" + str(i) + "].parameter"][1]()
            assert all(
                next(parameter_gen) is c.parameter for c in var[j].composite_list[i]
            )
            assert (
                var_flat[j]["composite_list[" + str(i) + "].parameter"][0]
                == horizon_len
            )
            assert (
                next(var_flat[j]["fixed_list[" + str(i) + "].variable"][1]())
                is var[j].fixed_list[i].variable
            )
            assert var_flat[j]["fixed_list[" + str(i) + "].variable"][0] == 1
            assert (
                next(var_flat[j]["fixed_list[" + str(i) + "].parameter"][1]())
                is var[j].fixed_list[i].parameter
            )
            assert var_flat[j]["fixed_list[" + str(i) + "].parameter"][0] == 1


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
    masses: list[MassFallingState] = dataclasses.field(
        metadata=time_varying_metadata(), default=None
    )
    g: StorageType = default_storage_field(Parameter)

    def __post_init__(self):
        self.g = -9.81 * np.ones(1)
        self.masses = []
        for _ in range(2):
            self.masses.append(MassFallingState())


def test_multiple_shooting():
    guess = MassFallingTestVariables()
    guess.masses = None

    problem, var = OptimalControlProblem.create(
        input_structure=MassFallingTestVariables(),
        horizon=100,
    )

    problem.add_dynamics(
        dot(["masses[0].x", "masses[0].v"])
        == (MassFallingState.get_dynamics(), {"masses[0].x": "x", "masses[0].v": "v"}),
        dt=0.01,
        integrator=ForwardEuler,
    )

    problem.add_constraint(var.masses[0][0].x == 0)
    problem.add_constraint(var.masses[0][0].v == 0)

    problem.add_dynamics(
        dot(["masses[1].x", "masses[1].v"])
        == (MassFallingState.get_dynamics(), {"masses[1].x": "x", "masses[1].v": "v"}),
        dt=0.01,
        integrator=ForwardEuler,
    )

    problem.add_constraint(var.masses[1][0].x == 0)
    problem.add_constraint(var.masses[1][0].v == 0)

    problem.set_initial_guess(guess)

    sol = problem.solve()

    # TODO Stefano: check that the output makes sense
