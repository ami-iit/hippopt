import dataclasses
import math

import casadi as cs
import pytest

from hippopt import ForwardEuler, ImplicitTrapezoid, dot, step


def get_test_function() -> cs.Function:
    _x = cs.MX.sym("x", 1)
    _lambda = cs.MX.sym("lambda", 1)

    _x_dot = _lambda * _x

    return cs.Function("test", [_x, _lambda], [_x_dot], ["x", "lambda"], ["x_dot"])


@dataclasses.dataclass
class Vars:
    x: cs.MX = dataclasses.field(default_factory=cs.MX)
    lam: cs.MX = dataclasses.field(default_factory=cs.MX)


def test_simple_dynamics_creation_with_operator():
    _x = cs.MX.sym("x", 1)
    _x_dot = -2.0 * _x

    f = cs.Function("test", [_x], [_x_dot], ["x"], ["x_dot"])

    dynamics = dot("x") == f  # noqa

    assert dynamics.state_variables() == ["x"]
    assert dynamics.input_names() == ["x"]


def test_dynamics_creation_with_operator():
    f = get_test_function()

    dynamics = dot("x") == (f, {"lam": "lambda"})  # noqa

    assert dynamics.state_variables() == ["x"]
    assert dynamics.input_names() == ["x", "lam"]


def test_dynamics_creation():
    f = get_test_function()

    dynamics = dot("x", t="time").equal(f, {"lam": "lambda"})

    assert dynamics.state_variables() == ["x"]
    assert dynamics.input_names() == ["x", "lam"]
    assert dynamics.time_name() == "time"


def test_forward_euler():
    f = get_test_function()

    dynamics = dot("x") == (f, {"lam": "lambda"})  # noqa

    inputs = Vars()

    inputs.lam = 1.0
    inputs.x = 0.5

    input_dict = dataclasses.asdict(inputs)

    dt = cs.MX(0.005)

    integrated = step(
        ForwardEuler, dynamics=dynamics, x0=input_dict, xf=input_dict, dt=dt
    )

    expected = inputs.x * math.exp(inputs.lam * dt)

    assert float(integrated["x"]) == pytest.approx(float(expected), rel=1e-4)


def test_implicit_trapezoid():
    f = get_test_function()

    dynamics = dot("x") == (f, {"lam": "lambda"})  # noqa

    inputs = Vars()

    inputs.lam = 1.0
    inputs.x = 0.5

    input_dict = dataclasses.asdict(inputs)

    dt = cs.MX(0.005)

    integrated = step(
        ImplicitTrapezoid,
        dynamics=dynamics,
        x0=input_dict,
        xf=input_dict,
        dt=dt,
        t0=cs.MX(0.0),
    )

    expected = inputs.x * math.exp(inputs.lam * dt)

    assert float(integrated["x"]) == pytest.approx(float(expected), rel=1e-4)


def test_forward_euler_multiple_time_dynamics():
    _x = cs.MX.sym("x", 1)
    _lambda = cs.MX.sym("lambda", 1)
    _t = cs.MX.sym("time", 1)
    _x_dot = _lambda * _x

    f = cs.Function(
        "test",
        [_x, _lambda, _t],
        [_x_dot, _t],
        ["x", "lambda", "time"],
        ["x_dot", "time_out"],
    )

    dynamics = dot(["x", ""], t="time") == (f, {"lam": "lambda"})  # noqa

    inputs = Vars()

    inputs.lam = 1.0
    inputs.x = 0.5

    input_dict = dataclasses.asdict(inputs)

    dt = cs.MX(0.005)

    integrated = step(
        ForwardEuler, dynamics=dynamics, x0=input_dict, xf=input_dict, dt=dt
    )

    expected = inputs.x * math.exp(inputs.lam * dt)

    assert float(integrated["x"]) == pytest.approx(float(expected), rel=1e-4)
