import abc
import dataclasses
import types
from enum import Enum
from typing import Generator, Generic, TypeVar

import casadi as cs
import numpy as np

from hippopt.base.optimization_object import TOptimizationObject

TGenericOptimizationObject = TypeVar("TGenericOptimizationObject")
TGenericSolver = TypeVar("TGenericSolver")
TInputObjects = TypeVar("TInputObjects")


class ProblemNotSolvedException(Exception):
    def __init__(self):
        super().__init__("No solution is available. Was solve() called successfully?")


class ExpressionType(Enum):
    skip = 0
    subject_to = 1
    minimize = 2


@dataclasses.dataclass
class Output(Generic[TGenericOptimizationObject]):
    values: TGenericOptimizationObject = dataclasses.field(default=None)
    cost_value: float = dataclasses.field(default=None)
    cost_values: dict[str, float] = dataclasses.field(default=None)
    constraint_values: dict[str, np.ndarray] = dataclasses.field(default=None)

    _values: dataclasses.InitVar[TGenericOptimizationObject] = dataclasses.field(
        default=None
    )
    _cost_value: dataclasses.InitVar[float] = dataclasses.field(default=None)
    _cost_values: dataclasses.InitVar[dict[str, np.ndarray]] = dataclasses.field(
        default=None
    )
    _constraint_values: dataclasses.InitVar[dict[str, np.ndarray]] = dataclasses.field(
        default=None
    )

    def __post_init__(
        self,
        _values: TGenericOptimizationObject,
        _cost_value: float,
        _cost_values: dict[str, float],
        _constraint_values: dict[str, np.ndarray],
    ):
        self.values = _values
        self.cost_value = _cost_value
        self.cost_values = _cost_values
        self.constraint_values = _constraint_values


@dataclasses.dataclass
class Problem(abc.ABC, Generic[TGenericSolver, TInputObjects]):
    _solver: TGenericSolver = dataclasses.field(default=None)
    _output: Output[TInputObjects] = dataclasses.field(default=None)

    def set_initial_guess(
        self, initial_guess: TOptimizationObject | list[TOptimizationObject]
    ) -> None:
        self.solver().set_initial_guess(initial_guess)

    def add_cost(
        self,
        expression: cs.MX | Generator[cs.MX, None, None],
        scaling: float | cs.MX = 1.0,
        name: str = None,
    ) -> None:
        if isinstance(expression, types.GeneratorType):
            i = 0
            for expr in expression:
                input_name = name
                if input_name is not None:
                    input_name = input_name + "{" + str(i) + "}"
                self.add_cost(expression=expr, scaling=scaling, name=input_name)
                i += 1
        else:
            assert isinstance(expression, cs.MX)
            if expression.is_op(cs.OP_LE) or expression.is_op(cs.OP_LT):
                raise ValueError(
                    "The conversion from an inequality to a cost is not yet supported"
                )
            if expression.is_op(cs.OP_EQ):
                error_expr = expression.dep(0) - expression.dep(1)
                self.solver().add_cost(
                    input_cost=scaling * cs.sumsqr(error_expr), name=name
                )
            else:
                self.solver().add_cost(input_cost=scaling * expression, name=name)

    def add_constraint(
        self,
        expression: cs.MX | Generator[cs.MX, None, None],
        expected_value: float | cs.MX = 0.0,
        name: str = None,
    ) -> None:
        if isinstance(expression, types.GeneratorType):
            i = 0
            for expr in expression:
                input_name = name
                if input_name is not None:
                    input_name = input_name + "{" + str(i) + "}"
                self.add_constraint(
                    expression=expr, expected_value=expected_value, name=input_name
                )
                i += 1
        else:
            assert isinstance(expression, cs.MX)
            if (
                expression.is_op(cs.OP_LE)
                or expression.is_op(cs.OP_LT)
                or expression.is_op(cs.OP_EQ)
            ):
                self.solver().add_constraint(input_constraint=expression, name=name)
            else:
                if not expression.is_scalar():
                    raise ValueError("The input expression is not supported.")
                self.solver().add_constraint(
                    input_constraint=expression == expected_value, name=name
                )

    def add_expression(
        self,
        mode: ExpressionType,
        expression: cs.MX | Generator[cs.MX, None, None],
        name: str = None,
        **kwargs,
    ) -> None:
        if isinstance(expression, types.GeneratorType):
            i = 0
            for expr in expression:
                input_name = name
                if input_name is not None:
                    input_name = input_name + "{" + str(i) + "}"
                self.add_expression(mode=mode, expression=expr, name=input_name)
                i += 1
        else:
            assert isinstance(expression, cs.MX)
            match mode:
                case ExpressionType.subject_to:
                    self.add_constraint(expression=expression, name=name, **kwargs)

                case ExpressionType.minimize:
                    self.add_cost(expression=expression, name=name, **kwargs)
                case _:
                    pass

    def solver(self) -> TGenericSolver:
        return self._solver

    def solve(self) -> Output:
        self.solver().solve()
        self._output = Output(
            _cost_value=self.solver().get_cost_value(),
            _values=self.solver().get_values(),
            _cost_values=self.solver().get_cost_values(),
            _constraint_values=self.solver().get_constraint_values(),
        )
        return self._output

    def get_output(self) -> Output:
        if self._output is None:
            raise ProblemNotSolvedException

        return self._output


# TODO: Add possibility to get cost and constraints expressions
