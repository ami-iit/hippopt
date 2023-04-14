import abc
import dataclasses
import types
from enum import Enum
from typing import Generator, Generic, TypeVar

import casadi as cs

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
    cost_value: float = None

    _values: dataclasses.InitVar[TGenericOptimizationObject] = dataclasses.field(
        default=None
    )
    _cost_value: dataclasses.InitVar[float] = dataclasses.field(default=None)

    def __post_init__(self, _values: TGenericOptimizationObject, _cost_value: float):
        self.values = _values
        self.cost_value = _cost_value


@dataclasses.dataclass
class Problem(abc.ABC, Generic[TGenericSolver, TInputObjects]):
    _solver: TGenericSolver = dataclasses.field(default=None)
    _output: Output[TInputObjects] = dataclasses.field(default=None)

    def add_cost(
        self,
        expression: cs.MX | Generator[cs.MX, None, None],
        scaling: float | cs.MX = 1.0,
    ) -> None:
        if isinstance(expression, types.GeneratorType):
            for expr in expression:
                self.add_cost(expr, scaling)
        else:
            assert isinstance(expression, cs.MX)
            if expression.is_op(cs.OP_LE) or expression.is_op(cs.OP_LT):
                raise ValueError(
                    "The conversion from an inequality to a cost is not yet supported"
                )
            if expression.is_op(cs.OP_EQ):
                error_expr = expression.dep(0) - expression.dep(1)
                self.solver().add_cost(scaling * cs.sumsqr(error_expr))
            else:
                self.solver().add_cost(scaling * expression)  # noqa

    def add_constraint(
        self,
        expression: cs.MX | Generator[cs.MX, None, None],
        expected_value: float | cs.MX = 0.0,
    ) -> None:
        if isinstance(expression, types.GeneratorType):
            for expr in expression:
                self.add_constraint(expr, expected_value)
        else:
            assert isinstance(expression, cs.MX)
            if (
                expression.is_op(cs.OP_LE)
                or expression.is_op(cs.OP_LT)
                or expression.is_op(cs.OP_EQ)
            ):
                self.solver().add_constraint(expression)
            else:
                if not expression.is_scalar():
                    raise ValueError("The input expression is not supported.")
                self.solver().add_constraint(expression == expected_value)  # noqa

    def add_expression(
        self,
        mode: ExpressionType,
        expression: cs.MX | Generator[cs.MX, None, None],
        **kwargs,
    ) -> None:
        if isinstance(expression, types.GeneratorType):
            for expr in expression:
                self.add_expression(mode, expr)
        else:
            assert isinstance(expression, cs.MX)
            match mode:
                case ExpressionType.subject_to:
                    self.add_constraint(expression, **kwargs)

                case ExpressionType.minimize:
                    self.add_cost(expression, **kwargs)
                case _:
                    pass

    def solver(self) -> TGenericSolver:
        return self._solver

    def solve(self) -> Output:
        self.solver().solve()
        self._output = Output(
            _cost_value=self.solver().get_cost_value(),
            _values=self.solver().get_values(),
        )
        return self._output

    def get_output(self) -> Output:
        if self._output is None:
            raise ProblemNotSolvedException

        return self._output


# TODO Stefano: Add possibility to get the task list
