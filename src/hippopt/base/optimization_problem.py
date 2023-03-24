import abc
import dataclasses
import types
from enum import Enum
from typing import Generator, List

import casadi as cs

from hippopt.base.opti_solver import OptiSolver
from hippopt.base.optimization_object import OptimizationObject, TOptimizationObject
from hippopt.base.optimization_solver import OptimizationSolver, TOptimizationSolver


class ExpressionType(Enum):
    skip = 0
    subject_to = 1
    minimize = 2


@dataclasses.dataclass
class OptimizationProblem(abc.ABC):
    optimization_solver: dataclasses.InitVar[OptimizationSolver] = dataclasses.field(
        default=None
    )
    _solver: TOptimizationSolver = dataclasses.field(default=None)

    def __post_init__(self, optimization_solver: TOptimizationSolver = None):
        self._solver = (
            optimization_solver
            if isinstance(optimization_solver, OptimizationSolver)
            else OptiSolver()
        )

    def generate_optimization_objects(
        self, input_structure: OptimizationObject | List[TOptimizationObject]
    ) -> TOptimizationObject | List[TOptimizationObject]:
        return self._solver.generate_optimization_objects(
            input_structure=input_structure
        )

    def add_expression(
        self,
        mode: ExpressionType,
        expression: cs.MX | Generator[cs.MX, None, None],
        expected_value: float = 0.0,
    ):
        if isinstance(expression, types.GeneratorType):
            for expr in expression:
                self.add_expression(mode, expr)
        else:
            assert isinstance(expression, cs.MX)
            match mode:
                case ExpressionType.subject_to:
                    if (
                        expression.is_op(cs.OP_LE)
                        or expression.is_op(cs.OP_LT)
                        or expression.is_op(cs.OP_EQ)
                    ):
                        self._solver.add_constraint(expression)
                    else:
                        if not expression.is_scalar():
                            raise ValueError("The input expression is not supported.")
                        self._solver.add_constraint(
                            expression == expected_value  # noqa
                        )

                case ExpressionType.minimize:
                    if expression.is_op(cs.OP_LE) or expression.is_op(cs.OP_LT):
                        raise ValueError(
                            "The conversion from an inequality to a cost is not yet supported"
                        )
                    if expression.is_op(cs.OP_EQ):
                        error_expr = expression.dep(0) - expression.dep(1)
                        self._solver.add_cost(cs.sumsqr(error_expr))
                    else:
                        self._solver.add_cost(expression)
                case _:
                    pass

    def solver(self) -> TOptimizationSolver:
        return self._solver
