import abc
import dataclasses
from enum import Enum
from functools import singledispatchmethod
from typing import Generator, List, Type

import casadi as cs

from hippopt.base.opti_solver import OptiSolver
from hippopt.base.optimization_object import OptimizationObject, TOptimizationObject
from hippopt.base.optimization_solver import TOptimizationSolver


class ExpressionType(Enum):
    skip = 0
    subject_to = 1
    minimize = 2


@dataclasses.dataclass
class OptimizationProblem(abc.ABC):
    _solver: TOptimizationSolver = dataclasses.field(default=OptiSolver)

    def generate_optimization_objects(
        self, input_structure: Type[OptimizationObject] | List[Type[OptimizationObject]]
    ) -> TOptimizationObject | List[TOptimizationObject]:
        return self._solver.generate_optimization_objects(input_structure)

    @singledispatchmethod
    def add_expression(
        self, mode: ExpressionType, expression: cs.MX | Generator[cs.MX]
    ):
        pass

    @add_expression.register
    def add_expression(self, mode: ExpressionType, expression: cs.MX):
        match mode:
            case ExpressionType.subject_to:
                self._solver.add_cost(expression)
            case ExpressionType.minimize:
                self._solver.add_constraint(expression)
            case _:
                pass

    @add_expression.register
    def add_expression(self, mode: ExpressionType, expressions: Generator[cs.MX]):
        for expr in expressions:
            self.add_expression(mode, expr)
