import abc
import dataclasses
import types
from enum import Enum
from typing import Generator, List, Type

import casadi as cs

from hippopt.base.opti_solver import OptiSolver
from hippopt.base.optimization_object import OptimizationObject, TOptimizationObject
from hippopt.base.optimization_solver import OptimizationSolver


class ExpressionType(Enum):
    skip = 0
    subject_to = 1
    minimize = 2


@dataclasses.dataclass
class OptimizationProblem(abc.ABC):
    _solver: OptimizationSolver = dataclasses.field(default_factory=OptiSolver)

    def generate_optimization_objects(
        self, input_structure: OptimizationObject | List[Type[OptimizationObject]]
    ) -> TOptimizationObject | List[TOptimizationObject]:
        return self._solver.generate_optimization_objects(
            input_structure=input_structure
        )

    def add_expression(
        self, mode: ExpressionType, expression: cs.MX | Generator[cs.MX, None, None]
    ):
        if isinstance(expression, types.GeneratorType):
            for expr in expression:
                self.add_expression(mode, expr)
        else:
            match mode:
                case ExpressionType.subject_to:
                    self._solver.add_constraint(expression)
                case ExpressionType.minimize:
                    self._solver.add_cost(expression)
                case _:
                    pass

    def solver(self) -> OptimizationSolver:
        return self._solver
