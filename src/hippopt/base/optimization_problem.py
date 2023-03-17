import abc
import dataclasses
import types
from enum import Enum
from typing import Generator, List

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
    _solver: TOptimizationSolver = dataclasses.field(default_factory=OptiSolver)

    def generate_optimization_objects(
        self, input_structure: OptimizationObject | List[TOptimizationObject]
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
            assert isinstance(expression, cs.MX)
            match mode:
                case ExpressionType.subject_to:
                    # TODO Stefano: Check if it is a cost. If so, set it equal to zero
                    self._solver.add_constraint(expression)
                case ExpressionType.minimize:
                    # TODO Stefano: Check if it is a constraint. If is an equality, add the 2-norm.
                    #  If it is an inequality?
                    self._solver.add_cost(expression)
                case _:
                    pass

    def solver(self) -> TOptimizationSolver:
        return self._solver
