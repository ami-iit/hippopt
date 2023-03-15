import abc
import dataclasses
from typing import List, Type, TypeVar

import casadi as cs

from hippopt.base.optimization_object import OptimizationObject

TOptimizationSolver = TypeVar("TOptimizationSolver", bound="OptimizationSolver")


@dataclasses.dataclass
class OptimizationSolver(abc.ABC):
    @abc.abstractmethod
    def generate_optimization_objects(
        self, input_structure: Type[OptimizationObject] | List[Type[OptimizationObject]]
    ):
        pass

    @abc.abstractmethod
    def solve(self):
        pass

    @abc.abstractmethod
    def add_cost(self, input_cost: cs.Function):
        pass

    @abc.abstractmethod
    def add_constraint(self, input_constraint: cs.Function):
        pass
