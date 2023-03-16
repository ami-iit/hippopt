import abc
import dataclasses
from typing import List, TypeVar

import casadi as cs

from hippopt.base.optimization_object import OptimizationObject

TOptimizationSolver = TypeVar("TOptimizationSolver", bound="OptimizationSolver")


@dataclasses.dataclass
class OptimizationSolver(abc.ABC):
    @abc.abstractmethod
    def generate_optimization_objects(
        self, input_structure: OptimizationObject | List[OptimizationObject]
    ):
        pass

    @abc.abstractmethod
    def solve(self):
        pass

    @abc.abstractmethod
    def add_cost(self, input_cost: cs.MX):
        pass

    @abc.abstractmethod
    def add_constraint(self, input_constraint: cs.MX):
        pass

    def cost(self) -> cs.MX:
        pass
