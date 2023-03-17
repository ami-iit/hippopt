import abc
import dataclasses
from typing import List, Tuple, TypeVar

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
    def set_initial_guess(
        self, initial_guess: OptimizationObject | List[OptimizationObject]
    ):
        pass

    @abc.abstractmethod
    def solve(self) -> Tuple[OptimizationObject, float]:
        pass

    @abc.abstractmethod
    def get_solution(self) -> OptimizationObject | List[OptimizationObject] | None:
        pass

    @abc.abstractmethod
    def get_cost_value(self) -> float | None:
        pass

    @abc.abstractmethod
    def add_cost(self, input_cost: cs.MX):
        pass

    @abc.abstractmethod
    def add_constraint(self, input_constraint: cs.MX):
        pass

    def cost_function(self) -> cs.MX:
        pass
