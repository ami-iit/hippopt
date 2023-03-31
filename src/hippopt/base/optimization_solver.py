import abc
import dataclasses
from typing import List, Tuple, TypeVar

import casadi as cs

from hippopt.base.optimization_object import TOptimizationObject

TOptimizationSolver = TypeVar("TOptimizationSolver", bound="OptimizationSolver")


@dataclasses.dataclass
class OptimizationSolver(abc.ABC):
    @abc.abstractmethod
    def generate_optimization_objects(
        self, input_structure: TOptimizationObject | List[TOptimizationObject], **kwargs
    ) -> TOptimizationObject | List[TOptimizationObject]:
        pass

    @abc.abstractmethod
    def set_initial_guess(
        self, initial_guess: TOptimizationObject | List[TOptimizationObject]
    ) -> None:
        pass

    @abc.abstractmethod
    def solve(self) -> Tuple[TOptimizationObject, float]:
        pass

    @abc.abstractmethod
    def get_solution(self) -> TOptimizationObject | List[TOptimizationObject] | None:
        pass

    @abc.abstractmethod
    def get_cost_value(self) -> float | None:
        pass

    @abc.abstractmethod
    def add_cost(self, input_cost: cs.MX) -> None:
        pass

    @abc.abstractmethod
    def add_constraint(self, input_constraint: cs.MX) -> None:
        pass

    @abc.abstractmethod
    def cost_function(self) -> cs.MX:
        pass
