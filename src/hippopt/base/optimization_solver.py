import abc
import dataclasses
from typing import List, TypeVar

import casadi as cs

from hippopt.base.optimization_object import TOptimizationObject
from hippopt.base.problem import Problem

TOptimizationSolver = TypeVar("TOptimizationSolver", bound="OptimizationSolver")


class SolutionNotAvailableException(Exception):
    def __init__(self):
        super().__init__("No solution is available. Was solve() called successfully?")


@dataclasses.dataclass
class OptimizationSolver(abc.ABC):
    @abc.abstractmethod
    def generate_optimization_objects(
        self, input_structure: TOptimizationObject | List[TOptimizationObject], **kwargs
    ) -> TOptimizationObject | List[TOptimizationObject]:
        pass

    @abc.abstractmethod
    def get_optimization_objects(
        self,
    ) -> TOptimizationObject | List[TOptimizationObject]:
        pass

    @abc.abstractmethod
    def register_problem(self, problem: Problem) -> None:
        pass

    @abc.abstractmethod
    def get_problem(self) -> Problem:
        pass

    @abc.abstractmethod
    def set_initial_guess(
        self, initial_guess: TOptimizationObject | List[TOptimizationObject]
    ) -> None:
        pass

    @abc.abstractmethod
    def solve(self) -> None:
        pass

    @abc.abstractmethod
    def get_values(self) -> TOptimizationObject | List[TOptimizationObject]:
        pass

    @abc.abstractmethod
    def get_cost_value(self) -> float:
        pass

    @abc.abstractmethod
    def add_cost(self, input_cost: cs.MX) -> None:
        pass

    @abc.abstractmethod
    def add_constraint(self, input_constraint: cs.MX) -> None:
        pass
