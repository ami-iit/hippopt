import abc
import dataclasses
from typing import TypeVar

import casadi as cs
import numpy as np

from hippopt.base.optimization_object import TOptimizationObject
from hippopt.base.problem import Problem

TOptimizationSolver = TypeVar("TOptimizationSolver", bound="OptimizationSolver")


class SolutionNotAvailableException(Exception):
    def __init__(self):
        super().__init__("No solution is available. Was solve() called successfully?")


class ProblemNotRegisteredException(Exception):
    def __init__(self):
        super().__init__("No problem has been registered.")


@dataclasses.dataclass
class OptimizationSolver(abc.ABC):
    @abc.abstractmethod
    def generate_optimization_objects(
        self, input_structure: TOptimizationObject | list[TOptimizationObject], **kwargs
    ) -> TOptimizationObject | list[TOptimizationObject]:
        pass

    @abc.abstractmethod
    def get_optimization_objects(
        self,
    ) -> TOptimizationObject | list[TOptimizationObject]:
        pass

    @abc.abstractmethod
    def register_problem(self, problem: Problem) -> None:
        pass

    @abc.abstractmethod
    def get_problem(self) -> Problem:
        pass

    @abc.abstractmethod
    def set_initial_guess(
        self, initial_guess: TOptimizationObject | list[TOptimizationObject]
    ) -> None:
        pass

    @abc.abstractmethod
    def get_initial_guess(self) -> TOptimizationObject | list[TOptimizationObject]:
        pass

    @abc.abstractmethod
    def solve(self) -> None:
        pass

    @abc.abstractmethod
    def get_values(self) -> TOptimizationObject | list[TOptimizationObject]:
        pass

    @abc.abstractmethod
    def get_cost_value(self) -> float:
        pass

    @abc.abstractmethod
    def add_cost(self, input_cost: cs.MX, name: str = None) -> None:
        pass

    @abc.abstractmethod
    def add_constraint(self, input_constraint: cs.MX, name: str = None) -> None:
        pass

    @abc.abstractmethod
    def get_cost_expressions(self) -> dict[str, cs.MX]:
        pass

    @abc.abstractmethod
    def get_constraint_expressions(self) -> dict[str, cs.MX]:
        pass

    @abc.abstractmethod
    def get_cost_values(self) -> dict[str, float]:
        pass

    @abc.abstractmethod
    def get_constraint_multipliers(self) -> dict[str, np.ndarray]:
        pass
