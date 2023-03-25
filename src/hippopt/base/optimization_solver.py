import abc
import dataclasses
from typing import Generic, List, Tuple, TypeVar

import casadi as cs

from hippopt.base.optimization_object import TOptimizationObject

TOptimizationSolver = TypeVar("TOptimizationSolver", bound="OptimizationSolver")
TGenericOptimizationObject = TypeVar("TGenericOptimizationObject")


@dataclasses.dataclass
class SolverOutput(Generic[TGenericOptimizationObject]):
    values: TGenericOptimizationObject = dataclasses.field(default=None)
    cost_value: float = None

    _values: dataclasses.InitVar[TGenericOptimizationObject] = dataclasses.field(
        default=None
    )
    _cost_value: dataclasses.InitVar[float] = dataclasses.field(default=None)

    def __post_init__(self, _values: TGenericOptimizationObject, _cost_value: float):
        self.values = _values
        self.cost_value = _cost_value


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
    def set_initial_guess(
        self, initial_guess: TOptimizationObject | List[TOptimizationObject]
    ) -> None:
        pass

    @abc.abstractmethod
    def solve(self) -> SolverOutput:
        pass

    @abc.abstractmethod
    def get_solution(self) -> TOptimizationObject | List[TOptimizationObject]:
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
