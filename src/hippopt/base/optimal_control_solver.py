import abc
import dataclasses
from typing import TypeVar

from .dynamics import TDynamics
from .optimization_solver import OptimizationSolver

TOptimalControlSolver = TypeVar("TOptimalControlSolver", bound="OptimalControlSolver")


@dataclasses.dataclass
class OptimalControlSolver(OptimizationSolver):
    @abc.abstractmethod
    def add_dynamics(self, time_derivative: TDynamics, **kwargs):
        pass
