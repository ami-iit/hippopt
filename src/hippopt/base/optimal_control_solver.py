import abc
import dataclasses
from typing import TypeVar

import casadi as cs

from .dynamics import TDynamics
from .optimization_solver import OptimizationSolver
from .problem import ExpressionType

TOptimalControlSolver = TypeVar("TOptimalControlSolver", bound="OptimalControlSolver")


@dataclasses.dataclass
class OptimalControlSolver(OptimizationSolver):
    @abc.abstractmethod
    def add_dynamics(
        self,
        dynamics: TDynamics,
        t0: cs.MX = cs.MX(0.0),
        mode: ExpressionType = ExpressionType.subject_to,
        **kwargs
    ) -> None:
        pass
