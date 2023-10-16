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
        x0: dict[str, cs.MX] = None,
        t0: cs.MX = cs.MX(0.0),
        mode: ExpressionType = ExpressionType.subject_to,
        name: str = None,
        x0_name: str = None,
        **kwargs
    ) -> None:
        pass
