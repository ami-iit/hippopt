import abc
import dataclasses
from typing import TypeVar

import casadi as cs

from .dynamics import TDynamics
from .optimization_object import TOptimizationObject
from .optimization_solver import OptimizationSolver
from .problem import ExpressionType

TOptimalControlSolver = TypeVar("TOptimalControlSolver", bound="OptimalControlSolver")


@dataclasses.dataclass
class OptimalControlSolver(OptimizationSolver):
    @abc.abstractmethod
    def add_dynamics(
        self,
        dynamics: TDynamics,
        x0: dict[str, cs.MX] | dict[cs.MX, cs.MX] | cs.MX = None,
        t0: cs.MX = cs.MX(0.0),
        mode: ExpressionType = ExpressionType.subject_to,
        name: str = None,
        x0_name: str = None,
        **kwargs
    ) -> None:
        pass

    @abc.abstractmethod
    def get_symbolic_structure(self) -> TOptimizationObject | list[TOptimizationObject]:
        pass

    @abc.abstractmethod
    def add_expression_to_horizon(
        self,
        expression: cs.MX,
        mode: ExpressionType = ExpressionType.subject_to,
        apply_to_first_elements: bool = False,
        name: str = None,
        **kwargs
    ) -> None:
        pass

    @abc.abstractmethod
    def initial(self, variable: str | cs.MX) -> cs.MX:
        pass

    @abc.abstractmethod
    def final(self, variable: str | cs.MX) -> cs.MX:
        pass
