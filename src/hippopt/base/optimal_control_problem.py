import dataclasses
from typing import TypeVar

import casadi as cs

from hippopt.base.dynamics import TDynamics
from hippopt.base.multiple_shooting_solver import MultipleShootingSolver
from hippopt.base.optimal_control_solver import (
    OptimalControlSolver,
    TOptimalControlSolver,
)
from hippopt.base.problem import ExpressionType, Problem, TInputObjects

TOptimalControlProblem = TypeVar(
    "TOptimalControlProblem", bound="OptimalControlProblem"
)


@dataclasses.dataclass
class OptimalControlProblem(Problem[TOptimalControlSolver, TInputObjects]):
    optimal_control_solver: dataclasses.InitVar[
        OptimalControlSolver
    ] = dataclasses.field(default=None)

    def __post_init__(
        self,
        optimal_control_solver: TOptimalControlSolver = None,
    ):
        self._solver = (
            optimal_control_solver
            if isinstance(optimal_control_solver, OptimalControlSolver)
            else MultipleShootingSolver()
        )
        self._solver.register_problem(self)

    @classmethod
    def create(
        cls,
        input_structure: TInputObjects,
        optimal_control_solver: TOptimalControlSolver = None,
        **kwargs
    ) -> tuple[TOptimalControlProblem, TInputObjects]:
        new_problem = cls(
            optimal_control_solver=optimal_control_solver,
        )
        new_problem._solver.generate_optimization_objects(
            input_structure=input_structure, **kwargs
        )
        return new_problem, new_problem._solver.get_optimization_objects()

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
        self.solver().add_dynamics(
            dynamics=dynamics,
            x0=x0,
            t0=t0,
            mode=mode,
            name=name,
            x0_name=x0_name,
            **kwargs
        )
