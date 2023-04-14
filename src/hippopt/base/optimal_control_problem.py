import dataclasses
from typing import Tuple, TypeVar

from hippopt.base.dynamics import TDynamics
from hippopt.base.multiple_shooting_solver import MultipleShootingSolver
from hippopt.base.optimal_control_solver import (
    OptimalControlSolver,
    TOptimalControlSolver,
)
from hippopt.base.problem import Problem

TInputObjects = TypeVar("TInputObjects")
TOptimalControlProblem = TypeVar(
    "TOptimalControlProblem", bound="OptimalControlProblem"
)


@dataclasses.dataclass
class OptimalControlProblem(Problem[TOptimalControlSolver, TInputObjects]):
    input_structure: dataclasses.InitVar[TInputObjects] = dataclasses.field(
        default=None
    )
    optimal_control_solver: dataclasses.InitVar[
        OptimalControlSolver
    ] = dataclasses.field(default=None)

    def __post_init__(
        self,
        input_structure: TInputObjects,
        optimal_control_solver: TOptimalControlSolver = None,
    ):
        self._solver = (
            optimal_control_solver
            if isinstance(optimal_control_solver, OptimalControlSolver)
            else MultipleShootingSolver()
        )

        self._solver.generate_optimization_objects(input_structure=input_structure)

    @classmethod
    def create(
        cls,
        input_structure: TInputObjects,
        optimal_control_solver: TOptimalControlSolver = None,
    ) -> Tuple[TOptimalControlProblem, TInputObjects]:
        new_problem = cls(
            input_structure=input_structure,
            optimal_control_solver=optimal_control_solver,
        )
        return new_problem, new_problem._solver.get_optimization_objects()

    # TODO Stefano. Add the possibility to set the dynamics as cost
    def add_dynamics(self, dynamics: TDynamics, **kwargs) -> None:
        self.solver().add_dynamics(dynamics, **kwargs)
