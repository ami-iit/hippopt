import dataclasses
from typing import List

from hippopt.base.dynamics import TDynamics
from hippopt.base.multiple_shooting_solver import MultipleShootingSolver
from hippopt.base.optimal_control_solver import (
    OptimalControlSolver,
    TOptimalControlSolver,
)
from hippopt.base.optimization_object import OptimizationObject, TOptimizationObject
from hippopt.base.optimization_solver import TOptimizationSolver
from hippopt.base.problem import Problem


@dataclasses.dataclass
class OptimalControlProblem(Problem):
    input_structure: dataclasses.InitVar[OptimizationObject | List[TOptimizationObject]]
    optimal_control_solver: dataclasses.InitVar[
        OptimalControlSolver
    ] = dataclasses.field(default=None)
    _solver: TOptimalControlSolver = dataclasses.field(default=None)

    def __post_init__(
        self,
        input_structure: OptimizationObject | List[TOptimizationObject],
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
        input_structure: TOptimizationObject | List[TOptimizationObject],
        optimal_control_solver: TOptimalControlSolver = None,
    ):
        new_problem = cls(
            input_structure=input_structure,
            optimal_control_solver=optimal_control_solver,
        )
        return new_problem, new_problem._solver.get_optimization_objects()

    def add_dynamics(self, time_derivative: TDynamics, **kwargs):
        self.solver().add_dynamics(time_derivative, **kwargs)

    def solver(self) -> TOptimizationSolver | TOptimalControlSolver:
        return self._solver
