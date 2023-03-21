import dataclasses
from typing import List, Tuple

import casadi as cs

from .opti_solver import OptiSolver
from .optimal_control_solver import OptimalControlSolver
from .optimization_object import TOptimizationObject
from .optimization_solver import OptimizationSolver, TOptimizationSolver


@dataclasses.dataclass
class MultipleShootingSolver(OptimalControlSolver):
    optimization_solver: dataclasses.InitVar[OptimizationSolver] = dataclasses.field(
        default=None
    )
    _optimization_solver: TOptimizationSolver = dataclasses.field(default=None)

    def __post_init__(self, optimization_solver: OptimizationSolver = None):
        self._optimization_solver = (
            optimization_solver
            if isinstance(optimization_solver, OptimizationSolver)
            else OptiSolver()
        )

    def generate_optimization_objects(
        self, input_structure: TOptimizationObject | List[TOptimizationObject], **kwargs
    ) -> TOptimizationObject | List[TOptimizationObject]:
        pass

    def set_initial_guess(
        self, initial_guess: TOptimizationObject | List[TOptimizationObject]
    ):
        self._optimization_solver.set_initial_guess(initial_guess=initial_guess)

    def solve(self) -> Tuple[TOptimizationObject, float]:
        return self._optimization_solver.solve()

    def get_solution(self) -> TOptimizationObject | List[TOptimizationObject] | None:
        return self._optimization_solver.get_solution()

    def get_cost_value(self) -> float | None:
        return self._optimization_solver.get_cost_value()

    def add_cost(self, input_cost: cs.MX):
        self._optimization_solver.add_cost(input_cost=input_cost)

    def add_constraint(self, input_constraint: cs.MX):
        self._optimization_solver.add_constraint(input_constraint=input_constraint)

    def cost_function(self) -> cs.MX:
        return self._optimization_solver.cost_function()
