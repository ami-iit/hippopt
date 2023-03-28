import dataclasses
from typing import List

from hippopt.base.opti_solver import OptiSolver
from hippopt.base.optimization_object import OptimizationObject, TOptimizationObject
from hippopt.base.optimization_solver import OptimizationSolver, TOptimizationSolver
from hippopt.base.problem import Problem


@dataclasses.dataclass
class OptimizationProblem(Problem):
    input_structure: dataclasses.InitVar[OptimizationObject | List[TOptimizationObject]]
    optimization_solver: dataclasses.InitVar[OptimizationSolver] = dataclasses.field(
        default=None
    )
    _solver: TOptimizationSolver = dataclasses.field(default=None)

    def __post_init__(
        self,
        input_structure: OptimizationObject | List[TOptimizationObject],
        optimization_solver: TOptimizationSolver = None,
    ):
        self._solver = (
            optimization_solver
            if isinstance(optimization_solver, OptimizationSolver)
            else OptiSolver()
        )

        self._solver.generate_optimization_objects(input_structure=input_structure)

    @classmethod
    def create(
        cls,
        input_structure: TOptimizationObject | List[TOptimizationObject],
        optimization_solver: TOptimizationSolver = None,
    ):
        new_problem = cls(
            input_structure=input_structure, optimization_solver=optimization_solver
        )
        return new_problem, new_problem._solver.get_optimization_objects()

    def solver(self) -> TOptimizationSolver:
        return self._solver
