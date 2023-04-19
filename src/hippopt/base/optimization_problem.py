import dataclasses
from typing import Tuple, TypeVar

from hippopt.base.opti_solver import OptiSolver
from hippopt.base.optimization_solver import OptimizationSolver, TOptimizationSolver
from hippopt.base.problem import Problem, TInputObjects

TOptimizationProblem = TypeVar("TOptimizationProblem", bound="OptimizationProblem")


@dataclasses.dataclass
class OptimizationProblem(Problem[TOptimizationSolver, TInputObjects]):
    input_structure: dataclasses.InitVar[TInputObjects] = dataclasses.field(
        default=None
    )
    optimization_solver: dataclasses.InitVar[OptimizationSolver] = dataclasses.field(
        default=None
    )

    def __post_init__(
        self,
        input_structure: TInputObjects,
        optimization_solver: TOptimizationSolver = None,
    ):
        self._solver = (
            optimization_solver
            if isinstance(optimization_solver, OptimizationSolver)
            else OptiSolver()
        )
        self._solver.register_problem(self)
        self._solver.generate_optimization_objects(input_structure=input_structure)

    @classmethod
    def create(
        cls,
        input_structure: TInputObjects,
        optimization_solver: TOptimizationSolver = None,
    ) -> Tuple[TOptimizationProblem, TInputObjects]:
        new_problem = cls(
            input_structure=input_structure, optimization_solver=optimization_solver
        )
        return new_problem, new_problem._solver.get_optimization_objects()
