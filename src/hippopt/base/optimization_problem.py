import dataclasses
from typing import TypeVar

from hippopt.base.opti_solver import OptiSolver
from hippopt.base.optimization_solver import OptimizationSolver, TOptimizationSolver
from hippopt.base.problem import Problem, TInputObjects

TOptimizationProblem = TypeVar("TOptimizationProblem", bound="OptimizationProblem")


@dataclasses.dataclass
class OptimizationProblemInstance:
    problem: TOptimizationProblem = dataclasses.field(default=None)
    variables: TInputObjects = dataclasses.field(default=None)

    _problem: dataclasses.InitVar[TOptimizationProblem] = dataclasses.field(
        default=None
    )
    _variables: dataclasses.InitVar[TInputObjects] = dataclasses.field(default=None)

    def __post_init__(
        self,
        _problem: TOptimizationProblem,
        _variables: TInputObjects,
    ):
        self.problem = _problem
        self.variables = _variables

    def __iter__(self):
        return iter([self.problem, self.variables])
        # Cannot use astuple here since it would perform a deepcopy
        # and would include InitVars too


@dataclasses.dataclass
class OptimizationProblem(Problem[TOptimizationSolver, TInputObjects]):
    optimization_solver: dataclasses.InitVar[OptimizationSolver] = dataclasses.field(
        default=None
    )

    def __post_init__(
        self,
        optimization_solver: TOptimizationSolver = None,
    ):
        self._solver = (
            optimization_solver
            if isinstance(optimization_solver, OptimizationSolver)
            else OptiSolver()
        )
        self._solver.register_problem(self)

    @classmethod
    def create(
        cls,
        input_structure: TInputObjects,
        optimization_solver: TOptimizationSolver = None,
        **kwargs
    ) -> OptimizationProblemInstance:
        new_problem = cls(optimization_solver=optimization_solver)
        new_problem._solver.generate_optimization_objects(
            input_structure=input_structure, **kwargs
        )
        return OptimizationProblemInstance(
            _problem=new_problem,
            _variables=new_problem._solver.get_optimization_objects(),
        )
