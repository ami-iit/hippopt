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
class OptimalControlProblemInstance:
    problem: TOptimalControlProblem = dataclasses.field(default=None)
    all_variables: TInputObjects = dataclasses.field(default=None)
    symbolic_structure: TInputObjects = dataclasses.field(default=None)

    _problem: dataclasses.InitVar[TOptimalControlProblem] = dataclasses.field(
        default=None
    )
    _all_variables: dataclasses.InitVar[TInputObjects] = dataclasses.field(default=None)
    _symbolic_structure: dataclasses.InitVar[TInputObjects] = dataclasses.field(
        default=None
    )

    def __post_init__(
        self,
        _problem: TOptimalControlProblem,
        _all_variables: TInputObjects,
        _symbolic_structure: TInputObjects,
    ):
        self.problem = _problem
        self.all_variables = _all_variables
        self.symbolic_structure = _symbolic_structure

    def __iter__(self):
        return iter([self.problem, self.all_variables, self.symbolic_structure])
        # Cannot use astuple here since it would perform a deepcopy
        # and would include InitVars too


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
    ) -> OptimalControlProblemInstance:
        new_problem = cls(
            optimal_control_solver=optimal_control_solver,
        )
        new_problem._solver.generate_optimization_objects(
            input_structure=input_structure, **kwargs
        )
        return OptimalControlProblemInstance(
            _problem=new_problem,
            _all_variables=new_problem._solver.get_optimization_objects(),
            _symbolic_structure=new_problem._solver.get_symbolic_structure(),
        )

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

    def add_expression_to_horizon(
        self,
        expression: cs.MX,
        mode: ExpressionType = ExpressionType.subject_to,
        apply_to_first_elements: bool = False,
        name: str = None,
        **kwargs
    ) -> None:
        self.solver().add_expression_to_horizon(
            expression=expression,
            mode=mode,
            apply_to_first_elements=apply_to_first_elements,
            name=name,
            **kwargs
        )
