import copy
import dataclasses
from collections.abc import Iterable
from typing import Any, ClassVar, List, Tuple

import casadi as cs
import numpy as np

from hippopt.base.optimization_object import OptimizationObject, TOptimizationObject
from hippopt.base.optimization_solver import OptimizationSolver
from hippopt.base.parameter import Parameter
from hippopt.base.variable import Variable


@dataclasses.dataclass
class OptiSolver(OptimizationSolver):
    DefaultSolverType: ClassVar[str] = "ipopt"
    _inner_solver: str = dataclasses.field(default=DefaultSolverType)
    _problem_type: dataclasses.InitVar[str] = dataclasses.field(default="nlp")

    _options_plugin: dict[str, Any] = dataclasses.field(default_factory=dict)
    _options_solver: dict[str, Any] = dataclasses.field(default_factory=dict)

    _cost: cs.MX = dataclasses.field(default=None)
    _solver: cs.Opti = dataclasses.field(default=None)
    _opti_solution: cs.OptiSol = dataclasses.field(default=None)
    _output_solution: TOptimizationObject | List[
        TOptimizationObject
    ] = dataclasses.field(default=None)
    _output_cost: float = dataclasses.field(default=None)
    _variables: TOptimizationObject | List[TOptimizationObject] = dataclasses.field(
        default=None
    )

    def __post_init__(self, _problem_type: str) -> None:
        self._solver = cs.Opti(_problem_type)
        self._solver.solver(
            self._inner_solver, self._options_plugin, self._options_solver
        )

    def _generate_objects_from_instance(
        self, input_structure: OptimizationObject
    ) -> OptimizationObject:
        output = copy.deepcopy(input_structure)

        for field in dataclasses.fields(output):
            has_storage_field = OptimizationObject.StorageTypeField in field.metadata

            if (
                has_storage_field
                and field.metadata[OptimizationObject.StorageTypeField]
                == Variable.StorageType
            ):
                value = output.__dict__[field.name]
                value = (
                    value
                    if not isinstance(value, np.ndarray)
                    else np.expand_dims(value, axis=1)
                )
                output.__setattr__(field.name, self._solver.variable(*value.shape))
                continue

            if (
                has_storage_field
                and field.metadata[OptimizationObject.StorageTypeField]
                == Parameter.StorageType
            ):
                value = output.__dict__[field.name]
                value = (
                    value
                    if not isinstance(value, np.ndarray)
                    else np.expand_dims(value, axis=1)
                )
                output.__setattr__(field.name, self._solver.parameter(*value.shape))
                continue

            composite_value = output.__getattribute__(field.name)

            is_iterable = isinstance(composite_value, Iterable)
            list_of_optimization_objects = is_iterable and all(
                isinstance(elem, OptimizationObject) for elem in composite_value
            )

            if (
                isinstance(composite_value, OptimizationObject)
                or list_of_optimization_objects
            ):
                output.__setattr__(
                    field.name, self.generate_optimization_objects(composite_value)
                )

        self._variables = output
        return output

    def _generate_objects_from_list(
        self, input_structure: list
    ) -> List[OptimizationObject]:
        list_of_optimization_objects = isinstance(input_structure, Iterable) and all(
            isinstance(elem, OptimizationObject) for elem in input_structure
        )

        assert (
            isinstance(input_structure, OptimizationObject)
            or list_of_optimization_objects
        )

        output = copy.deepcopy(input_structure)
        for i in range(len(output)):
            output[i] = self.generate_optimization_objects(output[i])

        self._variables = output
        return output

    def _generate_solution_output(
        self, variables: OptimizationObject | List[OptimizationObject]
    ) -> OptimizationObject | List[OptimizationObject]:
        output = copy.deepcopy(variables)

        if isinstance(variables, Iterable):
            i = 0
            for element in variables:
                output[i] = self._generate_solution_output(element)

            return output

        for field in dataclasses.fields(variables):
            has_storage_field = OptimizationObject.StorageTypeField in field.metadata

            if (
                has_storage_field
                and (
                    field.metadata[OptimizationObject.StorageTypeField]
                    == Variable.StorageType
                )
                or (
                    field.metadata[OptimizationObject.StorageTypeField]
                    == Parameter.StorageType
                )
            ):
                var = variables.__dict__[field.name]
                output.__setattr__(field.name, self._opti_solution.value(var))
                continue

            composite_variable = variables.__getattribute__(field.name)

            is_iterable = isinstance(composite_variable, Iterable)
            list_of_optimization_objects = is_iterable and all(
                isinstance(elem, OptimizationObject) for elem in composite_variable
            )

            if (
                isinstance(composite_variable, OptimizationObject)
                or list_of_optimization_objects
            ):
                output.__setattr__(
                    field.name, self._generate_solution_output(composite_variable)
                )

        return output

    def generate_optimization_objects(
        self, input_structure: OptimizationObject | List[OptimizationObject]
    ):
        if isinstance(input_structure, OptimizationObject):
            return self._generate_objects_from_instance(input_structure=input_structure)
        return self._generate_objects_from_list(input_structure=input_structure)

    def get_optimization_objects(
        self,
    ) -> TOptimizationObject | List[TOptimizationObject]:
        return self._variables

    def set_opti_options(
        self,
        inner_solver: str = None,
        options_plugin: dict[str, Any] = None,
        options_solver: dict[str, Any] = None,
    ):
        if inner_solver is not None:
            self._inner_solver = inner_solver
        if options_plugin is not None:
            self._options_plugin = options_plugin
        if options_solver is not None:
            self._options_solver = options_solver

        self._solver.solver(
            self._inner_solver, self._options_plugin, self._options_solver
        )

    def solve(self) -> Tuple[OptimizationObject, float]:
        self._solver.minimize(self._cost)
        self._opti_solution = self._solver.solve()
        self._output_cost = self._opti_solution.value(self._cost)
        self._output_solution = self._generate_solution_output(self._variables)
        return self._output_solution, self._output_cost

    def get_solution(self) -> OptimizationObject | List[OptimizationObject] | None:
        return self._output_solution

    def get_cost_value(self) -> float | None:
        return self._output_cost

    def add_cost(self, input_cost: cs.MX):
        # TODO Stefano: Check if it is a constraint. If is an equality, add the 2-norm. If it is an inequality?
        if self._cost is None:
            self._cost = input_cost
            return

        self._cost += input_cost

    def add_constraint(self, input_constraint: cs.MX):
        # TODO Stefano: Check if it is a cost. If so, set it equal to zero
        self._solver.subject_to(input_constraint)

    def cost_function(self) -> cs.MX:
        return self._cost
