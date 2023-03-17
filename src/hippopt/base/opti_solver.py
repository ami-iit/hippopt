import copy
import dataclasses
from typing import Any, ClassVar, List, Tuple

import casadi as cs
import numpy as np

from hippopt.base.continuous_variable import ContinuousVariable
from hippopt.base.optimization_object import OptimizationObject, TOptimizationObject
from hippopt.base.optimization_solver import OptimizationSolver
from hippopt.base.parameter import Parameter


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
                == ContinuousVariable.StorageType
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

            is_list = isinstance(composite_value, list)
            list_of_optimization_objects = is_list and all(
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
        list_of_optimization_objects = isinstance(input_structure, list) and all(
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

        if isinstance(variables, list):
            i = 0
            for element in variables:
                output[i] = self._generate_solution_output(element)
                i += 1

            return output

        for field in dataclasses.fields(variables):
            has_storage_field = OptimizationObject.StorageTypeField in field.metadata

            if has_storage_field and (
                (
                    field.metadata[OptimizationObject.StorageTypeField]
                    == ContinuousVariable.StorageType
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

            is_list = isinstance(composite_variable, list)
            list_of_optimization_objects = is_list and all(
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

    def _set_initial_guess_internal(
        self,
        initial_guess: OptimizationObject,
        corresponding_variable: OptimizationObject,
    ):
        for field in dataclasses.fields(initial_guess):
            has_storage_field = OptimizationObject.StorageTypeField in field.metadata

            if (
                has_storage_field
                and field.metadata[OptimizationObject.StorageTypeField]
                == ContinuousVariable.StorageType
            ):
                guess = initial_guess.__dict__[field.name]

                if guess is None:
                    continue

                if not isinstance(guess, np.ndarray):
                    raise ValueError(
                        "The guess for the field "
                        + field.name
                        + " is not an numpy array."
                    )

                if not hasattr(corresponding_variable, field.name):
                    raise ValueError(
                        "The guess has the field "
                        + field.name
                        + " but it is not present in the optimization variables"
                    )

                self._solver.set_initial(
                    corresponding_variable.__getattribute__(field.name), guess
                )
                continue

            if (
                has_storage_field
                and field.metadata[OptimizationObject.StorageTypeField]
                == Parameter.StorageType
            ):
                guess = initial_guess.__dict__[field.name]

                if guess is None:
                    continue

                if not isinstance(guess, np.ndarray):
                    raise ValueError(
                        "The guess for the field "
                        + field.name
                        + " is not an numpy array."
                    )

                if not hasattr(corresponding_variable, field.name):
                    raise ValueError(
                        "The guess has the field "
                        + field.name
                        + " but it is not present in the optimization parameters"
                    )

                self._solver.set_value(
                    corresponding_variable.__getattribute__(field.name), guess
                )
                continue

            composite_variable_guess = initial_guess.__getattribute__(field.name)

            if isinstance(composite_variable_guess, OptimizationObject):
                if not hasattr(corresponding_variable, field.name):
                    raise ValueError(
                        "The guess has the field "
                        + field.name
                        + " but it is not present in the optimization structure"
                    )

                self._set_initial_guess_internal(
                    initial_guess=composite_variable_guess,
                    corresponding_variable=corresponding_variable.__getattribute__(
                        field.name
                    ),
                )
                continue

            is_list = isinstance(composite_variable_guess, list)
            list_of_optimization_objects = is_list and all(
                isinstance(elem, OptimizationObject)
                for elem in composite_variable_guess
            )

            if list_of_optimization_objects:
                if not hasattr(corresponding_variable, field.name):
                    raise ValueError(
                        "The guess has the field "
                        + field.name
                        + " but it is not present in the optimization structure"
                    )
                corresponding_nested_variable = corresponding_variable.__getattribute__(
                    field.name
                )

                if not isinstance(corresponding_nested_variable, list):
                    raise ValueError(
                        "The guess has the field "
                        + field.name
                        + " as list, but the corresponding structure is not a list"
                    )

                i = 0
                for element in composite_variable_guess:
                    if i >= len(corresponding_nested_variable):
                        raise ValueError(
                            "The input guess is the list "
                            + field.name
                            + " but the corresponding variable structure is not a list"
                        )

                    self._set_initial_guess_internal(
                        initial_guess=element,
                        corresponding_variable=corresponding_nested_variable[i],
                    )
                    i += 1

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

    def set_initial_guess(
        self, initial_guess: OptimizationObject | List[OptimizationObject]
    ):
        if isinstance(initial_guess, list):
            if not isinstance(self._variables, list):
                raise ValueError(
                    "The input guess is a list, but the specified variables structure is not"
                )

            i = 0
            for element in initial_guess:
                if i >= len(self._variables):
                    raise ValueError(
                        "The input guess is a list, but the specified variables structure is not"
                    )

                self._set_initial_guess_internal(
                    initial_guess=element, corresponding_variable=self._variables[i]
                )
                i += 1
            return

        self._set_initial_guess_internal(
            initial_guess=initial_guess, corresponding_variable=self._variables
        )

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
        if self._cost is None:
            self._cost = input_cost
            return

        self._cost += input_cost

    def add_constraint(self, input_constraint: cs.MX):
        self._solver.subject_to(input_constraint)

    def cost_function(self) -> cs.MX:
        return self._cost
