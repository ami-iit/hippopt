import copy
import dataclasses
import logging
from typing import Any, ClassVar

import casadi as cs
import numpy as np

from hippopt.base.opti_callback import (
    CallbackCriterion,
    SaveBestUnsolvedVariablesCallback,
)
from hippopt.base.optimization_object import (
    OptimizationObject,
    StorageType,
    TOptimizationObject,
)
from hippopt.base.optimization_solver import (
    OptimizationSolver,
    ProblemNotRegisteredException,
    SolutionNotAvailableException,
)
from hippopt.base.parameter import Parameter
from hippopt.base.problem import Problem
from hippopt.base.variable import Variable


class OptiFailure(Exception):
    def __init__(self, message: Exception, callback_used: bool):
        callback_info = ""
        if callback_used:
            callback_info = (
                " and the callback did not manage to save an intermediate solution"
            )
        super().__init__(
            f"Opti failed to solve the problem{callback_info}. Message: {str(message)}"
        )


class InitialGuessFailure(Exception):
    def __init__(self, message: Exception):
        super().__init__(
            f"Failed to set the initial guess. Message: {message}. "
            "Use 'fill_initial_guess=False' to avoid filling the "
            "initial guess automatically."
        )


@dataclasses.dataclass
class OptiSolver(OptimizationSolver):
    DefaultSolverType: ClassVar[str] = "ipopt"
    _inner_solver: str = dataclasses.field(default=DefaultSolverType)
    inner_solver: dataclasses.InitVar[str] = dataclasses.field(default=None)
    problem_type: dataclasses.InitVar[str] = dataclasses.field(default="nlp")

    _options_plugin: dict[str, Any] = dataclasses.field(default_factory=dict)
    _options_solver: dict[str, Any] = dataclasses.field(default_factory=dict)
    options_solver: dataclasses.InitVar[dict[str, Any]] = dataclasses.field(
        default=None
    )
    options_plugin: dataclasses.InitVar[dict[str, Any]] = dataclasses.field(
        default=None
    )
    _callback_criterion: CallbackCriterion = dataclasses.field(default=None)
    callback_criterion: dataclasses.InitVar[CallbackCriterion] = dataclasses.field(
        default=None
    )
    _callback: SaveBestUnsolvedVariablesCallback = dataclasses.field(default=None)
    _callback_save_costs: bool = dataclasses.field(default=True)
    _callback_save_constraint_multipliers: bool = dataclasses.field(default=True)
    callback_save_costs: dataclasses.InitVar[bool] = dataclasses.field(default=None)
    callback_save_constraint_multipliers: dataclasses.InitVar[bool] = dataclasses.field(
        default=None
    )

    _cost: cs.MX = dataclasses.field(default=None)
    _cost_expressions: dict[str, cs.MX] = dataclasses.field(default=None)
    _constraint_expressions: dict[str, cs.MX] = dataclasses.field(default=None)
    _solver: cs.Opti = dataclasses.field(default=None)
    _output_solution: TOptimizationObject | list[TOptimizationObject] = (
        dataclasses.field(default=None)
    )
    _output_cost: float = dataclasses.field(default=None)
    _cost_values: dict[str, float] = dataclasses.field(default=None)
    _constraint_values: dict[str, np.ndarray] = dataclasses.field(default=None)
    _variables: TOptimizationObject | list[TOptimizationObject] = dataclasses.field(
        default=None
    )
    _problem: Problem = dataclasses.field(default=None)
    _guess: TOptimizationObject | list[TOptimizationObject] = dataclasses.field(
        default=None
    )
    _objects_type_map: dict[cs.MX, str] = dataclasses.field(default=None)
    _free_parameters: list[str] = dataclasses.field(default=None)
    _parameters_map: dict[cs.MX, str] = dataclasses.field(default=None)
    _variables_map: dict[cs.MX, str] = dataclasses.field(default=None)
    _logger: logging.Logger = dataclasses.field(default=None)

    def __post_init__(
        self,
        inner_solver: str = DefaultSolverType,
        problem_type: str = "nlp",
        options_solver: dict[str, Any] = None,
        options_plugin: dict[str, Any] = None,
        callback_criterion: CallbackCriterion = None,
        callback_save_costs: bool = True,
        callback_save_constraint_multipliers: bool = True,
    ):
        self._solver = cs.Opti(problem_type)
        self._inner_solver = (
            inner_solver if inner_solver is not None else self.DefaultSolverType
        )
        self._options_solver = (
            options_solver if isinstance(options_solver, dict) else {}
        )
        self._options_plugin = (
            options_plugin if isinstance(options_plugin, dict) else {}
        )
        self._solver.solver(
            self._inner_solver, self._options_plugin, self._options_solver
        )
        self._callback_criterion = callback_criterion
        self._callback_save_costs = callback_save_costs
        self._callback_save_constraint_multipliers = (
            callback_save_constraint_multipliers
        )
        self._cost_expressions = {}
        self._constraint_expressions = {}
        self._objects_type_map = {}
        self._free_parameters = []
        self._parameters_map = {}
        self._variables_map = {}
        self._logger = logging.getLogger("[hippopt::OptiSolver]")

    def _generate_opti_object(
        self, storage_type: str, name: str, value: StorageType
    ) -> cs.MX:
        if value is None:
            raise ValueError("Field " + name + " is tagged as storage, but it is None.")

        if isinstance(value, np.ndarray):
            if value.ndim > 2:
                raise ValueError(
                    "Field " + name + " has number of dimensions greater than 2."
                )
            if value.ndim == 0:
                raise ValueError("Field " + name + " is a zero-dimensional vector.")

            if value.ndim < 2:
                value = np.expand_dims(value, axis=1)

        if isinstance(value, float):
            value = value * np.ones((1, 1))

        if storage_type is Variable.StorageTypeValue:
            self._logger.debug("Creating variable " + name)
            opti_object = self._solver.variable(*value.shape)
            self._objects_type_map[opti_object] = Variable.StorageTypeValue
            self._variables_map[opti_object] = name
            return opti_object

        if storage_type is Parameter.StorageTypeValue:
            self._logger.debug("Creating parameter " + name)
            opti_object = self._solver.parameter(*value.shape)
            self._objects_type_map[opti_object] = Parameter.StorageTypeValue
            self._free_parameters.append(name)
            self._parameters_map[opti_object] = name
            return opti_object

        raise ValueError("Unsupported input storage type")

    def _generate_objects_from_instance(
        self,
        input_structure: TOptimizationObject,
        parent_metadata: dict,
        base_name: str,
    ) -> TOptimizationObject:
        output = copy.deepcopy(input_structure)

        for field in dataclasses.fields(output):
            composite_value = output.__getattribute__(field.name)

            is_list = isinstance(composite_value, list)
            list_of_optimization_objects = is_list and all(
                isinstance(elem, OptimizationObject) or isinstance(elem, list)
                for elem in composite_value
            )
            list_of_float = is_list and all(
                isinstance(elem, float) for elem in composite_value
            )
            if list_of_float:
                composite_value = np.array(composite_value)
                is_list = False

            if (
                isinstance(composite_value, OptimizationObject)
                or list_of_optimization_objects
            ):
                new_parent_metadata = parent_metadata
                has_composite_metadata = (
                    OptimizationObject.CompositeTypeField in field.metadata
                    and field.metadata[OptimizationObject.CompositeTypeField]
                    is not None
                )
                if has_composite_metadata:
                    composite_metadata = field.metadata[
                        OptimizationObject.CompositeTypeField
                    ]
                    use_old_metadata = (
                        parent_metadata is not None
                        and OptimizationObject.OverrideIfCompositeField
                        in composite_metadata
                        and composite_metadata[
                            OptimizationObject.OverrideIfCompositeField
                        ]
                    )

                    if not use_old_metadata:
                        new_parent_metadata = composite_metadata

                output.__setattr__(
                    field.name,
                    self.generate_optimization_objects(
                        input_structure=composite_value,
                        fill_initial_guess=False,
                        _parent_metadata=new_parent_metadata,
                        _base_name=base_name + field.name + ".",
                    ),
                )
                continue

            if OptimizationObject.StorageTypeField in field.metadata:
                value_list = composite_value if is_list else [composite_value]
                output_value = []
                for value in value_list:
                    should_override = (
                        OptimizationObject.OverrideIfCompositeField in field.metadata
                        and field.metadata[OptimizationObject.OverrideIfCompositeField]
                    )
                    parent_can_override = (
                        parent_metadata is not None
                        and OptimizationObject.StorageTypeField in parent_metadata
                    )
                    if should_override and parent_can_override:
                        storage_type = parent_metadata[
                            OptimizationObject.StorageTypeField
                        ]
                    else:
                        storage_type = field.metadata[
                            OptimizationObject.StorageTypeField
                        ]

                    output_value.append(
                        self._generate_opti_object(
                            storage_type=storage_type,
                            name=base_name + field.name,
                            value=value,
                        )
                    )

                output.__setattr__(
                    field.name, output_value if is_list else output_value[0]
                )
                continue

        self._variables = output
        return output

    def _generate_objects_from_list(
        self,
        input_structure: list[TOptimizationObject],
        parent_metadata: dict,
        base_name: str,
    ) -> list[TOptimizationObject]:
        assert isinstance(input_structure, list)

        output = copy.deepcopy(input_structure)
        for i in range(len(output)):
            output[i] = self.generate_optimization_objects(
                input_structure=output[i],
                fill_initial_guess=False,
                _parent_metadata=parent_metadata,
                _base_name=base_name + "[" + str(i) + "].",
            )

        self._variables = output
        return output

    def _get_opti_solution(
        self, variable: cs.MX, input_solution: cs.OptiSol | dict
    ) -> StorageType:
        try:
            if isinstance(input_solution, dict):
                return input_solution[variable]
            return input_solution.value(variable)
        except Exception as err:  # noqa
            self._logger.debug(
                "Failed to get the solution for variable "
                + self._variables_map[variable]
                + ". Message: "
                + str(err)
            )
            return None

    def _generate_solution_output(
        self,
        variables: (
            TOptimizationObject
            | list[TOptimizationObject]
            | list[list[TOptimizationObject]]
        ),
        input_solution: cs.OptiSol | dict,
    ) -> TOptimizationObject | list[TOptimizationObject]:
        output = copy.deepcopy(variables)

        if isinstance(variables, list):
            for i in range(len(variables)):
                output[i] = self._generate_solution_output(variables[i], input_solution)
            return output

        for field in dataclasses.fields(variables):
            has_storage_field = OptimizationObject.StorageTypeField in field.metadata

            if has_storage_field and (
                (
                    field.metadata[OptimizationObject.StorageTypeField]
                    is Variable.StorageTypeValue
                )
                or (
                    field.metadata[OptimizationObject.StorageTypeField]
                    is Parameter.StorageTypeValue
                )
            ):
                var = variables.__getattribute__(field.name)
                if isinstance(var, list):
                    output_val = []
                    for el in var:
                        output_val.append(
                            np.array(self._get_opti_solution(el, input_solution))
                        )
                else:
                    output_val = np.array(self._get_opti_solution(var, input_solution))

                output.__setattr__(field.name, output_val)
                continue

            composite_variable = variables.__getattribute__(field.name)

            is_list = isinstance(composite_variable, list)
            list_of_optimization_objects = is_list and all(
                isinstance(elem, OptimizationObject) or isinstance(elem, list)
                for elem in composite_variable
            )

            if (
                isinstance(composite_variable, OptimizationObject)
                or list_of_optimization_objects
            ):
                output.__setattr__(
                    field.name,
                    self._generate_solution_output(composite_variable, input_solution),
                )

        return output

    def _set_opti_guess(self, variable: cs.MX, value: np.ndarray) -> None:
        match self._objects_type_map[variable]:
            case Variable.StorageTypeValue:
                self._logger.debug(
                    "Setting initial value for variable "
                    + self._variables_map[variable]
                )
                self._solver.set_initial(variable, value)
            case Parameter.StorageTypeValue:
                self._logger.debug(
                    "Setting initial value for parameter "
                    + self._parameters_map[variable]
                )
                self._solver.set_value(variable, value)
                parameter_name = self._parameters_map[variable]
                if parameter_name in self._free_parameters:
                    self._free_parameters.remove(parameter_name)

        return

    def _set_initial_guess_internal(
        self,
        initial_guess: (
            TOptimizationObject
            | list[TOptimizationObject]
            | list[list[TOptimizationObject]]
        ),
        corresponding_variable: (
            TOptimizationObject
            | list[TOptimizationObject]
            | list[list[TOptimizationObject]]
        ),
        base_name: str = "",
    ) -> None:
        if isinstance(initial_guess, list):
            if not isinstance(corresponding_variable, list):
                raise ValueError(
                    "The input guess is a list, but the specified variable "
                    + base_name
                    + " is not"
                )

            if len(corresponding_variable) != len(initial_guess):
                raise ValueError(
                    "The input guess is a list but the variable "
                    + base_name
                    + " has a different dimension. Expected: "
                    + str(len(corresponding_variable))
                    + " Input: "
                    + str(len(initial_guess))
                )

            for i in range(len(corresponding_variable)):
                self._set_initial_guess_internal(
                    initial_guess=initial_guess[i],
                    corresponding_variable=corresponding_variable[i],
                    base_name=base_name + "[" + str(i) + "].",
                )
            return

        # Check that the initial guess is an optimization object
        if not isinstance(initial_guess, OptimizationObject):
            raise ValueError(
                "The initial guess for the variable "
                + base_name
                + " is not an optimization object."
                + " It is of type "
                + str(type(initial_guess))
            )

        for field in dataclasses.fields(initial_guess):
            guess = initial_guess.__getattribute__(field.name)

            if guess is None:
                continue

            if OptimizationObject.StorageTypeField in field.metadata:
                if not hasattr(corresponding_variable, field.name):
                    raise ValueError(
                        "The guess has the field "
                        + base_name
                        + field.name
                        + " but it is not present in the optimization parameters"
                    )

                corresponding_value = corresponding_variable.__getattribute__(
                    field.name
                )

                if isinstance(corresponding_value, list):
                    self._set_list_object_guess_internal(
                        base_name, corresponding_value, field, guess
                    )
                    continue

                if isinstance(guess, float):
                    guess = guess * np.ones((1, 1))

                if isinstance(guess, list) and all(
                    isinstance(elem, float) or isinstance(elem, int) for elem in guess
                ):
                    guess = np.array(guess)

                if not isinstance(guess, np.ndarray) and not isinstance(guess, cs.DM):
                    raise ValueError(
                        "The guess for the field "
                        + base_name
                        + field.name
                        + " is neither an numpy nor a DM array."
                    )

                if len(guess.shape) == 0:
                    continue

                input_shape = (
                    guess.shape if len(guess.shape) > 1 else (guess.shape[0], 1)
                )

                if corresponding_value.shape != input_shape:
                    raise ValueError(
                        f"The guess has the field {base_name}{field.name} "
                        f"but its dimension ({input_shape}) does not match with the"
                        f" corresponding optimization variable "
                        f"({corresponding_value.shape})."
                    )

                self._set_opti_guess(
                    variable=corresponding_value,
                    value=guess,
                )
                continue

            composite_variable_guess = initial_guess.__getattribute__(field.name)

            if not isinstance(
                composite_variable_guess, OptimizationObject
            ) and not isinstance(composite_variable_guess, list):
                continue

            if not hasattr(corresponding_variable, field.name):
                raise ValueError(
                    "The guess has the field "
                    + base_name
                    + field.name
                    + " but it is not present in the optimization structure"
                )

            self._set_initial_guess_internal(
                initial_guess=composite_variable_guess,
                corresponding_variable=corresponding_variable.__getattribute__(
                    field.name
                ),
                base_name=base_name + field.name + ".",
            )
            continue

    def _set_list_object_guess_internal(
        self,
        base_name: str,
        corresponding_value: list,
        field: dataclasses.Field,
        guess: list,
    ) -> None:
        if not isinstance(guess, list):
            raise ValueError(
                "The guess for the field "
                + base_name
                + field.name
                + " is supposed to be a list. "
                + "Received "
                + str(type(guess))
                + " instead."
            )
        if len(corresponding_value) != len(guess):
            raise ValueError(
                "The guess for the field "
                + base_name
                + field.name
                + " is a list of the wrong size. Expected: "
                + str(len(corresponding_value))
                + ". Guess: "
                + str(len(guess))
            )
        for i in range(len(corresponding_value)):
            value = guess[i]
            if isinstance(value, float):
                value = value * np.ones((1, 1))

            if not isinstance(value, np.ndarray):
                raise ValueError(
                    "The field "
                    + base_name
                    + field.name
                    + "["
                    + str(i)
                    + "] is marked as a variable or a parameter. Its guess "
                    + "is supposed to be an array (or even a float if scalar)."
                )

            input_shape = value.shape if len(value.shape) > 1 else (value.shape[0], 1)

            if corresponding_value[i].shape != input_shape:
                raise ValueError(
                    "The dimension of the guess for the field "
                    + base_name
                    + field.name
                    + "["
                    + str(i)
                    + "] does not match with the corresponding"
                    + " optimization variable"
                )

            self._set_opti_guess(
                variable=corresponding_value[i],
                value=value,
            )

    def generate_optimization_objects(
        self, input_structure: TOptimizationObject | list[TOptimizationObject], **kwargs
    ) -> TOptimizationObject | list[TOptimizationObject]:
        if not isinstance(input_structure, OptimizationObject) and not isinstance(
            input_structure, list
        ):
            raise ValueError(
                "The input structure is neither an optimization object nor a list."
            )

        parent_metadata = (
            kwargs["_parent_metadata"] if "_parent_metadata" in kwargs else None
        )

        base_name = kwargs["_base_name"] if "_base_name" in kwargs else ""

        if isinstance(input_structure, OptimizationObject):
            output = self._generate_objects_from_instance(
                input_structure=input_structure,
                parent_metadata=parent_metadata,
                base_name=base_name,
            )
        else:
            output = self._generate_objects_from_list(
                input_structure=input_structure,
                parent_metadata=parent_metadata,
                base_name=base_name,
            )

        fill_initial_guess = (
            kwargs["fill_initial_guess"] if "fill_initial_guess" in kwargs else True
        )

        if fill_initial_guess:
            try:
                self.set_initial_guess(initial_guess=input_structure)
            except Exception as err:
                raise InitialGuessFailure(err)

        return output

    def get_optimization_objects(
        self,
    ) -> TOptimizationObject | list[TOptimizationObject]:
        return self._variables

    def register_problem(self, problem: Problem) -> None:
        self._problem = problem

    def get_problem(self) -> Problem:
        if self._problem is None:
            raise ProblemNotRegisteredException
        return self._problem

    def set_initial_guess(
        self, initial_guess: TOptimizationObject | list[TOptimizationObject]
    ) -> None:
        self._set_initial_guess_internal(
            initial_guess=initial_guess, corresponding_variable=self._variables
        )

        self._guess = initial_guess

    def get_initial_guess(self) -> TOptimizationObject | list[TOptimizationObject]:
        return self._guess

    def set_opti_options(
        self,
        inner_solver: str = None,
        options_solver: dict[str, Any] = None,
        options_plugin: dict[str, Any] = None,
    ) -> None:
        if inner_solver is not None:
            self._inner_solver = inner_solver
        if options_plugin is not None:
            self._options_plugin = options_plugin
        if options_solver is not None:
            self._options_solver = options_solver

        self._solver.solver(
            self._inner_solver, self._options_plugin, self._options_solver
        )

    def solve(self) -> None:
        self._cost = self._cost if self._cost is not None else cs.MX(0)
        self._solver.minimize(self._cost)
        if len(self._free_parameters):
            raise ValueError(
                "The following parameters are not set: " + str(self._free_parameters)
            )
        use_callback = self._callback_criterion is not None
        if use_callback:
            variables = []
            parameters = []
            for obj in self._objects_type_map:
                if self._objects_type_map[obj] is Variable.StorageTypeValue:
                    variables.append(obj)
                elif self._objects_type_map[obj] is Parameter.StorageTypeValue:
                    parameters.append(obj)

            self._callback = SaveBestUnsolvedVariablesCallback(
                criterion=self._callback_criterion,
                opti=self._solver,
                variables=variables,
                parameters=parameters,
                costs=(
                    list(self._cost_expressions.values())
                    if self._callback_save_costs
                    else []
                ),
                constraints=(
                    list(self._constraint_expressions.values())
                    if self._callback_save_constraint_multipliers
                    else []
                ),
            )
            self._solver.callback(self._callback)
        try:
            opti_solution = self._solver.solve()
        except Exception as err:  # noqa
            if use_callback and self._callback.best_iteration is not None:
                self._logger.warning(
                    "Opti failed to solve the problem, but the callback managed to save"
                    " an intermediate solution at "
                    f"iteration {self._callback.best_iteration}."
                )
                self._output_cost = self._callback.best_cost
                self._output_solution = self._generate_solution_output(
                    variables=self._variables,
                    input_solution=self._callback.best_objects,
                )
                self._cost_values = (
                    {
                        name: float(
                            self._callback.best_cost_values[
                                self._cost_expressions[name]
                            ]
                        )
                        for name in self._cost_expressions
                    }
                    if self._callback_save_costs
                    else {}
                )
                self._constraint_values = (
                    {
                        name: np.array(
                            (
                                self._callback.best_constraint_multipliers[
                                    self._constraint_expressions[name]
                                ]
                            )
                        )
                        for name in self._constraint_expressions
                    }
                    if self._callback_save_constraint_multipliers
                    else {}
                )
                return

            raise OptiFailure(message=err, callback_used=use_callback)

        self._output_cost = opti_solution.value(self._cost)
        self._output_solution = self._generate_solution_output(
            variables=self._variables, input_solution=opti_solution
        )
        self._cost_values = {
            name: float(opti_solution.value(self._cost_expressions[name]))
            for name in self._cost_expressions
        }
        self._constraint_values = {
            name: np.array(
                opti_solution.value(
                    self._solver.dual(self._constraint_expressions[name])
                )
            )
            for name in self._constraint_expressions
        }

    def get_values(self) -> TOptimizationObject | list[TOptimizationObject]:
        if self._output_solution is None:
            raise SolutionNotAvailableException
        return self._output_solution

    def get_cost_value(self) -> float:
        if self._output_cost is None:
            raise SolutionNotAvailableException
        return self._output_cost

    def add_cost(self, input_cost: cs.MX, name: str = None) -> None:
        if name is None:
            name = str(input_cost)

        if name in self._cost_expressions:
            raise ValueError("The cost " + name + " is already present.")

        if self._cost is None:
            self._cost = input_cost
        else:
            self._cost += input_cost

        self._cost_expressions[name] = input_cost

    def add_constraint(self, input_constraint: cs.MX, name: str = None) -> None:
        if name is None:
            name = str(input_constraint)

        if name in self._constraint_expressions:
            raise ValueError("The constraint " + name + " is already present.")

        self._solver.subject_to(input_constraint)

        self._constraint_expressions[name] = input_constraint

    def cost_function(self) -> cs.MX:
        return self._cost

    def get_cost_expressions(self) -> dict[str, cs.MX]:
        return self._cost_expressions

    def get_constraint_expressions(self) -> dict[str, cs.MX]:
        return self._constraint_expressions

    def get_cost_values(self) -> dict[str, float]:
        return self._cost_values

    def get_constraint_multipliers(self) -> dict[str, np.ndarray]:
        return self._constraint_values

    def get_object_type(self, obj: cs.MX) -> str:
        if obj not in self._objects_type_map:
            raise ValueError("The object is not an optimization object.")
        return self._objects_type_map[obj]

    def get_free_parameters_names(self) -> list[str]:
        return self._free_parameters
