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
    _objects: TOptimizationObject | list[TOptimizationObject] = dataclasses.field(
        default=None
    )
    _objects_dict: dict[str, cs.MX] = dataclasses.field(default=None)
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

        if not isinstance(value, np.ndarray):
            raise ValueError(
                f"Field {name} is tagged as storage, but it is not an array "
                f"(it is a {str(type(value))})."
            )

        if value.ndim != 2:
            raise ValueError(
                f"Field {name} has number of dimensions different from 2 "
                f"(input: {value.ndim})."
            )

        if value.shape[0] * value.shape[1] == 0:
            raise ValueError("Field " + name + " has a zero dimension.")

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

    def _get_opti_solution(
        self, variable: cs.MX, input_solution: cs.OptiSol | dict
    ) -> StorageType:
        try:
            if isinstance(input_solution, dict):
                return np.array(input_solution[variable])
            return np.array(input_solution.value(variable))
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
        is_list = isinstance(output, list)

        # Get the values from the opti solution
        output_dict = {}
        for variable in self._variables_map:
            output_dict[self._variables_map[variable]] = self._get_opti_solution(
                variable, input_solution
            )

        for parameter in self._parameters_map:
            output_dict[self._parameters_map[parameter]] = self._get_opti_solution(
                parameter, input_solution
            )

        # Convert the dict to the output structure
        if is_list:
            for i in range(len(output)):
                output[i].from_dict(input_dict=output_dict, prefix=f"[{str(i)}].")
        else:
            assert isinstance(output, OptimizationObject)
            output.from_dict(input_dict=output_dict)

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

    def generate_optimization_objects(
        self, input_structure: TOptimizationObject | list[TOptimizationObject], **kwargs
    ) -> TOptimizationObject | list[TOptimizationObject]:
        if not isinstance(input_structure, OptimizationObject) and not isinstance(
            input_structure, list
        ):
            raise ValueError(
                "The input structure is neither an optimization object nor a list."
            )
        output = copy.deepcopy(input_structure)
        is_list = isinstance(output, list)
        input_list = output if is_list else [output]
        input_as_dict = {}
        input_metadata_as_dict = {}
        output_dict = {}

        # In case of list, flatten to a single dict
        for i, elem in enumerate(input_list):
            prefix = f"[{i}]." if is_list else ""
            elem_dict, elem_metadata = elem.to_dicts(prefix=prefix)
            input_as_dict.update(elem_dict)
            input_metadata_as_dict.update(elem_metadata)

        reverse_input_dict = {}
        duplicates = {}
        for obj_name in input_as_dict:
            value = input_as_dict[obj_name]
            if value is None or isinstance(value, float):
                continue
            value_id = id(value)
            if value_id in reverse_input_dict:
                if value_id not in duplicates:
                    duplicates[value_id] = []
                duplicates[value_id].append(obj_name)
            else:
                reverse_input_dict[value_id] = obj_name

        duplicates_string = ""
        for value_id in duplicates:
            duplicates_string += (
                f"{reverse_input_dict[value_id]} is duplicated in the "
                f"following fields: {duplicates[value_id]}\n"
            )
        if len(duplicates):
            raise ValueError(
                "The following fields share the same object as value:\n"
                + duplicates_string
                + "This can cause issues when assigning a new value to these fields."
            )

        # For each element of the dict, create an opti object
        for obj_name in input_as_dict:
            value_id = input_as_dict[obj_name]
            storage_type = input_metadata_as_dict[obj_name][
                OptimizationObject.StorageTypeField
            ]
            output_dict[obj_name] = self._generate_opti_object(
                storage_type=storage_type, name=obj_name, value=value_id
            )

        # Convert the dict to the output structure
        if is_list:
            for i in range(len(output)):
                output[i].from_dict(input_dict=output_dict, prefix=f"[{str(i)}].")
        else:
            assert isinstance(output, OptimizationObject)
            output.from_dict(input_dict=output_dict)

        self._objects = output
        self._objects_dict = output_dict

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
        return self._objects

    def register_problem(self, problem: Problem) -> None:
        self._problem = problem

    def get_problem(self) -> Problem:
        if self._problem is None:
            raise ProblemNotRegisteredException
        return self._problem

    def set_initial_guess(
        self, initial_guess: TOptimizationObject | list[TOptimizationObject]
    ) -> None:

        is_list = isinstance(initial_guess, list)
        if is_list and not isinstance(self._objects, list):
            raise ValueError(
                "The input guess is a list, but the optimization structure is not"
            )

        if is_list and len(self._objects) != len(initial_guess):
            raise ValueError(
                "The input guess is a list but the optimization structure has"
                " a different dimension. Expected: "
                + str(len(self._objects))
                + " Input: "
                + str(len(initial_guess))
            )

        guess_list = initial_guess if is_list else [initial_guess]
        guess_as_dict = {}
        for i, elem in enumerate(guess_list):
            prefix = f"[{i}]." if is_list else ""
            elem_dict = elem.to_dict(prefix=prefix)
            guess_as_dict.update(elem_dict)

        for obj_name in guess_as_dict:
            guess_value = guess_as_dict[obj_name]
            if guess_value is None:
                continue

            if not isinstance(guess_value, np.ndarray) and not isinstance(
                guess_value, cs.DM
            ):
                raise ValueError(
                    f"The guess for the field {obj_name} is neither a numpy nor a"
                    f" DM array. (Type: {str(type(guess_value))})."
                )

            if len(guess_value.shape) == 0:
                continue

            if obj_name not in self._objects_dict:
                continue

            corresponding_object = self._objects_dict[obj_name]

            if not isinstance(corresponding_object, cs.MX):
                raise ValueError(f"The field {obj_name} has not been added to opti.")

            input_shape = (
                guess_value.shape
                if len(guess_value.shape) > 1
                else (guess_value.shape[0], 1)
            )

            if corresponding_object.shape != input_shape:
                raise ValueError(
                    f"The guess has the field {obj_name} "
                    f"but its dimension ({input_shape}) does not match with the"
                    f" corresponding optimization variable "
                    f"({obj_name.shape})."
                )

            self._set_opti_guess(
                variable=corresponding_object,
                value=guess_value,
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
                    variables=self._objects,
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
            variables=self._objects, input_solution=opti_solution
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
