import copy
import dataclasses
from collections.abc import Iterable
from functools import singledispatchmethod
from typing import Any, ClassVar, List, Type

import casadi as cs
import numpy as np

from hippopt.base.optimization_object import OptimizationObject, TOptimizationObject
from hippopt.base.parameter import Parameter
from hippopt.base.solver import Solver
from hippopt.base.variable import Variable


@dataclasses.dataclass
class OptiSolver(Solver):
    DefaultSolverType: ClassVar[str] = "ipopt"
    _inner_solver: str = dataclasses.field(default=DefaultSolverType)
    _problem_type: dataclasses.InitVar[str] = dataclasses.field(default="nlp")

    _options_plugin: dict[str, Any] = dataclasses.field(default_factory=dict)
    _options_solver: dict[str, Any] = dataclasses.field(default_factory=dict)

    _cost: cs.MX = dataclasses.field(default=None)
    _solver: cs.Opti = dataclasses.field(default=None)
    _solution: cs.OptiSol = dataclasses.field(default=None)
    _variables: TOptimizationObject | List[TOptimizationObject] = dataclasses.field(
        default=None
    )

    def __post_init__(self, _problem_type: str) -> None:
        self._solver = cs.Opti(_problem_type)
        self._solver.solver(
            self._inner_solver, self._options_plugin, self._options_solver
        )

    @singledispatchmethod
    def generate_optimization_objects(
        self, input_structure: Type[OptimizationObject] | List[Type[OptimizationObject]]
    ):
        pass

    @generate_optimization_objects.register
    def _(self, input_structure: OptimizationObject) -> TOptimizationObject:
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

    @generate_optimization_objects.register
    def _(self, input_structure: list) -> List[TOptimizationObject]:
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

    def solve(self):
        self._solver.minimize(self._cost)
        self._solution = self._solver.solve()

    def add_cost(self, input_cost: cs.Function):
        # TODO Stefano: Check if it is a constraint. If is an equality, add the 2-norm. If it is an inequality?
        if self._cost is None:
            _cost = input_cost
            return

        self._cost += input_cost

    def add_constraint(self, input_constraint: cs.Function):
        # TODO Stefano: Check if it is a cost. If so, set it equal to zero
        self._solver.subject_to(input_constraint)
