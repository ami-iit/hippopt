import copy
import dataclasses
from collections.abc import Iterable
from functools import singledispatchmethod
from typing import List, Type

import casadi as cs
import numpy as np

from hippopt.base.optimization_object import OptimizationObject, TOptimizationObject
from hippopt.base.parameter import Parameter
from hippopt.base.variable import Variable


@dataclasses.dataclass
class OptiSolver:
    _solver: cs.Opti = dataclasses.field(default_factory=cs.Opti)
    _variables: TOptimizationObject | List[TOptimizationObject] = dataclasses.field(
        default=None
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
