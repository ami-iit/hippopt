import copy
import dataclasses
import re
from collections.abc import Iterator
from typing import Dict, List, Tuple

import casadi as cs
import numpy as np

from hippopt.integrators.implicit_trapezoid import ImplicitTrapezoid

from .dynamics import TDynamics
from .opti_solver import OptiSolver
from .optimal_control_solver import OptimalControlSolver
from .optimization_object import OptimizationObject, TimeExpansion, TOptimizationObject
from .optimization_solver import OptimizationSolver, TOptimizationSolver
from .single_step_integrator import SingleStepIntegrator


@dataclasses.dataclass
class MultipleShootingSolver(OptimalControlSolver):
    optimization_solver: dataclasses.InitVar[OptimizationSolver] = dataclasses.field(
        default=None
    )
    _optimization_solver: TOptimizationSolver = dataclasses.field(default=None)

    default_integrator: dataclasses.InitVar[SingleStepIntegrator] = dataclasses.field(
        default=None
    )
    _default_integrator: SingleStepIntegrator = dataclasses.field(default=None)

    def __post_init__(
        self,
        optimization_solver: OptimizationSolver,
        default_integrator: SingleStepIntegrator,
    ):
        self._optimization_solver = (
            optimization_solver
            if isinstance(optimization_solver, OptimizationSolver)
            else OptiSolver()
        )

        self._default_integrator = (
            default_integrator
            if isinstance(default_integrator, SingleStepIntegrator)
            else ImplicitTrapezoid()
        )

        # TODO Stefano: Implement

    # def _get_variable_from_string(
    #     self, name: str, list_index: int
    # ) -> Tuple[dict, cs.MX] | Tuple[dict, List[cs.MX]]:
    #     self._optimization_solver.get_optimization_objects()
    #
    #     rad = re.split(pattern=".|].|]", string=name, maxsplit=1)[
    #         0
    #     ]  # Get the string before the first occurrence
    #     # of either . or ]. or ]
    #     if rad.find("[") >= 0:
    #         pass
    #     pass

    def generate_optimization_objects(
        self, input_structure: TOptimizationObject | List[TOptimizationObject], **kwargs
    ) -> TOptimizationObject | List[TOptimizationObject]:
        if isinstance(input_structure, list):
            output_list = []
            for element in input_structure:
                output_list.append(
                    self.generate_optimization_objects(element, **kwargs)
                )
            return output_list

        if "horizon" not in kwargs and "horizons" not in kwargs:
            return self._optimization_solver.generate_optimization_objects(
                input_structure=input_structure, **kwargs
            )

        default_horizon_length = int(1)
        if "horizon" in kwargs:
            default_horizon_length = int(kwargs["horizon"])
            if default_horizon_length < 1:
                raise ValueError(
                    "The specified horizon needs to be a strictly positive integer"
                )

        output = copy.deepcopy(input_structure)
        for field in dataclasses.fields(output):
            horizon_length = default_horizon_length

            constant = (
                OptimizationObject.TimeDependentField in field.metadata
                and not field.metadata[OptimizationObject.TimeDependentField]
            )

            custom_horizon = False
            if "horizons" in kwargs:
                horizons_dict = kwargs["horizons"]
                if isinstance(horizons_dict, dict) and field.name in horizons_dict:
                    constant = False
                    horizon_length = int(horizons_dict[field.name])
                    custom_horizon = True
                    if horizon_length < 1:
                        raise ValueError(
                            "The specified horizon for "
                            + field.name
                            + " needs to be a strictly positive integer"
                        )

            if constant:
                continue

            field_value = output.__getattribute__(field.name)

            expand_storage = False
            if OptimizationObject.TimeExpansionField in field.metadata:
                expand_storage = (
                    field.metadata[OptimizationObject.TimeExpansionField]
                    is TimeExpansion.Matrix
                )

            if OptimizationObject.StorageTypeField in field.metadata:
                if expand_storage:
                    if not isinstance(field_value, np.ndarray):
                        raise ValueError(
                            "Field "
                            + field.name
                            + "is not a Numpy array. Cannot expand it to the horizon."
                            ' Consider using "TimeExpansion.List" as time_expansion strategy.'
                        )

                    if field_value.ndim > 1 and field_value.shape[1] > 1:
                        raise ValueError(
                            "Cannot expand "
                            + field.name
                            + " since it is already a matrix."
                            ' Consider using "TimeExpansion.List" as time_expansion strategy.'
                        )
                    output.__setattr__(
                        field.name, np.zeros(field_value.shape[0], horizon_length)
                    )  # This is only needed to get the structure for the optimization variables.
                else:
                    output_value = []
                    for _ in range(horizon_length):
                        output_value.append(copy.deepcopy(field_value))

                    output.__setattr__(field.name, output_value)

                continue

            if (
                isinstance(field_value, OptimizationObject)
                or (
                    isinstance(field_value, list)
                    and all(
                        isinstance(elem, OptimizationObject) for elem in field_value
                    )
                )
            ) and (
                OptimizationObject.TimeDependentField
                in field.metadata  # If true, the field has to be true, see above
                or custom_horizon
            ):  # Nested variables are extended only if it is set as time dependent or if explicitly specified
                output_value = []
                for _ in range(horizon_length):
                    output_value.append(copy.deepcopy(field_value))

                output.__setattr__(field.name, output_value)

        return self._optimization_solver.generate_optimization_objects(
            input_structure=output, **kwargs
        )

    # TODO Stefano: Implement
    def _generate_flatten_optimization_objects(
        self,
        object_in: TOptimizationObject | List[TOptimizationObject],
        top_level: bool = True,
        base_string: str = "",
        base_iterator: Tuple[
            int, Iterator[TOptimizationObject | List[TOptimizationObject]]
        ] = None,
    ) -> Dict[str, Tuple[int, Iterator[cs.MX]]]:
        output = {}
        for field in dataclasses.fields(object_in):
            field_value = output.__getattribute__(field.name)

            time_dependent = (
                OptimizationObject.TimeDependentField in field.metadata
                and field.metadata[OptimizationObject.TimeDependentField]
                and top_level  # only top level variables can be time dependent
            )

            expand_storage = False
            if OptimizationObject.TimeExpansionField in field.metadata:
                expand_storage = (
                    field.metadata[OptimizationObject.TimeExpansionField]
                    is TimeExpansion.Matrix
                )

            # cases:
            # storage time dependent or not,
            # aggregate not time dependent (otherwise it would be a list),
            # list[aggregate] time dependent or not,
            # list[list of aggregate] but only if time dependent

            if OptimizationObject.StorageTypeField in field.metadata:  # storage
                if not time_dependent:
                    if base_iterator is not None:
                        new_generator = (
                            val.__getattribute__(field.name) for val in base_iterator[1]
                        )
                        output[base_string + field.name] = (
                            base_iterator[0],
                            new_generator,
                        )
                    else:
                        output[base_string + field.name] = (1, [field_value])
                    continue

                if expand_storage:
                    n = field_value.shape[1]
                    output[base_string + field.name] = (
                        n,
                        (field_value[:, k] for k in range(n)),
                    )
                    continue

                assert isinstance(field_value, list)
                n = len(field_value)  # list case
                output[base_string + field.name] = (
                    n,
                    (field_value[k] for k in range(n)),
                )
                continue

            if isinstance(field_value, OptimizationObject):  # aggregate
                generator = (
                    (val.__getattribute__(field.name) for val in base_iterator[1])
                    if base_iterator is not None
                    else None
                )

                output = output | self._generate_flatten_optimization_objects(
                    object_in=field_value,
                    top_level=False,
                    base_string=base_string + field.name + ".",
                    base_iterator=(base_iterator[0], generator),
                )
                continue

            if isinstance(field_value, list) and all(
                isinstance(elem, OptimizationObject) for elem in field_value
            ):  # list[aggregate]
                if not time_dependent:
                    for k in range(len(field_value)):
                        generator = (
                            (
                                val.__getattribute__(field.name)[k]
                                for val in base_iterator[1]
                            )
                            if base_iterator is not None
                            else None
                        )
                        output = output | self._generate_flatten_optimization_objects(
                            object_in=field_value,
                            top_level=False,
                            base_string=base_string + field.name + "[" + str(k) + "].",
                            base_iterator=(base_iterator[0], generator),
                        )
                    continue

                generator = (val for val in field_value)
                for k in range(len(field_value)):
                    output = output | self._generate_flatten_optimization_objects(
                        object_in=field_value,
                        top_level=False,
                        base_string=base_string + field.name + ".",
                        base_iterator=(len(field_value), generator),
                    )
                continue

            if isinstance(field_value, list) and all(
                isinstance(elem, list) for elem in field_value
            ):  # list[list[aggregate]]
                # TODO Stefano: finish the cases down below
                pass

        return output

    def get_optimization_objects(
        self,
    ) -> TOptimizationObject | List[TOptimizationObject]:
        return self._optimization_solver.get_optimization_objects()

    # TODO Stefano: To implement
    def add_dynamics(self, dynamics: TDynamics, **kwargs) -> None:
        if "dt" not in kwargs:
            raise ValueError(
                "MultipleShootingSolver needs dt to be specified when adding a dynamics"
            )

        top_level_index = -1
        if isinstance(self.get_optimization_objects(), list):
            if "top_level_index" not in kwargs:
                raise ValueError(
                    "The optimization objects are in a list, but top_level_index has not been specified."
                )
            top_level_index = kwargs["top_level_index"]

        dt_in = kwargs["dt"]

        dt = [{OptimizationObject.TimeDependentField: False}, cs.MX(0)]

        if isinstance(dt_in, float):
            dt[1] = dt_in
        elif isinstance(dt_in, str):
            dt = self._get_variable_from_string(dt_in, top_level_index)
        else:
            raise ValueError("Unsupported dt type")

        integrator = (
            kwargs["integrator"]
            if "integrator" in kwargs
            and isinstance(kwargs["integrator"], SingleStepIntegrator)
            else self._default_integrator
        )

        variables = {}
        for var in dynamics.state_variables():
            variables[var] = self._get_variable_from_string(var, top_level_index)

        # lhs_names = dynamics.state_variables()
        #
        # lhs_vars = []
        # for name in lhs_names:
        #     lhs_vars.append(self.get_optimization_objects().__getattribute__(name))

    def set_initial_guess(
        self, initial_guess: TOptimizationObject | List[TOptimizationObject]
    ):
        self._optimization_solver.set_initial_guess(initial_guess=initial_guess)

    def solve(self) -> None:
        self._optimization_solver.solve()

    def get_values(self) -> TOptimizationObject | List[TOptimizationObject] | None:
        return self._optimization_solver.get_values()

    def get_cost_value(self) -> float | None:
        return self._optimization_solver.get_cost_value()

    def add_cost(self, input_cost: cs.MX):
        self._optimization_solver.add_cost(input_cost=input_cost)

    def add_constraint(self, input_constraint: cs.MX):
        self._optimization_solver.add_constraint(input_constraint=input_constraint)

    def cost_function(self) -> cs.MX:
        return self._optimization_solver.cost_function()
