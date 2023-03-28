import copy
import dataclasses
from typing import List, Tuple

import casadi as cs
import numpy as np

from .dynamics import TDynamics
from .opti_solver import OptiSolver
from .optimal_control_solver import OptimalControlSolver
from .optimization_object import OptimizationObject, TOptimizationObject
from .optimization_solver import OptimizationSolver, TOptimizationSolver


@dataclasses.dataclass
class MultipleShootingSolver(OptimalControlSolver):
    optimization_solver: dataclasses.InitVar[OptimizationSolver] = dataclasses.field(
        default=None
    )
    _optimization_solver: TOptimizationSolver = dataclasses.field(default=None)

    def __post_init__(self, optimization_solver: OptimizationSolver = None):
        self._optimization_solver = (
            optimization_solver
            if isinstance(optimization_solver, OptimizationSolver)
            else OptiSolver()
        )

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

        expand_storage = False
        if "expand_storage" in kwargs:
            expand_storage = bool(kwargs["expand_storage"])

        output = copy.deepcopy(input_structure)
        for field in dataclasses.fields(output):
            horizon_length = default_horizon_length

            constant = (
                OptimizationObject.TimeDependentField in field.metadata
                and not field.metadata[OptimizationObject.TimeDependentField]
            )

            if "horizons" in kwargs:
                horizons_dict = kwargs["horizons"]
                if isinstance(horizons_dict, dict) and field.name in horizons_dict:
                    constant = False
                    horizon_length = int(horizons_dict[field.name])
                    if horizon_length < 1:
                        raise ValueError(
                            "The specified horizon for "
                            + field.name
                            + " needs to be a strictly positive integer"
                        )

            if constant:
                continue

            field_value = output.__getattribute__(field.name)

            if OptimizationObject.StorageTypeField in field.metadata and expand_storage:
                if not isinstance(field_value, np.ndarray):
                    raise ValueError(
                        "Field "
                        + field.name
                        + 'is not a Numpy array. Cannot expand it to the horizon. Consider using "expand_storage=False"'
                    )

                if field_value.ndim > 1 and field_value.shape[1] > 1:
                    raise ValueError(
                        "Cannot expand " + field.name + " since it is already a matrix."
                    )
                output.__setattr__(
                    field.name, np.zeros(field_value.shape[0], horizon_length)
                )  # This is only needed to get the structure for the optimization variables.
            else:
                if (
                    OptimizationObject.StorageTypeField in field.metadata
                    or isinstance(field_value, OptimizationObject)
                    or (
                        isinstance(field_value, list)
                        and all(
                            isinstance(elem, OptimizationObject) for elem in field_value
                        )
                    )
                ):
                    output_value = []
                    for _ in range(horizon_length):
                        output_value.append(copy.deepcopy(field_value))

                    output.__setattr__(field.name, output_value)

        return self._optimization_solver.generate_optimization_objects(
            input_structure=output, **kwargs
        )

    def get_optimization_objects(
        self,
    ) -> TOptimizationObject | List[TOptimizationObject]:
        return self._optimization_solver.get_optimization_objects()

    # TODO Stefano: To implement
    def add_dynamics(self, time_derivative: TDynamics, **kwargs):
        lhs_names = time_derivative.state_variables()

        lhs_list = lhs_names if isinstance(lhs_names, list) else [lhs_names]

        lhs_vars = []
        for name in lhs_list:
            lhs_vars.append(self.get_optimization_objects().__dict__[name])

    def set_initial_guess(
        self, initial_guess: TOptimizationObject | List[TOptimizationObject]
    ):
        self._optimization_solver.set_initial_guess(initial_guess=initial_guess)

    def solve(self) -> Tuple[TOptimizationObject, float]:
        return self._optimization_solver.solve()

    def get_solution(self) -> TOptimizationObject | List[TOptimizationObject] | None:
        return self._optimization_solver.get_solution()

    def get_cost_value(self) -> float | None:
        return self._optimization_solver.get_cost_value()

    def add_cost(self, input_cost: cs.MX):
        self._optimization_solver.add_cost(input_cost=input_cost)

    def add_constraint(self, input_constraint: cs.MX):
        self._optimization_solver.add_constraint(input_constraint=input_constraint)

    def cost_function(self) -> cs.MX:
        return self._optimization_solver.cost_function()
