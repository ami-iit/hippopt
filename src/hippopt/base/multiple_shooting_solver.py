import copy
import dataclasses
import itertools
from collections.abc import Iterator
from functools import partial
from typing import Callable

import casadi as cs
import numpy as np

from hippopt.integrators.implicit_trapezoid import ImplicitTrapezoid

from .dynamics import TDynamics
from .opti_solver import OptiSolver
from .optimal_control_solver import OptimalControlSolver
from .optimization_object import OptimizationObject, TimeExpansion, TOptimizationObject
from .optimization_solver import OptimizationSolver, TOptimizationSolver
from .problem import ExpressionType, Problem
from .single_step_integrator import SingleStepIntegrator, step

FlattenedVariableIterator = Callable[[], Iterator[cs.MX]]
FlattenedVariableTuple = tuple[int, FlattenedVariableIterator]
FlattenedVariableDict = dict[str, FlattenedVariableTuple]


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

    _flattened_variables: FlattenedVariableDict = dataclasses.field(default=None)

    _symbolic_structure: TOptimizationObject | list[TOptimizationObject] = (
        dataclasses.field(default=None)
    )

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
        self._flattened_variables = {}

    @staticmethod
    def _extend_structure_to_horizon(
        input_structure: TOptimizationObject | list[TOptimizationObject], **kwargs
    ) -> TOptimizationObject | list[TOptimizationObject]:
        if "horizon" not in kwargs and "horizons" not in kwargs:
            return input_structure

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

            expand_storage = (
                field.metadata[OptimizationObject.TimeExpansionField]
                is TimeExpansion.Matrix
                if OptimizationObject.TimeExpansionField in field.metadata
                else False
            )

            if OptimizationObject.StorageTypeField in field.metadata:
                if expand_storage:
                    if not isinstance(field_value, np.ndarray):
                        raise ValueError(
                            "Field "
                            + field.name
                            + "is not a Numpy array. Cannot expand it to the horizon."
                            ' Consider using "TimeExpansion.List"'
                            " as time_expansion strategy."
                        )

                    if field_value.ndim > 1 and field_value.shape[1] > 1:
                        raise ValueError(
                            "Cannot expand "
                            + field.name
                            + " since it is already a matrix."
                            ' Consider using "TimeExpansion.List"'
                            " as time_expansion strategy."
                        )
                    # Repeat the vector along the horizon
                    if field_value.ndim < 2:
                        field_value = np.expand_dims(field_value, axis=1)
                    output.__setattr__(
                        field.name, np.tile(field_value, (1, horizon_length))
                    )
                else:
                    output_value = []
                    for _ in range(horizon_length):
                        output_value.append(copy.deepcopy(field_value))

                    output.__setattr__(field.name, output_value)

                continue

            if (
                OptimizationObject.TimeDependentField not in field.metadata
                and not custom_horizon
            ):
                continue
            # We expand nested variables (following two cases) only if time dependent

            if isinstance(field_value, OptimizationObject):
                output_value = []
                for _ in range(horizon_length):
                    output_value.append(copy.deepcopy(field_value))

                output.__setattr__(field.name, output_value)
                continue

            if isinstance(field_value, list) and all(
                isinstance(elem, OptimizationObject) for elem in field_value
            ):
                # Nested variables are extended only if it is set as time dependent
                #  or if it has custom horizon
                if not len(field_value):  # skip empty lists
                    continue

                output_value = []
                for el in field_value:
                    element = []
                    for _ in range(horizon_length):
                        element.append(copy.deepcopy(el))
                    output_value.append(element)

                output.__setattr__(field.name, output_value)
                continue

        return output

    def generate_optimization_objects(
        self, input_structure: TOptimizationObject | list[TOptimizationObject], **kwargs
    ) -> TOptimizationObject | list[TOptimizationObject]:
        if isinstance(input_structure, list):
            expanded_structure = []
            self._symbolic_structure = []
            for element in input_structure:
                expanded_structure.append(
                    self._extend_structure_to_horizon(input_structure=element, **kwargs)
                )

            variables = self._optimization_solver.generate_optimization_objects(
                input_structure=expanded_structure, **kwargs
            )

            for k in range(len(variables)):
                flattened, symbolic = self._generate_flattened_and_symbolic_objects(
                    object_in=variables[k], base_string="[" + str(k) + "]."
                )
                self._flattened_variables = self._flattened_variables | flattened
                self._symbolic_structure.append(symbolic)

        else:
            expanded_structure = self._extend_structure_to_horizon(
                input_structure=input_structure, **kwargs
            )
            variables = self._optimization_solver.generate_optimization_objects(
                input_structure=expanded_structure, **kwargs
            )
            flattened, symbolic = self._generate_flattened_and_symbolic_objects(
                object_in=variables
            )
            self._flattened_variables = flattened
            self._symbolic_structure = symbolic

        return variables

    def _generate_flattened_and_symbolic_objects(  # TODO: remove some indentation
        self,
        object_in: TOptimizationObject,
        top_level: bool = True,
        base_string: str = "",
        base_iterator: tuple[
            int, Callable[[], Iterator[TOptimizationObject | list[TOptimizationObject]]]
        ] = None,
    ) -> tuple[FlattenedVariableDict, TOptimizationObject]:
        assert (bool(top_level) != bool(base_iterator is not None)) or (
            not top_level and base_iterator is None
        )  # Cannot be top level and have base iterator

        output_dict = {}
        output_symbolic = copy.deepcopy(object_in)

        for field in dataclasses.fields(object_in):
            field_value = object_in.__getattribute__(field.name)

            time_dependent = (
                OptimizationObject.TimeDependentField in field.metadata
                and field.metadata[OptimizationObject.TimeDependentField]
                and top_level  # only top level variables can be time-dependent
            )

            expand_storage = (
                field.metadata[OptimizationObject.TimeExpansionField]
                is TimeExpansion.Matrix
                if OptimizationObject.TimeExpansionField in field.metadata
                else False
            )

            # cases:
            # storage, time dependent or not,
            # aggregate, not time dependent (otherwise it would be a list),
            # list[aggregate], time dependent or not,
            # list[list of aggregate], but only if time dependent

            if OptimizationObject.StorageTypeField in field.metadata:  # storage
                if not time_dependent:
                    if base_iterator is not None:
                        # generators cannot be rewound, but we might need to reuse them.
                        # Hence, we store a lambda that can return a generator.
                        #  Since in python it is not possible
                        # to have capture lists as in C++, we use partial from functools
                        # to store the inputs of the lambda within itself
                        # (inspired from https://stackoverflow.com/a/10101476)
                        new_generator = partial(
                            (
                                lambda name, base_generator_fun: (
                                    val.__getattribute__(name)
                                    for val in base_generator_fun()
                                )
                            ),
                            field.name,
                            base_iterator[1],
                        )
                        output_dict[base_string + field.name] = (
                            base_iterator[0],
                            new_generator,
                        )
                    else:
                        constant_generator = partial(
                            (lambda value: itertools.repeat(value)), field_value
                        )
                        output_dict[base_string + field.name] = (
                            1,
                            constant_generator,
                        )

                    output_symbolic.__setattr__(
                        field.name,
                        cs.MX.sym(
                            base_string + field.name,
                            field_value.rows(),
                            field_value.columns(),
                        ),
                    )

                    continue

                if expand_storage:
                    n = field_value.shape[1]
                    new_generator = partial(
                        (lambda value, dim: (value[:, i] for i in range(dim))),
                        field_value,
                        n,
                    )
                    output_dict[base_string + field.name] = (n, new_generator)
                    output_symbolic.__setattr__(
                        field.name,
                        cs.MX.sym(
                            base_string + field.name,
                            field_value.rows(),
                            1,
                        ),
                    )
                    continue

                assert isinstance(
                    field_value, list
                )  # time dependent and not expand_storage
                n = len(field_value)  # list case
                assert n > 0
                new_generator = partial(
                    (lambda value, dim: (value[i] for i in range(dim))),
                    field_value,
                    n,
                )
                output_dict[base_string + field.name] = (n, new_generator)
                first_element = field_value[0]  # Assume all elements have same dims
                output_symbolic.__setattr__(
                    field.name,
                    cs.MX.sym(
                        base_string + field.name,
                        first_element.rows(),
                        first_element.columns(),
                    ),
                )
                continue

            if isinstance(
                field_value, OptimizationObject
            ):  # aggregate (cannot be time-dependent)
                generator = (
                    partial(
                        (
                            lambda name, base_generator_fun: (
                                val.__getattribute__(name)
                                for val in base_generator_fun()
                            )
                        ),
                        field.name,
                        base_iterator[1],
                    )
                    if base_iterator is not None
                    else None
                )

                (
                    inner_dict,
                    inner_symbolic,
                ) = self._generate_flattened_and_symbolic_objects(
                    object_in=field_value,
                    top_level=False,
                    base_string=base_string + field.name + ".",
                    base_iterator=(
                        (base_iterator[0], generator) if generator is not None else None
                    ),
                )

                output_dict = output_dict | inner_dict
                output_symbolic.__setattr__(
                    field.name,
                    inner_symbolic,
                )
                continue

            if isinstance(field_value, list) and all(
                isinstance(elem, OptimizationObject) for elem in field_value
            ):  # list[aggregate]
                if not time_dependent:
                    symbolic_list = output_symbolic.__getattribute__(field.name)
                    for k in range(len(field_value)):
                        generator = (
                            partial(
                                (
                                    lambda name, base_generator_fun, i: (
                                        val.__getattribute__(name)[i]
                                        for val in base_generator_fun()
                                    )
                                ),
                                field.name,
                                base_iterator[1],
                                k,
                            )
                            if base_iterator is not None
                            else None
                        )

                        (
                            inner_dict,
                            inner_symbolic,
                        ) = self._generate_flattened_and_symbolic_objects(
                            object_in=field_value[k],
                            top_level=False,
                            base_string=base_string
                            + field.name
                            + "["
                            + str(k)
                            + "].",  # we flatten the list. Note the added [k]
                            base_iterator=(
                                (base_iterator[0], generator)
                                if generator is not None
                                else None
                            ),
                        )

                        output_dict = output_dict | inner_dict
                        symbolic_list[k] = inner_symbolic

                    continue

                if not len(field_value):
                    continue

                iterable = iter(field_value)
                first = next(iterable)
                assert all(
                    isinstance(el, type(first)) for el in iterable
                )  # check that each element has same type

                # If we are time-dependent (and hence top_level has to be true),
                # there is no base generator
                new_generator = partial(
                    (lambda value: (val for val in value)), field_value
                )

                (
                    inner_dict,
                    inner_symbolic,
                ) = self._generate_flattened_and_symbolic_objects(
                    # since they are al equal, expand only the first
                    # and exploit the base_iterator
                    object_in=first,
                    top_level=False,
                    base_string=base_string
                    + field.name
                    + ".",  # we don't flatten the list
                    base_iterator=(len(field_value), new_generator),
                )

                output_dict = output_dict | inner_dict
                output_symbolic.__setattr__(field.name, inner_symbolic)
                continue

            if (
                isinstance(field_value, list)
                and time_dependent
                and all(isinstance(elem, list) for elem in field_value)
            ):  # list[list[aggregate]], only time dependent
                # The inner list is the time dependency
                symbolic_list = output_symbolic.__getattribute__(field.name)
                for k in range(len(field_value)):
                    inner_list = field_value[k]

                    if not len(inner_list):
                        break

                    iterable = iter(inner_list)
                    first = next(iterable)
                    assert all(
                        isinstance(el, type(first)) for el in iterable
                    )  # check that each element has same type

                    new_generator = partial(
                        (lambda value: (val for val in value)), inner_list
                    )

                    (
                        inner_dict,
                        inner_symbolic,
                    ) = self._generate_flattened_and_symbolic_objects(
                        object_in=first,
                        top_level=False,
                        base_string=base_string + field.name + "[" + str(k) + "].",
                        base_iterator=(len(inner_list), new_generator),
                    )

                    output_dict = output_dict | inner_dict
                    symbolic_list[k] = inner_symbolic
                continue

        return output_dict, output_symbolic

    def get_optimization_objects(
        self,
    ) -> TOptimizationObject | list[TOptimizationObject]:
        return self._optimization_solver.get_optimization_objects()

    def get_optimization_structure(
        self,
    ) -> TOptimizationObject | list[TOptimizationObject]:
        return self._optimization_solver.get_optimization_structure()

    def register_problem(self, problem: Problem) -> None:
        self._optimization_solver.register_problem(problem)

    def get_problem(self) -> Problem:
        return self._optimization_solver.get_problem()

    def get_flattened_optimization_objects(
        self,
    ) -> FlattenedVariableDict:
        return self._flattened_variables

    def get_symbolic_structure(self) -> TOptimizationObject | list[TOptimizationObject]:
        return self._symbolic_structure

    def add_dynamics(
        self,
        dynamics: TDynamics,
        x0: dict[str, cs.MX] | dict[cs.MX, cs.MX] | cs.MX = None,
        t0: cs.MX = cs.MX(0.0),
        mode: ExpressionType = ExpressionType.subject_to,
        name: str = None,
        x0_name: str = None,
        **kwargs
    ) -> None:
        """
        Add a dynamics to the optimal control problem
        :param dynamics: The dynamics to add. The variables involved need to have a
                         name corresponding to the name of a flattened variable.
                         If the variable is nested, you can use "." as separator
                         (e.g. "a.b" will look for variable b within a).
                         If there is a list, you can use "[k]" with "k" the element
                         to pick. For example "a.b[k].c" will look for variable "c"
                         defined in the k-th element of "b" within "a".
                         Only the top level variables can be time-dependent.
                         In this case, "a" could be time-dependent and being a list,
                         but this is automatically detected, and there is no need to
                         specify the time-dependency. The "[k]" keyword is used only
                         in case the list is not time-dependent.
                         In case we have a list of optimization objects, prepend "[k]."
                         to name of the variable, with k the top level index.
        :param x0: The initial state. It is a dictionary with the key equal to the state
                   variable name, or its corresponding symbolic variable.
                   It can also be a single MX in case there is only one state variable.
                   If no dict is provided, or a given variable is not
                   found in the dictionary, the initial condition is not set.
        :param t0: The initial time
        :param mode: Optional argument to set the mode with which the
                     dynamics is added to the problem.
                     Default: constraint
        :param name: The name used when adding the dynamics expression.
        :param x0_name: The name used when adding the initial condition expression.
        :param kwargs: Additional arguments. There are some required arguments:
                                            - "dt": the integration time delta.
                                                    It can be a float in case it
                                                    is constant, or a string to indicate
                                                    the (flattened) name of the
                                                    variable to use, or a cs.MX when
                                                    using the corresponding variable in
                                                    the symbolic structure

                                            Optional arguments:
                                            - "max_steps": the number of integration
                                                           steps. If not specified, the
                                                           entire horizon of the
                                                           specific state variable
                                                           is used
                                            - "integrator": specify the
                                                            `SingleStepIntegrator` to
                                                            use. This needs to be a type
                                            - optional arguments of the
                                              `Problem.add_expression` method.
        :return: None
        """
        if "dt" not in kwargs:
            raise ValueError(
                "MultipleShootingSolver needs dt to be specified when adding a dynamics"
            )

        dt_in = kwargs["dt"]

        max_n = 0

        if "max_steps" in kwargs:
            max_n = kwargs["max_steps"]

            if not isinstance(max_n, int) or max_n < 2:
                raise ValueError(
                    "max_steps is specified, but it needs to be an integer"
                    " greater than 1"
                )

        dt_size = 1

        if isinstance(dt_in, float):
            dt_generator = itertools.repeat(cs.MX(dt_in))
        elif isinstance(dt_in, str) or isinstance(dt_in, cs.MX):
            dt_in = dt_in.name() if isinstance(dt_in, cs.MX) else dt_in
            if dt_in not in self._flattened_variables:
                raise ValueError(
                    "The specified dt name is not found in the optimization variables"
                )
            dt_var_tuple = self._flattened_variables[dt_in]
            dt_size = dt_var_tuple[0]
            dt_generator = dt_var_tuple[1]()
        else:
            raise ValueError("Unsupported dt type")

        dt_tuple = (dt_size, dt_generator)

        integrator = (
            kwargs["integrator"] if "integrator" in kwargs else self._default_integrator
        )

        if not issubclass(integrator, SingleStepIntegrator):
            raise ValueError(
                "The integrator has been defined, but it is not "
                "a subclass of SingleStepIntegrator"
            )

        variables = {}
        n = max_n
        for var in dynamics.state_variables():
            if var not in self._flattened_variables:
                raise ValueError(
                    "Variable " + var + " not found in the optimization variables."
                )
            var_tuple = self._flattened_variables[var]
            var_n = var_tuple[0]
            if n == 0:
                if var_n < 2:
                    raise ValueError(
                        "The state variable " + var + " is not time dependent."
                    )
                n = var_n

            if var_n < n:
                raise ValueError(
                    "The state variable " + var + " has a too short prediction horizon."
                )

            # With var_tuple[1]() we get a new generator for the specific variable
            variables[var] = (var_tuple[0], var_tuple[1]())

        if 1 < dt_tuple[0] < n:
            raise ValueError("The specified dt has a too small prediction horizon.")

        additional_inputs = {}
        for inp in dynamics.input_names():
            if inp not in self._flattened_variables:
                raise ValueError(
                    "Variable " + inp + " not found in the optimization variables."
                )

            if inp not in variables:
                inp_tuple = self._flattened_variables[inp]

                inp_n = inp_tuple[0]

                if 1 < inp_n < n:
                    raise ValueError(
                        "The input "
                        + inp
                        + " is time dependent, but it has a too small "
                        "prediction horizon."
                    )

                additional_inputs[inp] = (inp_tuple[0], inp_tuple[1]())

        base_name = (
            name
            if name is not None
            else "dot(" + ", ".join(dynamics.state_variables()) + ")"
        )

        x_k = {name: next(variables[name][1]) for name in variables}
        u_k = {name: next(additional_inputs[name][1]) for name in additional_inputs}

        initial_conditions = {}
        if x0 is not None:
            if not isinstance(x0, dict):
                if len(x_k) > 1:
                    raise ValueError(
                        "The initial condition is a single MX, but the dynamics "
                        "has more than one state variable."
                    )
                x0 = {list(x_k.keys())[0]: x0}
            for var in x0:
                var_name = var if isinstance(var, str) else var.name()  # noqa
                if var_name in x_k:
                    initial_conditions[var_name] = x0[var]

            if x0_name is None and name is not None:
                x0_name = name + "[0]"

            # In the following, we add the initial condition expressions
            # through the problem interface.
            # In this way, we can exploit the machinery handling the generators,
            # and we can switch the dynamics from constraints to costs
            self.get_problem().add_expression(
                mode=mode,
                expression=(
                    cs.MX(x_k[name] == initial_conditions[name])
                    for name in initial_conditions
                ),
                name=x0_name,
                **kwargs
            )

        for i in range(n - 1):
            x_next = {name: next(variables[name][1]) for name in variables}
            u_next = {
                name: next(additional_inputs[name][1]) for name in additional_inputs
            }
            dt = next(dt_tuple[1])
            integrated = step(
                integrator,
                dynamics=dynamics,
                x0=x_k | u_k,
                xf=x_next | u_next,
                dt=dt,
                t0=t0 + cs.MX(i) * dt,
            )

            name = base_name + "[" + str(i + 1) + "]"

            # In the following, we add the dynamics expressions through the problem
            # interface, rather than the solver interface. In this way, we can exploit
            # the machinery handling the generators, and we can switch the dynamics
            # from constraints to costs
            self.get_problem().add_expression(
                mode=mode,
                expression=(cs.MX(x_next[name] == integrated[name]) for name in x_next),
                name=name,
                **kwargs
            )

            x_k = x_next
            u_k = u_next

    def add_expression_to_horizon(
        self,
        expression: cs.MX,
        mode: ExpressionType = ExpressionType.subject_to,
        apply_to_first_elements: bool = False,
        name: str = None,
        **kwargs
    ) -> None:
        """
        Add an expression to the whole horizon of the optimal control problem
        :param expression: The expression to add. Use the symbolic_structure to set up
                           expression
        :param mode: Optional argument to set the mode with which the
                     dynamics is added to the problem.
                     Default: constraint
        :param apply_to_first_elements: Flag to define if the constraint need to be
                                        applied also to the first elements. If True
                                        the expression will be applied also to the first
                                        elements. Default: False
        :param name: The name used when adding the expression.
        :param kwargs: Optional arguments:
                             - "max_steps": the number of integration
                                            steps. If not specified, the
                                            number of steps is determined from the
                                            variables involved.
                             - optional arguments of the
                               `Problem.add_expression` method.
        :return: None
        """

        input_variables = cs.symvar(expression)
        input_variables_names = [var.name() for var in input_variables]
        base_name = name if name is not None else str(expression)

        max_n = 1
        max_iterations_set = False

        if "max_steps" in kwargs:
            max_n = kwargs["max_steps"]
            max_iterations_set = True
            if not isinstance(max_n, int) or max_n < 1:
                raise ValueError(
                    "max_steps is specified, but it needs to be an integer"
                    " greater than 0"
                )

        variables_generators = []
        n = 1
        for var in input_variables_names:
            if var not in self._flattened_variables:
                raise ValueError(
                    "Variable " + var + " not found in the optimization variables."
                )
            var_tuple = self._flattened_variables[var]
            var_n = var_tuple[0]
            n = var_n if n == 1 or 1 < var_n < n else n

            # With var_tuple[1]() we get a new generator for the specific variable
            variables_generators.append(var_tuple[1]())

        if max_iterations_set:
            n = min(max_n, n)

        for i in range(n):
            x_k = [next(var) for var in variables_generators]

            name = base_name + "[" + str(i) + "]"

            if i == 0 and not apply_to_first_elements and not n == 1:
                continue

            # In the following, we add the expressions through the problem
            # interface, rather than the solver interface. In this way, we can exploit
            # the machinery handling the generators, and we can switch the expression
            # from constraints to costs
            self.get_problem().add_expression(
                mode=mode,
                expression=cs.substitute([expression], input_variables, x_k)[0],
                name=name,
                **kwargs
            )

    def initial(self, variable: str | cs.MX) -> cs.MX:
        """
        Get the initial value of a variable
        :param variable: The name of the flattened variable.
                         If the variable is nested, you can use "." as separator
                         (e.g. "a.b" will look for variable b within a).
                         If there is a list, you can use "[k]" with "k" the element
                         to pick. For example "a.b[k].c" will look for variable "c"
                         defined in the k-th element of "b" within "a".
                         As an alternative it is possible to use the corresponding
                         variable from the symbolic structure.
        :return: The first value of the variable
        """
        if isinstance(variable, cs.MX):
            variable = variable.name()

        if variable not in self._flattened_variables:
            raise ValueError(
                "Variable " + variable + " not found in the optimization variables."
            )

        return next(self._flattened_variables[variable][1]())

    def final(self, variable: str | cs.MX) -> cs.MX:
        """
        Get the final value of a variable
        :param variable: The name of the flattened variable.
                         If the variable is nested, you can use "." as separator
                         (e.g. "a.b" will look for variable b within a).
                         If there is a list, you can use "[k]" with "k" the element
                         to pick. For example "a.b[k].c" will look for variable "c"
                         defined in the k-th element of "b" within "a".
                         As an alternative it is possible to use the corresponding
                         variable from the symbolic structure.
        :return: The last value of the variable
        """
        if isinstance(variable, cs.MX):
            variable = variable.name()

        if variable not in self._flattened_variables:
            raise ValueError(
                "Variable " + variable + " not found in the optimization variables."
            )

        flattened = self._flattened_variables[variable]

        if flattened[0] == 1:
            return next(flattened[1]())
        else:
            generator = flattened[1]()
            final_value = None
            for value in generator:
                final_value = value
            return final_value

    def set_initial_guess(
        self, initial_guess: TOptimizationObject | list[TOptimizationObject]
    ):
        self._optimization_solver.set_initial_guess(initial_guess=initial_guess)

    def get_initial_guess(self) -> TOptimizationObject | list[TOptimizationObject]:
        return self._optimization_solver.get_initial_guess()

    def solve(self) -> None:
        self._optimization_solver.solve()

    def get_values(self) -> TOptimizationObject | list[TOptimizationObject] | None:
        return self._optimization_solver.get_values()

    def get_cost_value(self) -> float | None:
        return self._optimization_solver.get_cost_value()

    def add_cost(self, input_cost: cs.MX, name: str = None):
        self._optimization_solver.add_cost(input_cost=input_cost, name=name)

    def add_constraint(self, input_constraint: cs.MX, name: str = None):
        self._optimization_solver.add_constraint(
            input_constraint=input_constraint, name=name
        )

    def cost_function(self) -> cs.MX:
        return self._optimization_solver.cost_function()

    def get_cost_expressions(self) -> dict[str, cs.MX]:
        return self._optimization_solver.get_cost_expressions()

    def get_constraint_expressions(self) -> dict[str, cs.MX]:
        return self._optimization_solver.get_constraint_expressions()

    def get_cost_values(self) -> dict[str, float]:
        return self._optimization_solver.get_cost_values()

    def get_constraint_multipliers(self) -> dict[str, np.ndarray]:
        return self._optimization_solver.get_constraint_multipliers()
