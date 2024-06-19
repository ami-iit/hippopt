import dataclasses

import casadi as cs
import numpy as np

from hippopt import (
    CompositeType,
    OptimizationObject,
    OptiSolver,
    OverridableParameter,
    OverridableVariable,
    Parameter,
    StorageType,
    Variable,
    default_composite_field,
    default_storage_field,
)


@dataclasses.dataclass
class CustomVariable(OptimizationObject):
    variable: StorageType = default_storage_field(cls=Variable)
    parameter: StorageType = default_storage_field(cls=Parameter)
    scalar: StorageType = default_storage_field(cls=Variable)

    def __post_init__(self):
        self.variable = np.ones(shape=3)
        self.parameter = np.ones(shape=3)
        self.scalar = 1.0


@dataclasses.dataclass
class AggregateClass(OptimizationObject):
    aggregated: CompositeType[CustomVariable] = default_composite_field(
        factory=CustomVariable
    )
    aggregated_list: CompositeType[list[CustomVariable]] = default_composite_field(
        factory=list
    )
    other_parameter: StorageType = default_storage_field(cls=Parameter)
    other: str = ""

    def __post_init__(self):
        self.other_parameter = np.ones(3)
        self.other = "untouched"
        for _ in range(3):
            self.aggregated_list.append(CustomVariable())


def test_generate_objects():
    test_var = AggregateClass()
    test_var_as_list = test_var.to_list()
    assert (
        len(test_var_as_list) == 3 + 3 * 3 + 1
    )  # 3 for aggregated, 3*3 for aggregated_list, 1 for other_parameter
    solver = OptiSolver()
    opti_var = solver.generate_optimization_objects(test_var)
    assert isinstance(opti_var.aggregated.parameter, cs.MX)
    assert opti_var.aggregated.parameter.shape == (3, 1)
    assert isinstance(opti_var.aggregated.variable, cs.MX)
    assert opti_var.aggregated.variable.shape == (3, 1)
    assert isinstance(opti_var.other_parameter, cs.MX)
    assert opti_var.other_parameter.shape == (3, 1)
    assert isinstance(opti_var.aggregated.scalar, cs.MX)
    assert opti_var.aggregated.scalar.shape == (1, 1)
    assert isinstance(opti_var.aggregated_list, list)
    for opti_var_list in opti_var.aggregated_list:
        assert isinstance(opti_var_list.parameter, cs.MX)
        assert opti_var_list.parameter.shape == (3, 1)
        assert isinstance(opti_var_list.variable, cs.MX)
        assert opti_var_list.variable.shape == (3, 1)
        assert isinstance(opti_var_list.scalar, cs.MX)
        assert opti_var_list.scalar.shape == (1, 1)
    assert opti_var.other == "untouched"
    assert solver.get_optimization_objects() is opti_var
    assert len(solver.get_free_parameters_names()) == 0
    assert (len(opti_var.to_list())) == len(test_var_as_list)
    expected_len = 7 + 3 * 7 + 3
    assert opti_var.to_mx().shape == (expected_len, 1)


def test_generate_objects_list():
    test_var_list = []
    for _ in range(2):
        test_var_list.append(AggregateClass())
    solver = OptiSolver()
    opti_var_list = solver.generate_optimization_objects(test_var_list)
    assert len(opti_var_list) == 2
    for opti_var in opti_var_list:
        assert isinstance(opti_var.aggregated.parameter, cs.MX)
        assert opti_var.aggregated.parameter.shape == (3, 1)
        assert isinstance(opti_var.aggregated.variable, cs.MX)
        assert opti_var.aggregated.variable.shape == (3, 1)
        assert isinstance(opti_var.aggregated.scalar, cs.MX)
        assert opti_var.aggregated.scalar.shape == (1, 1)
        assert isinstance(opti_var.other_parameter, cs.MX)
        assert opti_var.other_parameter.shape == (3, 1)
        assert opti_var.other == "untouched"
    assert solver.get_optimization_objects() is opti_var_list


@dataclasses.dataclass
class CustomOverridableVariable(OptimizationObject):
    overridable: StorageType = default_storage_field(cls=OverridableVariable)
    not_overridable: StorageType = default_storage_field(cls=Variable)

    def __post_init__(self):
        self.overridable = 0.0
        self.not_overridable = 0.0


@dataclasses.dataclass
class CustomCompositeOverridableVariable(OptimizationObject):
    composite: CompositeType[CustomOverridableVariable] = default_composite_field(
        cls=Parameter, factory=CustomOverridableVariable
    )


def test_generate_variables_overridable():
    test_var = CustomCompositeOverridableVariable()
    solver = OptiSolver()
    opti_var = solver.generate_optimization_objects(test_var)
    assert isinstance(opti_var.composite.overridable, cs.MX)
    assert opti_var.composite.overridable.shape == (1, 1)
    assert (
        solver.get_object_type(opti_var.composite.overridable)
        == Parameter.StorageTypeValue
    )
    assert isinstance(opti_var.composite.not_overridable, cs.MX)
    assert opti_var.composite.not_overridable.shape == (1, 1)
    assert (
        solver.get_object_type(opti_var.composite.not_overridable)
        == Variable.StorageTypeValue
    )


@dataclasses.dataclass
class CustomOverridableParameter(OptimizationObject):
    overridable: StorageType = default_storage_field(cls=OverridableParameter)
    not_overridable: StorageType = default_storage_field(cls=Parameter)

    def __post_init__(self):
        self.overridable = 0.0
        self.not_overridable = 0.0


@dataclasses.dataclass
class CustomCompositeOverridableParameter(OptimizationObject):
    composite: CompositeType[CustomOverridableParameter] = default_composite_field(
        cls=Variable, factory=CustomOverridableParameter
    )


def test_generate_parameters_overridable():
    test_var = CustomCompositeOverridableParameter()
    solver = OptiSolver()
    opti_var = solver.generate_optimization_objects(test_var)
    assert isinstance(opti_var.composite.overridable, cs.MX)
    assert opti_var.composite.overridable.shape == (1, 1)
    assert (
        solver.get_object_type(opti_var.composite.overridable)
        == Variable.StorageTypeValue
    )
    assert isinstance(opti_var.composite.not_overridable, cs.MX)
    assert opti_var.composite.not_overridable.shape == (1, 1)
    assert (
        solver.get_object_type(opti_var.composite.not_overridable)
        == Parameter.StorageTypeValue
    )


@dataclasses.dataclass
class CustomCustomOverridableVariableInner(OptimizationObject):
    composite: CompositeType[CustomOverridableVariable] = default_composite_field(
        cls=OverridableVariable, factory=CustomOverridableVariable
    )


@dataclasses.dataclass
class CustomCustomOverridableVariableNested(OptimizationObject):
    composite: CompositeType[CustomCustomOverridableVariableInner] = (
        default_composite_field(
            cls=OverridableParameter, factory=CustomCustomOverridableVariableInner
        )
    )


def test_generate_nested_overridable_class():
    test_var = CustomCustomOverridableVariableNested()
    solver = OptiSolver()
    opti_var = solver.generate_optimization_objects(test_var)
    assert isinstance(opti_var.composite.composite.overridable, cs.MX)
    assert opti_var.composite.composite.overridable.shape == (1, 1)
    assert (
        solver.get_object_type(opti_var.composite.composite.overridable)
        == Parameter.StorageTypeValue
    )
    assert isinstance(opti_var.composite.composite.not_overridable, cs.MX)
    assert opti_var.composite.composite.not_overridable.shape == (1, 1)
    assert (
        solver.get_object_type(opti_var.composite.composite.not_overridable)
        == Variable.StorageTypeValue
    )

    _, metadata_dict = test_var.to_dicts()
    assert (
        metadata_dict["composite.composite.overridable"][
            OptimizationObject.StorageTypeField
        ]
        == Parameter.StorageTypeValue
    )
    assert (
        metadata_dict["composite.composite.not_overridable"][
            OptimizationObject.StorageTypeField
        ]
        == Variable.StorageTypeValue
    )
