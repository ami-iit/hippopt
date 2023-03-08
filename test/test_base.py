from hippopt import Variable, Parameter, StorageType, default_storage_type
import dataclasses
import numpy as np


@dataclasses.dataclass
class TestVariable(Variable):
    storage: StorageType = default_storage_type(Variable)


@dataclasses.dataclass
class TestParameter(Parameter):
    storage: StorageType = default_storage_type(Parameter)


def test_zero_variable():
    test_var = TestVariable()
    test_var.storage = np.ones(shape=3)
    test_var_zero = test_var.zero_copy()
    assert np.all(test_var_zero.storage == 0)


def test_zero_parameter():
    test_par = TestParameter()
    test_par.storage = np.ones(shape=3)
    test_par_zero = test_par.zero_copy()
    assert np.all(test_par_zero.storage == 0)
