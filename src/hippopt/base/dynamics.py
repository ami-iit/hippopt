import abc
import dataclasses
from typing import Dict, List, TypeVar

import casadi as cs

TTime = TypeVar("TTime", bound="Time")
TDynamics = TypeVar("TDynamics", bound="Dynamics")
TDynamicsLHS = TypeVar("TDynamicsLHS", bound="DynamicsLHS")


class Time(abc.ABC):
    @abc.abstractmethod
    def t(self, *args, **kwargs) -> cs.MX:
        pass


@dataclasses.dataclass
class DynamicsRHS:
    _f: cs.Function = dataclasses.field(default=None)
    _names_map: Dict[str, str] = dataclasses.field(default=None)
    f: dataclasses.InitVar[cs.Function] = None
    names_map_in: dataclasses.InitVar[Dict[str, str]] = None

    def __post_init__(self, f: cs.Function, names_map_in: Dict[str, str]):
        """
        Create the DynamicsRHS object
        :param f: The CasADi function describing the dynamics
        :param names_map_in: A map describing how to switch from the names used in the function,
         to those of the optimization objects. The order is irrelevant as we construct a dictionary
         containing both the forward and the reverse map. Hence, the provided map needs to be bijective.
         It is also possible to specify labels for nested variables using "::", e.g. "a::b" corresponds
         to the variable b within a
        :return: Nothing
        """
        self._f = f
        inv_map = {v: k for k, v in names_map_in.items()}  # inverse dict
        self._names_map = names_map_in | inv_map  # merging forward and inverse map

    def function(self) -> cs.Function:
        return self._f

    def input_names(self) -> List[str]:
        function_inputs = self._f.name_in()
        output = []
        for el in function_inputs:
            output_name = self._names_map[el] if el in self._names_map else el
            output.append(output_name)

        return output

    def number_of_outputs(self) -> int:
        return len(self._f.name_out())


@dataclasses.dataclass
class DynamicsLHS:
    _x: List[str] = dataclasses.field(default=None)
    x: dataclasses.InitVar[List[str] | str] = None
    _t: TTime = dataclasses.field(default=None)
    _t_label: str = "t"

    def __post_init__(self, x: List[str] | str):
        """
        Constructs the DynamicsLHS object
        :param x: List of variable names on the left hand side of dot{x} = f(y).
          The list can contain empty strings if some output of f needs to be discarded. If one output
          needs to be mapped to a nested item, use "::" as separator, e.g. "a::b"
        :return: Nothing
        """
        self._x = x if isinstance(x, list) else [x]

    def dt(self, t: TTime, label: str = "t") -> TDynamicsLHS:
        self._t = t
        self._t_label = label
        return self

    def equal(self, f: cs.Function, names_map: Dict[str, str] = None) -> TDynamics:
        rhs = DynamicsRHS(f, names_map)
        if rhs.number_of_outputs() != len(self._x):
            raise ValueError(
                "The number of outputs of the dynamics function does not match the specified number of state variables."
            )
        return Dynamics(lhs=self, rhs=rhs)

    def __eq__(self, other: cs.Function) -> TDynamics:
        return self.equal(f=other)

    def state_variables(self) -> List[str]:
        return self._x

    def time_label(self) -> str:
        return self._t_label


def d(x: str | List[str]) -> TDynamicsLHS:
    return DynamicsLHS(x)


@dataclasses.dataclass
class Dynamics:
    _lhs: DynamicsLHS = dataclasses.field(default=None)
    lhs: dataclasses.InitVar[DynamicsLHS] = None
    _rhs: DynamicsRHS = dataclasses.field(default=None)
    rhs: dataclasses.InitVar[DynamicsRHS] = None

    def __post_init__(self, lhs: DynamicsLHS, rhs: DynamicsRHS):
        self._lhs = lhs
        self._rhs = rhs

    def state_variables(self) -> List[str]:
        return self._lhs.state_variables()

    def input_names(self) -> List[str]:
        return self._rhs.input_names()

    def time_name(self) -> str:
        return self._lhs.time_label()

    def evaluate(self, variables: Dict[str, cs.MX], time: cs.MX) -> Dict[str, cs.MX]:
        input_dict = variables
        input_dict[self.time_name()] = time

        return self._rhs.function()(input_dict)
