import abc
import dataclasses
from typing import TypeVar

import casadi as cs

TDynamics = TypeVar("TDynamics", bound="Dynamics")
TDynamicsLHS = TypeVar("TDynamicsLHS", bound="DynamicsLHS")


@dataclasses.dataclass
class DynamicsRHS:
    _f: cs.Function = dataclasses.field(default=None)
    _names_map: dict[str, str] = dataclasses.field(default=None)
    _names_map_inv: dict[str, str] = dataclasses.field(default=None)
    f: dataclasses.InitVar[cs.Function] = None
    names_map_in: dataclasses.InitVar[dict[str, str]] = None

    def __post_init__(self, f: cs.Function, names_map_in: dict[str, str]):
        """
        Create the DynamicsRHS object
        :param f: The CasADi function describing the dynamics. The output order should match the list provided
         in the dot function.
        :param names_map_in: A dict describing how to switch from the input names to those used in the function.
         The key is the name provided by the user, while the value is the input name expected by the function.
         It is also possible to specify labels for nested variables using ".", e.g. "a.b" corresponds
         to the variable "b" within "a".
         This is valid only for the keys.
         If time is an input, its label needs to be provided using the "dot" function.
        :return: Nothing
        """
        self._f = f
        self._names_map = names_map_in if names_map_in else {}
        self._names_map_inv = {v: k for k, v in self._names_map.items()}  # inverse dict

    def evaluate(
        self, variables: dict[str, cs.MX], time: cs.MX, time_name: str
    ) -> dict[str, cs.MX]:
        input_dict = {}

        for name in self.input_names():
            if name == time_name:
                input_dict[time_name] = time
                continue

            if name not in variables:
                raise ValueError(name + " input not found in variables.")
            key = name if name not in self._names_map else self._names_map[name]
            input_dict[key] = variables[name]

        return self._f(**input_dict)

    def input_names(self) -> list[str]:
        function_inputs = self._f.name_in()
        output = []
        for el in function_inputs:
            output_name = self._names_map_inv[el] if el in self._names_map_inv else el
            output.append(output_name)

        return output

    def outputs(self) -> list[str]:
        return self._f.name_out()


@dataclasses.dataclass
class DynamicsLHS:
    _x: list[str] = dataclasses.field(default=None)
    x: dataclasses.InitVar[list[str] | str] = None
    _t_label: str = "t"
    t_label: dataclasses.InitVar[str] = None

    def __post_init__(self, x: list[str] | str, t_label: str):
        """
        Constructs the DynamicsLHS object
        :param x: List of variable names on the left hand side of dot{x} = f(y).
          The list can contain empty strings if some output of f needs to be discarded. If one output
          needs to be mapped to a nested item, use "." as separator, e.g. "a.b"
        :param t_label: The label of the time variable. Default "t"
        :return: Nothing
        """
        self._x = x if isinstance(x, list) else [x]
        self._t_label = t_label if isinstance(t_label, str) else "t"

    def equal(self, f: cs.Function, names_map: dict[str, str] = None) -> TDynamics:
        rhs = DynamicsRHS(f=f, names_map_in=names_map)
        if len(rhs.outputs()) != len(self._x):
            raise ValueError(
                "The number of outputs of the dynamics function does not match the specified number of state variables."
            )
        return TypedDynamics(lhs=self, rhs=rhs)

    def __eq__(
        self, other: cs.Function | tuple[cs.Function, dict[str, str]]
    ) -> TDynamics:
        if isinstance(other, tuple):
            return self.equal(f=other[0], names_map=other[1])

        assert isinstance(other, cs.Function)
        return self.equal(f=other)

    def state_variables(self) -> list[str]:
        return self._x

    def time_label(self) -> str:
        return self._t_label


def dot(x: str | list[str], t: str = "t") -> TDynamicsLHS:
    return DynamicsLHS(x=x, t_label=t)


class Dynamics(abc.ABC):
    @abc.abstractmethod
    def state_variables(self) -> list[str]:
        pass

    @abc.abstractmethod
    def input_names(self) -> list[str]:
        pass

    @abc.abstractmethod
    def time_name(self) -> str:
        pass

    @abc.abstractmethod
    def evaluate(self, variables: dict[str, cs.MX], time: cs.MX) -> dict[str, cs.MX]:
        pass


@dataclasses.dataclass
class TypedDynamics(Dynamics):
    _lhs: DynamicsLHS = dataclasses.field(default=None)
    lhs: dataclasses.InitVar[DynamicsLHS] = None
    _rhs: DynamicsRHS = dataclasses.field(default=None)
    rhs: dataclasses.InitVar[DynamicsRHS] = None

    def __post_init__(self, lhs: DynamicsLHS, rhs: DynamicsRHS):
        self._lhs = lhs
        self._rhs = rhs

    def state_variables(self) -> list[str]:
        return [x for x in self._lhs.state_variables() if x]  # Remove empty strings

    def input_names(self) -> list[str]:
        return self._rhs.input_names()

    def time_name(self) -> str:
        return self._lhs.time_label()

    def evaluate(self, variables: dict[str, cs.MX], time: cs.MX) -> dict[str, cs.MX]:
        f_output = self._rhs.evaluate(
            variables=variables, time=time, time_name=self._lhs.time_label()
        )

        assert (
            len(f_output)
            == len(self._rhs.outputs())
            == len(self._lhs.state_variables())
        )

        output_dict = {}

        for i in range(len(f_output)):
            key = self._lhs.state_variables()[i]
            if key:
                output_dict[key] = f_output[self._rhs.outputs()[i]]

        return output_dict
