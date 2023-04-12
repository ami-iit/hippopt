import abc
from typing import Dict, Type, TypeVar

import casadi as cs

from hippopt.base.dynamics import Dynamics

TSingleStepIntegrator = TypeVar("TSingleStepIntegrator", bound="SingleStepIntegrator")


class SingleStepIntegrator(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def create(cls, dynamics: Dynamics) -> TSingleStepIntegrator:
        pass

    @abc.abstractmethod
    def step(
        self,
        x0: Dict[str, cs.MX],
        xf: Dict[str, cs.MX],
        dt: cs.MX,
        t0: cs.MX = cs.MX(0.0),
    ) -> Dict[str, cs.MX]:
        pass


def step(
    cls: Type[SingleStepIntegrator],
    dynamics: Dynamics,
    x0: Dict[str, cs.MX],
    xf: Dict[str, cs.MX],
    dt: cs.MX,
    t0: cs.MX = cs.MX(0.0),
) -> Dict[str, cs.MX]:
    integrator = cls.create(dynamics=dynamics)
    return integrator.step(x0=x0, xf=xf, dt=dt, t0=t0)
