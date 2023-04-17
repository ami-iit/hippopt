import dataclasses
from typing import Dict

import casadi as cs

from hippopt.base.dynamics import Dynamics
from hippopt.base.single_step_integrator import (
    SingleStepIntegrator,
    TSingleStepIntegrator,
)


@dataclasses.dataclass
class ForwardEuler(SingleStepIntegrator):
    _f: Dynamics = dataclasses.field(default=None)
    f: dataclasses.InitVar[Dynamics] = dataclasses.field(default=None)

    def __post_init__(self, f: Dynamics):
        self._f = f

    @classmethod
    def create(cls, dynamics: Dynamics) -> TSingleStepIntegrator:
        return cls(f=dynamics)

    def step(
        self,
        x0: Dict[str, cs.MX],
        xf: Dict[str, cs.MX],  # xf not used
        dt: cs.MX,
        t0: cs.MX = 0.0,
    ) -> Dict[str, cs.MX]:
        f = self._f.evaluate(variables=x0, time=t0)

        output = {x: x0[x] + dt * f[x] for x in self._f.state_variables()}

        return output
