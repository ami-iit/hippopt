import dataclasses

import casadi as cs

from hippopt.base.dynamics import Dynamics
from hippopt.base.single_step_integrator import (
    SingleStepIntegrator,
    TSingleStepIntegrator,
)


@dataclasses.dataclass
class ImplicitTrapezoid(SingleStepIntegrator):
    _f: Dynamics = dataclasses.field(default=None)
    f: dataclasses.InitVar[Dynamics] = dataclasses.field(default=None)

    def __post_init__(self, f: Dynamics):
        self._f = f

    @classmethod
    def create(cls, dynamics: Dynamics) -> TSingleStepIntegrator:
        return cls(f=dynamics)

    def step(
        self,
        x0: dict[str, cs.MX],
        xf: dict[str, cs.MX],
        dt: cs.MX,
        t0: cs.MX = 0.0,
    ) -> dict[str, cs.MX]:
        f_initial = self._f.evaluate(variables=x0, time=t0)
        f_final = self._f.evaluate(variables=xf, time=t0 + dt)

        output = {
            x: x0[x] + 0.5 * dt * (f_initial[x] + f_final[x])
            for x in self._f.state_variables()
        }

        return output
