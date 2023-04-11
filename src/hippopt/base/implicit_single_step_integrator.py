import abc
from typing import Dict

import casadi as cs

from hippopt.base.single_step_integrator import SingleStepIntegrator


class ImplicitSingleStepIntegrator(SingleStepIntegrator):
    @abc.abstractmethod
    def step(
        self,
        dt: cs.MX,
        t0: cs.MX,
        x0: Dict[str, cs.MX],
        xf: Dict[str, cs.MX],
        **kwargs  # additional arguments for compatibility
    ) -> Dict[str, cs.MX]:
        pass
