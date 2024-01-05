import abc
import logging
from typing import final

import casadi as cs


class Callback(cs.OptiCallback, abc.ABC):
    """Abstract class of an Opti callback."""

    def __init__(self) -> None:
        cs.OptiCallback.__init__(self)

    @final
    def __call__(self, i: int) -> None:
        self.call(i)

    @abc.abstractmethod
    def call(self, i) -> None:
        pass


class StoppingCriterion(abc.ABC):
    """"""

    def __init__(self) -> None:
        """"""
        self.opti = None

    @abc.abstractmethod
    def satisfied(self) -> bool:
        pass

    @abc.abstractmethod
    def update(self) -> None:
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        pass

    def __or__(
        self, stopping_criterion: "StoppingCriterion"
    ) -> "CombinedStoppingCriterion":
        if not isinstance(stopping_criterion, StoppingCriterion):
            raise TypeError(stopping_criterion)

        return CombinedStoppingCriterion([self, stopping_criterion])

    def set_opti(self, opti: cs.Opti) -> None:
        """"""
        self.opti = opti


class BestCost(StoppingCriterion):
    """"""

    def __init__(self) -> None:
        """"""

        StoppingCriterion.__init__(self)

        self.best_cost = None
        self.reset()

    @final
    def reset(self) -> None:
        """"""

        self.best_cost = cs.inf

    @final
    def satisfied(self) -> bool:
        """"""

        return self._get_current_cost() < self.best_cost

    @final
    def update(self) -> None:
        """"""

        best_cost = self._get_current_cost()

        _logger = logging.getLogger(f"[hippopt::{self.__class__.__name__}]")
        _logger.debug(f"New best cost: {best_cost} (old: {self.best_cost})")

        self.best_cost = self._get_current_cost()

    def _get_current_cost(self) -> float:
        """"""

        return self.opti.debug.value(self.opti.f)


class AcceptableCost(StoppingCriterion):
    """"""

    def __init__(self, acceptable_cost: float = cs.inf) -> None:
        """"""

        StoppingCriterion.__init__(self)

        self.acceptable_cost = acceptable_cost

        self.best_acceptable_cost = None
        self.reset()

    @final
    def reset(self) -> None:
        """"""

        self.best_acceptable_cost = cs.inf

    def satisfied(self) -> bool:
        """"""

        return self._get_current_cost() < self.acceptable_cost

    def update(self) -> None:
        """"""

        current_cost = self._get_current_cost()

        if current_cost < self.best_acceptable_cost:
            _logger = logging.getLogger(f"[hippopt::{self.__class__.__name__}]")
            _logger.debug(
                f"[New acceptable cost: {current_cost}"
                f" (old: {self.best_acceptable_cost})"
            )

            self.best_acceptable_cost = current_cost

    def _get_current_cost(self) -> float:
        """"""

        return self.opti.debug.value(self.opti.f)


class AcceptablePrimalInfeasibility(StoppingCriterion):
    """"""

    def __init__(self, acceptable_primal_infeasibility: float = cs.inf) -> None:
        """"""

        StoppingCriterion.__init__(self)

        self.acceptable_primal_infeasibility = acceptable_primal_infeasibility

        self.best_acceptable_primal_infeasibility = None
        self.reset()

    @final
    def reset(self) -> None:
        """"""

        self.best_acceptable_primal_infeasibility = cs.inf

    def satisfied(self) -> bool:
        """"""

        return (
            self._get_current_primal_infeasibility()
            < self.acceptable_primal_infeasibility
        )

    def update(self) -> None:
        """"""

        current_primal_infeasibility = self._get_current_primal_infeasibility()

        if current_primal_infeasibility < self.best_acceptable_primal_infeasibility:
            _logger = logging.getLogger(f"[hippopt::{self.__class__.__name__}]")
            _logger.debug(
                f"New acceptable primal infeasibility: "
                f"{current_primal_infeasibility} "
                f"(old: {self.best_acceptable_primal_infeasibility})"
            )

            self.best_acceptable_primal_infeasibility = current_primal_infeasibility

    def _get_current_primal_infeasibility(self) -> float:
        """"""

        return self.opti.debug.stats()["iterations"]["inf_pr"][-1]


class BestPrimalInfeasibility(StoppingCriterion):
    """"""

    def __init__(self) -> None:
        """"""

        StoppingCriterion.__init__(self)

        self.best_primal_infeasibility = None
        self.reset()

    @final
    def reset(self) -> None:
        """"""

        self.best_primal_infeasibility = cs.inf

    def satisfied(self) -> bool:
        """"""

        return self._get_current_primal_infeasibility() < self.best_primal_infeasibility

    def update(self) -> None:
        """"""

        best_primal_infeasibility = self._get_current_primal_infeasibility()

        _logger = logging.getLogger(f"[hippopt::{self.__class__.__name__}]")
        _logger.debug(
            f"New best primal infeasibility: {best_primal_infeasibility}"
            f" (old: {self.best_primal_infeasibility})"
        )

        self.best_primal_infeasibility = best_primal_infeasibility

    def _get_current_primal_infeasibility(self) -> float:
        """"""

        return self.opti.debug.stats()["iterations"]["inf_pr"][-1]


class CombinedStoppingCriterion(StoppingCriterion):
    """"""

    def __init__(self, stopping_criteria: list[StoppingCriterion]) -> None:
        """"""

        StoppingCriterion.__init__(self)
        self.stopping_criteria = stopping_criteria

    def __or__(
        self, stopping_criterion: StoppingCriterion
    ) -> "CombinedStoppingCriterion":
        if isinstance(stopping_criterion, CombinedStoppingCriterion):
            ret = CombinedStoppingCriterion(
                stopping_criteria=self.stopping_criteria
                + stopping_criterion.stopping_criteria
            )

        elif isinstance(stopping_criterion, StoppingCriterion):
            ret = CombinedStoppingCriterion(
                stopping_criteria=self.stopping_criteria + [stopping_criterion]
            )

        else:
            raise TypeError(stopping_criterion)

        return ret

    @final
    def reset(self) -> None:
        """"""

        _ = [
            stopping_criterion.reset() for stopping_criterion in self.stopping_criteria
        ]

    @final
    def satisfied(self) -> bool:
        """"""

        return all(
            [
                stopping_criterion.satisfied()
                for stopping_criterion in self.stopping_criteria
            ]
        )

    @final
    def update(self) -> None:
        """"""

        for stopping_criterion in self.stopping_criteria:
            stopping_criterion.update()

    @final
    def set_opti(self, opti: cs.Opti) -> None:
        """"""

        for stopping_criterion in self.stopping_criteria:
            stopping_criterion.set_opti(opti)


class SaveBestUnsolvedVariablesCallback(Callback):
    """Class to save the best unsolved variables."""

    def __init__(
        self,
        criterion: StoppingCriterion,
        opti: cs.Opti,
        optimization_objects: list[cs.MX],
    ) -> None:
        """"""

        Callback.__init__(self)

        self.criterion = criterion
        self.opti = opti
        self.criterion.set_opti(self.opti)
        self.optimization_objects = optimization_objects

        self.best_stats = None
        self.best_variables = {}

    def call(self, i: int) -> None:
        """"""

        if self.criterion.satisfied():
            self.criterion.update()

            _logger = logging.getLogger(f"[hippopt::{self.__class__.__name__}]")
            _logger.info(f"[i={i}] New best intermediate variables")

            self.best_stats = self.opti.debug.stats()
            self.best_variables = {
                optimization_object: self.opti.debug.value(optimization_object)
                for optimization_object in self.optimization_objects
            }
