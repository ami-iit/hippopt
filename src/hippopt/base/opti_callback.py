import abc
import logging
import weakref
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


class CallbackCriterion(abc.ABC):
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
        self, stopping_criterion: "CallbackCriterion"
    ) -> "CombinedCallbackCriterion":
        if not isinstance(stopping_criterion, CallbackCriterion):
            raise TypeError(stopping_criterion)

        return OrCombinedCallbackCriterion(lhs=self, rhs=stopping_criterion)

    def __ror__(self, other):
        return self.__or__(other)

    def __and__(
        self, stopping_criterion: "CallbackCriterion"
    ) -> "CombinedCallbackCriterion":
        if not isinstance(stopping_criterion, CallbackCriterion):
            raise TypeError(stopping_criterion)

        return AndCombinedCallbackCriterion(lhs=self, rhs=stopping_criterion)

    def __rand__(self, other):
        return self.__and__(other)

    def set_opti(self, opti: cs.Opti) -> None:
        """"""
        # In theory, the callback is included in opti,
        # so the weakref is to avoid circular references
        self.opti = weakref.proxy(opti)
        self.reset()


class BestCost(CallbackCriterion):
    """"""

    def __init__(self) -> None:
        """"""

        CallbackCriterion.__init__(self)

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


class AcceptableCost(CallbackCriterion):
    """"""

    def __init__(self, acceptable_cost: float = cs.inf) -> None:
        """"""

        CallbackCriterion.__init__(self)

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


class AcceptablePrimalInfeasibility(CallbackCriterion):
    """"""

    def __init__(self, acceptable_primal_infeasibility: float = cs.inf) -> None:
        """"""

        CallbackCriterion.__init__(self)

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


class BestPrimalInfeasibility(CallbackCriterion):
    """"""

    def __init__(self) -> None:
        """"""

        CallbackCriterion.__init__(self)

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


class CombinedCallbackCriterion(CallbackCriterion, abc.ABC):
    """"""

    def __init__(self, lhs: CallbackCriterion, rhs: CallbackCriterion) -> None:
        """"""

        CallbackCriterion.__init__(self)
        self.lhs = lhs
        self.rhs = rhs

    @final
    def reset(self) -> None:
        """"""

        self.lhs.reset()
        self.rhs.reset()

    @final
    def update(self) -> None:
        """"""

        self.lhs.update()
        self.rhs.update()

    @final
    def set_opti(self, opti: cs.Opti) -> None:
        """"""

        self.lhs.set_opti(opti)
        self.rhs.set_opti(opti)


class OrCombinedCallbackCriterion(CombinedCallbackCriterion):
    """"""

    @final
    def satisfied(self) -> bool:
        """"""

        return self.lhs.satisfied() or self.rhs.satisfied()


class AndCombinedCallbackCriterion(CombinedCallbackCriterion):
    """"""

    @final
    def satisfied(self) -> bool:
        """"""

        return self.lhs.satisfied() and self.rhs.satisfied()


class SaveBestUnsolvedVariablesCallback(Callback):
    """Class to save the best unsolved variables."""

    def __init__(
        self,
        criterion: CallbackCriterion,
        opti: cs.Opti,
        optimization_objects: list[cs.MX],
        costs: list[cs.MX],
        constraints: list[cs.MX],
    ) -> None:
        """"""

        Callback.__init__(self)

        self.criterion = criterion
        # In theory, the callback is included in opti,
        # so the weakref is to avoid circular references
        self.opti = weakref.proxy(opti)
        self.criterion.set_opti(opti)
        self.optimization_objects = optimization_objects
        self.cost = costs
        self.constraints = constraints

        self.best_iteration = None
        self.best_objects = {}
        self.best_cost = None
        self.best_cost_values = {}
        self.best_constraint_multipliers = {}
        self.ignore_map = {obj: False for obj in self.optimization_objects}

    def call(self, i: int) -> None:
        """"""

        if self.criterion.satisfied():
            self.criterion.update()

            _logger = logging.getLogger(f"[hippopt::{self.__class__.__name__}]")
            _logger.info(f"[i={i}] New best intermediate variables")

            self.best_iteration = i
            self.best_cost = self.opti.debug.value(self.opti.f)
            self.best_objects = {}
            for optimization_object in self.optimization_objects:
                if self.ignore_map[optimization_object]:
                    continue
                try:
                    self.best_objects[optimization_object] = self.opti.debug.value(
                        optimization_object
                    )
                except Exception as err:  # noqa
                    self.ignore_map[optimization_object] = True

            self.best_cost_values = {
                cost: self.opti.debug.value(cost) for cost in self.cost
            }
            self.best_constraint_multipliers = {
                constraint: self.opti.debug.value(self.opti.dual(constraint))
                for constraint in self.constraints
            }
