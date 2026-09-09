#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from ..._strict import AbstractAttribute, StrictModule
from ..._trainable import NonTrainableState


if TYPE_CHECKING:
    from ._cashflows import CashflowBatch
    from ._resolution import ContractResolutionContext


class AbstractPayoff(StrictModule, NonTrainableState):
    """Immutable product-specific payoff marker without a universal payoff DSL."""

    __strict_abstract__ = True
    payoff_id: AbstractAttribute[str]


class AbstractContract(StrictModule, NonTrainableState):
    """Host-side financial contract definition before deterministic resolution."""

    __strict_abstract__ = True
    contract_id: AbstractAttribute[str]

    @abstractmethod
    def resolve(self, context: ContractResolutionContext, /) -> AbstractResolvedContract:
        """Resolve calendars, references, and known obligations on the host."""

        raise NotImplementedError


class AbstractResolvedContract(StrictModule, NonTrainableState):
    """Host-resolved contract suitable for product-specific fixed-shape preparation."""

    __strict_abstract__ = True
    contract_id: AbstractAttribute[str]
    resolved_id: AbstractAttribute[str]

    @property
    @abstractmethod
    def known_cashflows(self) -> CashflowBatch:
        """Return only currently known obligations, never latent floating formulas."""

        raise NotImplementedError


__all__ = ["AbstractContract", "AbstractPayoff", "AbstractResolvedContract"]
