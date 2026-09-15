#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import equinox as eqx
from jaxtyping import Array

from .._strict import StrictModule
from ._dynamics import AtomisticDynamicsState, PreparedAtomisticDynamics


class AbstractAtomisticObserverPlan(StrictModule):
    """Fixed-shape accepted-step statistic integrated into atomistic rollout."""

    observer_id: str = eqx.field(static=True)

    @abstractmethod
    def initialize(
        self,
        dynamics: PreparedAtomisticDynamics,
        state: AtomisticDynamicsState,
        /,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        observer_state: Any,
        dynamics: PreparedAtomisticDynamics,
        state: AtomisticDynamicsState,
        accepted: Array,
        /,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def finalize(self, observer_state: Any, /) -> Any:
        raise NotImplementedError


__all__ = ["AbstractAtomisticObserverPlan"]
