#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import equinox as eqx
from jax import Array

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._restoration import RestorationValidationContext


if TYPE_CHECKING:
    from ._generic import FiniteElementDofMap


class AbstractFiniteElementSourceOwnership(StrictModule, NonTrainableState):
    """Provide source DOF frames and validation for a resident projection."""

    plan_id: eqx.AbstractVar[str]
    source_dof_map: eqx.AbstractVar[FiniteElementDofMap]
    source_lifted_routes: eqx.AbstractVar[tuple[Array, ...]]
    source_lifted_frames: eqx.AbstractVar[Array]

    @abstractmethod
    def source_dof_identity_keys(
        self, dof_map: FiniteElementDofMap, /
    ) -> tuple[str, ...]:
        """Return scientific source DOF keys in the supplied map's row order."""
        ...

    @abstractmethod
    def validate_restored(
        self, /, *, validation_context: RestorationValidationContext
    ) -> None:
        """Validate retained source ownership within one restoration replay."""
        ...
