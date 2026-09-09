#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit local cellular source bindings for structured fiber operators."""

from __future__ import annotations

import abc

import equinox as eqx
from jaxtyping import Array

from ...._strict import StrictModule
from ..cellular import ShortenFastTwitchModel


class AbstractFiberReaction(StrictModule):
    """One source-bound local cell, never a full-bundle reaction solve.

    Time is ms, current is uA/cm², stretch is dimensionless and its signed
    material rate is per ms. A non-isometric source binding must explicitly
    convert these kinematics to its own sarcomere-length/velocity inputs.
    Source constants remain dynamic array leaves; source/layout/kinematic
    semantics must be static fields so preparation identifies both separately.
    """

    source_id: str = eqx.field(static=True)
    state_count: int = eqx.field(static=True)
    voltage_index: int = eqx.field(static=True)

    @abc.abstractmethod
    def initialize(self, batch_shape: tuple[int, ...], /) -> Array:
        """Return the source initial state with the supplied leading axes."""
        raise NotImplementedError

    @abc.abstractmethod
    def rhs(
        self,
        time_ms: Array,
        values: Array,
        current_uA_per_cm2: Array,
        stretch: Array,
        stretch_rate_per_ms: Array,
        /,
    ) -> Array:
        """Evaluate one local state vector, without spatial coupling."""
        raise NotImplementedError

    @abc.abstractmethod
    def admissible(
        self,
        time_ms: Array,
        values: Array,
        current_uA_per_cm2: Array,
        stretch: Array,
        stretch_rate_per_ms: Array,
        /,
    ) -> Array:
        """Return a scalar source-support predicate for one cell."""
        raise NotImplementedError


class Shorten2007FiberReaction(AbstractFiberReaction):
    """Pinned isometric Shorten kinetics; geometry does not alter this source.

    Moving cable diffusion is a numerical geometry capability, not an added
    stretch-feedback law in the original 56-state cellular model.
    """

    model: ShortenFastTwitchModel
    kinematic_policy: str = eqx.field(static=True)

    def __init__(self, model: ShortenFastTwitchModel, /):
        if not isinstance(model, ShortenFastTwitchModel):
            raise TypeError("model must be ShortenFastTwitchModel.")
        self.model = model
        self.source_id = model.model_id
        self.state_count = 56
        self.voltage_index = 0
        self.kinematic_policy = "source-isometric-no-length-or-velocity-feedback"

    def initialize(self, batch_shape: tuple[int, ...], /) -> Array:
        return self.model.initialize(batch_shape)

    def rhs(
        self, time_ms, values, current_uA_per_cm2, stretch, stretch_rate_per_ms, /
    ) -> Array:
        del stretch, stretch_rate_per_ms
        return self.model.rhs(
            time_ms, values, stimulus_current_uA_per_cm2=current_uA_per_cm2
        )

    def admissible(
        self, time_ms, values, current_uA_per_cm2, stretch, stretch_rate_per_ms, /
    ) -> Array:
        rates = self.rhs(
            time_ms, values, current_uA_per_cm2, stretch, stretch_rate_per_ms
        )
        return self.model.admissible(values, precomputed_rates=rates)


__all__ = ["AbstractFiberReaction", "Shorten2007FiberReaction"]
