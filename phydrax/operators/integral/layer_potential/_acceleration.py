#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._core import BoundaryPanelization2D
from ._quadrature2d import (
    _panel_direct,
    classify_panel_interactions_2d,
)


class LayerBackendEvaluation2D(StrictModule, NonTrainableState):
    """Backend values and explicit near/far work accounting."""

    values: Array
    error_estimate: Array
    status: Array
    num_evaluations: Array
    accuracy_supported: Array
    near_panel_count: int = eqx.field(static=True)
    far_panel_count: int = eqx.field(static=True)


class AbstractLayerAccelerationBackend(StrictModule, NonTrainableState):
    """Explicit backend contract for far-field layer acceleration."""

    @property
    @abc.abstractmethod
    def backend_id(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(
        self,
        potential: object,
        targets: ArrayLike,
        /,
        *,
        near_ratio: float,
    ) -> LayerBackendEvaluation2D:
        raise NotImplementedError


class DirectNearFarBackend2D(AbstractLayerAccelerationBackend):
    """Corrected near/far decomposition using direct point interactions."""

    @property
    def backend_id(self) -> str:
        return "direct-near-far-2d-v1"

    def evaluate(
        self,
        potential: object,
        targets: ArrayLike,
        /,
        *,
        near_ratio: float,
    ) -> LayerBackendEvaluation2D:
        if not isinstance(potential.panelization, BoundaryPanelization2D):
            raise TypeError("DirectNearFarBackend2D requires 2D panelization.")
        values = jnp.asarray(targets, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
            raise ValueError("Backend targets must have shape (target_count, 2).")
        interactions = classify_panel_interactions_2d(
            potential.panelization,
            values,
            near_ratio=near_ratio,
        )
        output = []
        near_count = 0
        far_count = 0
        for target_index, target in enumerate(values):
            value = jnp.asarray(0.0)
            for panel_id in range(potential.panelization.panel_count):
                near = bool(interactions.near_mask[target_index, panel_id])
                near_count += int(near)
                far_count += int(not near)
                value = value + _panel_direct(potential, target, panel_id)
            output.append(value)
        values_ = jnp.stack(output)
        finite = jnp.all(jnp.isfinite(values_))
        return LayerBackendEvaluation2D(
            values=values_,
            error_estimate=jnp.asarray(0.0),
            status=jnp.asarray(0, dtype=jnp.int32),
            num_evaluations=jnp.asarray(
                values.shape[0] * potential.panelization.node_count,
                dtype=jnp.int32,
            ),
            accuracy_supported=finite,
            near_panel_count=near_count,
            far_panel_count=far_count,
        )


__all__ = [
    "AbstractLayerAccelerationBackend",
    "DirectNearFarBackend2D",
    "LayerBackendEvaluation2D",
]
