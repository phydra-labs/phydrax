#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..integration._api import IntegrationRealization
from ..integration._targets import (
    DensityTarget,
    DiscreteMeasureTarget,
    WeightedSampleTarget,
)
from ._costs import AbstractGroundCost, GroundCost, PrecomputedCost
from ._geometry import cost_matrix
from ._measure import _FiniteTransportMeasure, EventEncoder, lower_transport_measure
from ._problem import TransportProblemProvenance


_TransportMeasureInput = (
    DiscreteMeasureTarget | WeightedSampleTarget | DensityTarget | IntegrationRealization
)


class UnbalancedTransportProblem(StrictModule):
    """Finite unbalanced transport with two physical masses and KL relaxations."""

    source: _FiniteTransportMeasure
    target: _FiniteTransportMeasure
    cost: GroundCost
    source_marginal_penalty: Array
    target_marginal_penalty: Array
    provenance: TransportProblemProvenance

    def __init__(
        self,
        source: _FiniteTransportMeasure,
        target: _FiniteTransportMeasure,
        cost: GroundCost,
        /,
        *,
        source_marginal_penalty: ArrayLike,
        target_marginal_penalty: ArrayLike,
    ):
        if not isinstance(source, _FiniteTransportMeasure):
            raise TypeError("source must be a canonical finite transport measure.")
        if not isinstance(target, _FiniteTransportMeasure):
            raise TypeError("target must be a canonical finite transport measure.")
        if not isinstance(cost, (AbstractGroundCost, PrecomputedCost)):
            raise TypeError("cost must be an AbstractGroundCost or PrecomputedCost.")
        if isinstance(cost, PrecomputedCost):
            if cost.shape != (source.num_atoms, target.num_atoms):
                raise ValueError(
                    "Precomputed cost shape must match source and target atom counts."
                )
        elif source.feature_size != target.feature_size:
            raise ValueError(
                "Source and target features must have equal size for a ground cost."
            )
        source_penalty = jnp.asarray(source_marginal_penalty, dtype=float).reshape(())
        target_penalty = jnp.asarray(target_marginal_penalty, dtype=float).reshape(())
        source_penalty = eqx.error_if(
            source_penalty,
            ~jnp.isfinite(source_penalty) | (source_penalty <= 0.0),
            "source_marginal_penalty must be finite and strictly positive.",
        )
        target_penalty = eqx.error_if(
            target_penalty,
            ~jnp.isfinite(target_penalty) | (target_penalty <= 0.0),
            "target_marginal_penalty must be finite and strictly positive.",
        )
        self.source = source
        self.target = target
        self.cost = cost
        self.source_marginal_penalty = source_penalty
        self.target_marginal_penalty = target_penalty
        self.provenance = TransportProblemProvenance(
            source.provenance,
            target.provenance,
            cost.cost_id,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.source.num_atoms, self.target.num_atoms

    @property
    def source_mass(self) -> Array:
        return self.source.mass

    @property
    def target_mass(self) -> Array:
        return self.target.mass

    @property
    def source_weights(self) -> Array:
        return self.source.physical_weights

    @property
    def target_weights(self) -> Array:
        return self.target.physical_weights

    def cost_matrix(self) -> Array:
        """Materialize the complete ground-cost matrix."""
        return cost_matrix(
            self.cost,
            self.source.points,
            self.target.points,
        )

    def cost_at(self, source_indices: Array, target_indices: Array, /) -> Array:
        """Evaluate costs on broadcast-compatible source and target indices."""
        source_indices_ = jnp.asarray(source_indices, dtype=jnp.int32)
        target_indices_ = jnp.asarray(target_indices, dtype=jnp.int32)
        if isinstance(self.cost, PrecomputedCost):
            return self.cost.values[source_indices_, target_indices_]
        source_points = self.source.points[source_indices_]
        target_points = self.target.points[target_indices_]
        flat_source = source_points.reshape((-1, source_points.shape[-1]))
        flat_target = target_points.reshape((-1, target_points.shape[-1]))
        values = jax.vmap(self.cost.pairwise)(flat_source, flat_target)
        return values.reshape(
            jnp.broadcast_shapes(source_indices_.shape, target_indices_.shape)
        )


def unbalanced_problem(
    source: _TransportMeasureInput,
    target: _TransportMeasureInput,
    /,
    *,
    cost: GroundCost,
    source_marginal_penalty: ArrayLike,
    target_marginal_penalty: ArrayLike,
    source_encoder: EventEncoder | None = None,
    target_encoder: EventEncoder | None = None,
) -> UnbalancedTransportProblem:
    """Construct unbalanced transport without equating or normalizing physical masses."""
    source_measure = lower_transport_measure(
        source,
        encoder=source_encoder,
        name="source",
    )
    target_measure = lower_transport_measure(
        target,
        encoder=target_encoder,
        name="target",
    )
    return UnbalancedTransportProblem(
        source_measure,
        target_measure,
        cost,
        source_marginal_penalty=source_marginal_penalty,
        target_marginal_penalty=target_marginal_penalty,
    )


__all__ = [
    "UnbalancedTransportProblem",
    "unbalanced_problem",
]
