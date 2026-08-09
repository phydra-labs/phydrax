#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule
from ..integration._api import IntegrationRealization
from ..integration._targets import (
    DensityTarget,
    DiscreteMeasureTarget,
    WeightedSampleTarget,
)
from ._costs import AbstractGroundCost, GroundCost, PrecomputedCost
from ._measure import _FiniteTransportMeasure, EventEncoder, lower_transport_measure


class TransportProblemProvenance(StrictModule):
    """Static source, target, and cost identity for a transport problem."""

    source: str = eqx.field(static=True)
    target: str = eqx.field(static=True)
    cost: str = eqx.field(static=True)

    def __init__(self, source: str, target: str, cost: str, /):
        self.source = str(source)
        self.target = str(target)
        self.cost = str(cost)


class DiscreteTransportProblem(StrictModule):
    """Balanced finite-measure transport problem in canonical coordinates."""

    source: _FiniteTransportMeasure
    target: _FiniteTransportMeasure
    cost: GroundCost
    mass: Array
    provenance: TransportProblemProvenance
    mass_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        source: _FiniteTransportMeasure,
        target: _FiniteTransportMeasure,
        cost: GroundCost,
        /,
        *,
        mass_tolerance: float = 1e-8,
    ):
        if not isinstance(source, _FiniteTransportMeasure):
            raise TypeError("source must be a canonical finite transport measure.")
        if not isinstance(target, _FiniteTransportMeasure):
            raise TypeError("target must be a canonical finite transport measure.")
        if not isinstance(cost, (AbstractGroundCost, PrecomputedCost)):
            raise TypeError("cost must be an AbstractGroundCost or PrecomputedCost.")
        tolerance = float(mass_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("mass_tolerance must be finite and nonnegative.")
        if isinstance(cost, PrecomputedCost):
            if cost.shape != (source.num_atoms, target.num_atoms):
                raise ValueError(
                    "Precomputed cost shape must match source and target atom counts."
                )
        elif source.feature_size != target.feature_size:
            raise ValueError(
                "Source and target features must have equal size for a ground cost."
            )
        mass = eqx.error_if(
            source.mass,
            ~jnp.isclose(
                source.mass,
                target.mass,
                rtol=tolerance,
                atol=tolerance,
            ),
            "Balanced transport requires equal positive source and target mass.",
        )
        self.source = source
        self.target = target
        self.cost = cost
        self.mass = mass
        self.provenance = TransportProblemProvenance(
            source.provenance,
            target.provenance,
            cost.cost_id,
        )
        self.mass_tolerance = tolerance

    @property
    def shape(self) -> tuple[int, int]:
        return self.source.num_atoms, self.target.num_atoms

    @property
    def source_probabilities(self) -> Array:
        return self.source.probabilities

    @property
    def target_probabilities(self) -> Array:
        return self.target.probabilities

    @property
    def source_weights(self) -> Array:
        return self.mass * self.source.probabilities

    @property
    def target_weights(self) -> Array:
        return self.mass * self.target.probabilities

    def cost_matrix(self) -> Array:
        """Materialize the complete ground-cost matrix."""
        if isinstance(self.cost, PrecomputedCost):
            return self.cost.values
        return self.cost.matrix(self.source.points, self.target.points)

    def cost_at(self, source_indices: Array, target_indices: Array, /) -> Array:
        """Evaluate costs on two index arrays with broadcast-compatible shapes."""
        source_indices_, target_indices_ = jnp.broadcast_arrays(
            jnp.asarray(source_indices, dtype=jnp.int32),
            jnp.asarray(target_indices, dtype=jnp.int32),
        )
        if isinstance(self.cost, PrecomputedCost):
            return self.cost.values[source_indices_, target_indices_]
        source_points = self.source.points[source_indices_]
        target_points = self.target.points[target_indices_]
        flat_source = source_points.reshape((-1, source_points.shape[-1]))
        flat_target = target_points.reshape((-1, target_points.shape[-1]))
        values = jax.vmap(self.cost.pairwise)(flat_source, flat_target)
        return values.reshape(source_indices_.shape)


def discrete_problem(
    source: (
        DiscreteMeasureTarget
        | WeightedSampleTarget
        | DensityTarget
        | IntegrationRealization
    ),
    target: (
        DiscreteMeasureTarget
        | WeightedSampleTarget
        | DensityTarget
        | IntegrationRealization
    ),
    /,
    *,
    cost: GroundCost,
    source_encoder: EventEncoder | None = None,
    target_encoder: EventEncoder | None = None,
    mass_tolerance: float = 1e-8,
) -> DiscreteTransportProblem:
    """Construct a balanced transport problem from existing measure contracts."""
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
    return DiscreteTransportProblem(
        source_measure,
        target_measure,
        cost,
        mass_tolerance=mass_tolerance,
    )


__all__ = [
    "DiscreteTransportProblem",
    "TransportProblemProvenance",
    "discrete_problem",
]
