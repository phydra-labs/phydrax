#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from ..._fingerprint import canonical_fingerprint
from ..._probability import AbstractProbabilityLaw
from ..._strict import StrictModule
from ._density import ContinuousFlowDensityResult, ContinuousFlowLaw
from ._transport import ContinuousTransport


class RiemannianFlowDensityResult(StrictModule):
    coordinate_result: ContinuousFlowDensityResult
    log_volume_density: Array
    log_prob: Array
    tangent_residual: Array
    rank_margin: Array
    valid: Array
    status: Array
    geometry_id: str = eqx.field(static=True)
    reference_measure: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)


class RiemannianContinuousFlowLaw(AbstractProbabilityLaw):
    """Same-dimensional flow density relative to represented Riemannian volume."""

    coordinate_law: ContinuousFlowLaw
    manifold: Any
    chart_plan: Any
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        transport: ContinuousTransport,
        manifold: Any,
        /,
        *,
        chart_plan: Any = None,
        max_exact_dimension: int = 32,
        flow_id: str | None = None,
    ):
        if not callable(manifold.local_geometry):
            raise TypeError("manifold must expose local_geometry(point) evidence.")
        coordinate_law = ContinuousFlowLaw(
            transport,
            max_exact_dimension=max_exact_dimension,
            flow_id=flow_id,
        )
        initial_evidence = manifold.local_geometry(
            transport.source_law.sample(jax.random.PRNGKey(0))
        )
        if not bool(initial_evidence.valid):
            raise ValueError("Initial manifold tangent/measure evidence is invalid.")
        resolved_id = flow_id or canonical_fingerprint(
            {
                "kind": "riemannian-continuous-flow-law-v1",
                "transport": transport.transport_id,
                "geometry": manifold.manifold_id,
                "max_exact_dimension": max_exact_dimension,
            }
        )
        self.coordinate_law = coordinate_law
        self.manifold = manifold
        self.chart_plan = chart_plan
        self.law_id = resolved_id

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self.coordinate_law.event_shape

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.coordinate_law.batch_shape

    @property
    def density_measure_kind(self) -> str:
        return "riemannian"

    def sample(self, key: Key[Array, ""], sample_shape: tuple[int, ...] = ()) -> Array:
        return self.coordinate_law.sample(key, sample_shape)

    def log_prob_with_diagnostics(
        self, value: ArrayLike, /
    ) -> RiemannianFlowDensityResult:
        coordinate = self.coordinate_law.log_prob_with_diagnostics(value)
        flat = coordinate.data_state.reshape((-1,) + self.event_shape)

        def one(state):
            evidence = self.manifold.local_geometry(state)
            projected = evidence.tangent_projector @ state.reshape((-1,))
            tangent_residual = jnp.sqrt(jnp.sum((projected - state.reshape((-1,))) ** 2))
            return (
                evidence.log_volume,
                tangent_residual,
                evidence.rank_margin,
                evidence.valid,
            )

        log_volume, tangent_residual, rank_margin, geometry_valid = jax.vmap(one)(flat)
        leading = coordinate.log_prob.shape
        log_volume = log_volume.reshape(leading)
        tangent_residual = tangent_residual.reshape(leading)
        rank_margin = rank_margin.reshape(leading)
        geometry_valid = geometry_valid.reshape(leading)
        density = coordinate.log_prob - log_volume
        valid = (
            coordinate.valid
            & geometry_valid
            & (rank_margin > 0.0)
            & jnp.isfinite(density)
        )
        status = jnp.where(valid, 0, jnp.where(~geometry_valid, 2, 1)).astype(jnp.int32)
        return RiemannianFlowDensityResult(
            coordinate_result=coordinate,
            log_volume_density=log_volume,
            log_prob=jnp.where(valid, density, -jnp.inf),
            tangent_residual=tangent_residual,
            rank_margin=rank_margin,
            valid=valid,
            status=status,
            geometry_id=self.manifold.manifold_id,
            reference_measure="riemannian",
            law_id=self.law_id,
        )

    def log_prob(self, value: ArrayLike, /) -> Array:
        return self.log_prob_with_diagnostics(value).log_prob

    def contains(self, value: ArrayLike, /) -> Array:
        return self.log_prob_with_diagnostics(value).valid


__all__ = ["RiemannianContinuousFlowLaw", "RiemannianFlowDensityResult"]
