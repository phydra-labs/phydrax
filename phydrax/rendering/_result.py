#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Evidence-bearing scientific rendering results."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..measurement import PreparedQuantityField


class RenderEvidence(StrictModule, NonTrainableState):
    finite: Array
    coverage_complete: Array
    capacity_sufficient: Array
    visibility_exact: Array
    route_stable: Array
    status: Array
    approximation: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        finite: ArrayLike,
        coverage_complete: ArrayLike,
        capacity_sufficient: ArrayLike,
        visibility_exact: ArrayLike,
        route_stable: ArrayLike,
        status: ArrayLike,
        /,
        *,
        approximation: str,
        plan_id: str,
        support_id: str,
        geometry_id: str,
    ):
        self.finite = jnp.asarray(finite, dtype=bool)
        self.coverage_complete = jnp.asarray(coverage_complete, dtype=bool)
        self.capacity_sufficient = jnp.asarray(capacity_sufficient, dtype=bool)
        self.visibility_exact = jnp.asarray(visibility_exact, dtype=bool)
        self.route_stable = jnp.asarray(route_stable, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.approximation = str(approximation)
        self.plan_id = str(plan_id)
        self.support_id = str(support_id)
        self.geometry_id = str(geometry_id)

    @property
    def successful(self) -> Array:
        return (
            self.finite
            & self.coverage_complete
            & self.capacity_sufficient
            & self.visibility_exact
            & (self.status == 0)
        )


class ImageRenderResult(StrictModule, NonTrainableState):
    prediction: PreparedQuantityField
    depth: Array
    hit: Array
    primitive_ids: Array
    entity_ids: Array
    local_coordinates: Array
    world_points: Array
    normals: Array
    front_facing: Array
    uniqueness_margin: Array
    evidence: RenderEvidence


__all__ = ["ImageRenderResult", "RenderEvidence"]
