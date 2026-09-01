#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import AbstractLinearOperator
from ._guarantee import ContactGuaranteeLevel
from ._precision import ContactPrecisionPolicy
from ._surface import (
    CollisionSurfacePlan,
    PreparedCollisionSurface,
)


class ContactProxyEvidence(StrictModule):
    approximation_error: Array
    maximum_error: Array
    inflated_minimum_separation: Array
    guarantee_level: Array
    finite: Array
    certified: Array
    successful: Array
    proxy_id: str = eqx.field(static=True)


class PreparedContactProxy(StrictModule, NonTrainableState):
    surface: PreparedCollisionSurface
    approximation_error: Array
    evidence: ContactProxyEvidence
    proxy_id: str = eqx.field(static=True)

    def positions(self, state, /) -> Array:
        return self.surface.positions(state)

    def pullback(self, force: ArrayLike, /):
        return self.surface.pullback(force)


class ContactProxyPlan(StrictModule, NonTrainableState):
    """Piecewise-linear collision proxy with a certified geometric error."""

    topology: CollisionSurfacePlan
    approximation_error: Array
    certified: bool = eqx.field(static=True)
    proxy_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: CollisionSurfacePlan,
        approximation_error: ArrayLike,
        /,
        *,
        certified: bool,
    ):
        if not isinstance(topology, CollisionSurfacePlan):
            raise TypeError("topology must be CollisionSurfacePlan.")
        error = np.asarray(approximation_error, dtype=float)
        if error.shape == ():
            error = np.full((topology.vertex_count,), float(error), dtype=float)
        if error.shape != (topology.vertex_count,):
            raise ValueError(
                "approximation_error must be scalar or one value per proxy vertex."
            )
        if np.any(~np.isfinite(error)) or np.any(error < 0.0):
            raise ValueError(
                "Contact proxy approximation error must be finite and nonnegative."
            )
        self.topology = topology
        self.approximation_error = jnp.asarray(error)
        self.certified = bool(certified)
        self.proxy_id = canonical_fingerprint(
            {
                "kind": "contact-proxy-plan",
                "topology": topology.topology_id,
                "error": array_tree_fingerprint(error),
                "certified": bool(certified),
            }
        )

    def prepare(
        self,
        rest_positions: ArrayLike,
        displacement_operator: AbstractLinearOperator,
        /,
        *,
        precision: ContactPrecisionPolicy | None = None,
    ) -> PreparedContactProxy:
        inflated = self.topology.vertex_minimum_separation + self.approximation_error
        inflated_topology = CollisionSurfacePlan(
            self.topology.vertex_ids,
            ambient_dimension=self.topology.ambient_dimension,
            edges=self.topology.edges,
            faces=self.topology.faces,
            orientable_mask=self.topology.orientable_mask,
            codimensional_mask=self.topology.codimensional_mask,
            pair_policy=self.topology.pair_policy,
            minimum_separation=inflated,
        )
        surface = PreparedCollisionSurface(
            inflated_topology,
            rest_positions,
            displacement_operator,
            precision=precision,
        )
        error = self.approximation_error.astype(surface.precision.geometry_dtype)
        maximum = jnp.max(error, initial=0.0)
        finite = jnp.all(jnp.isfinite(error))
        level = (
            ContactGuaranteeLevel.ANALYTIC_CONSERVATIVE
            if self.certified
            else ContactGuaranteeLevel.HEURISTIC
        )
        evidence = ContactProxyEvidence(
            error,
            maximum,
            inflated.astype(surface.precision.geometry_dtype),
            jnp.asarray(int(level), dtype=jnp.int32),
            finite,
            jnp.asarray(self.certified),
            finite & jnp.asarray(self.certified),
            self.proxy_id,
        )
        return PreparedContactProxy(
            surface,
            error,
            evidence,
            self.proxy_id,
        )


class ContactProxyTransfer(StrictModule, NonTrainableState):
    """Explicit old-to-new route ownership for accepted proxy refinement."""

    old_vertex_ids: Array
    new_vertex_ids: Array
    new_parent_vertices: Array
    parent_weights: Array
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        old_vertex_ids: ArrayLike,
        new_vertex_ids: ArrayLike,
        new_parent_vertices: ArrayLike,
        parent_weights: ArrayLike,
        /,
    ):
        old_ids = np.asarray(old_vertex_ids)
        new_ids = np.asarray(new_vertex_ids)
        parents = np.asarray(new_parent_vertices)
        weights = np.asarray(parent_weights, dtype=float)
        if old_ids.ndim != 1 or new_ids.ndim != 1:
            raise ValueError("Proxy vertex IDs must be vectors.")
        if parents.ndim != 2 or weights.shape != parents.shape:
            raise ValueError(
                "Proxy parent vertices and weights must be matching matrices."
            )
        if parents.shape[0] != new_ids.size:
            raise ValueError("Proxy transfer needs one parent row per new vertex.")
        if np.any(parents < 0) or np.any(parents >= old_ids.size):
            raise ValueError("Proxy transfer parent index is invalid.")
        if np.any(~np.isfinite(weights)) or not np.allclose(weights.sum(axis=1), 1.0):
            raise ValueError("Proxy transfer weights must be finite and affine.")
        self.old_vertex_ids = jnp.asarray(old_ids, dtype=jnp.int64)
        self.new_vertex_ids = jnp.asarray(new_ids, dtype=jnp.int64)
        self.new_parent_vertices = jnp.asarray(parents, dtype=jnp.int32)
        self.parent_weights = jnp.asarray(weights)
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "contact-proxy-transfer",
                "old": array_tree_fingerprint(old_ids),
                "new": array_tree_fingerprint(new_ids),
                "parents": array_tree_fingerprint(parents),
                "weights": array_tree_fingerprint(weights),
            }
        )

    def transfer_vertex_field(self, old_values: ArrayLike, /) -> Array:
        values = jnp.asarray(old_values)
        if values.shape[0] != self.old_vertex_ids.size:
            raise ValueError("Proxy transfer source field has invalid leading dimension.")
        gathered = values[self.new_parent_vertices]
        weights = self.parent_weights.astype(values.dtype)
        while weights.ndim < gathered.ndim:
            weights = weights[..., None]
        return jnp.sum(weights * gathered, axis=1)


__all__ = [
    "ContactProxyEvidence",
    "ContactProxyPlan",
    "ContactProxyTransfer",
    "PreparedContactProxy",
]
