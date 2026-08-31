#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class IntegrationDomain(StrictModule, NonTrainableState):
    """Rule-free cell or facet routes over one prepared discrete support."""

    kind: str = eqx.field(static=True)
    entity_indices: Array
    owner_cells: Array
    neighbour_cells: Array
    owner_local_entities: Array
    neighbour_local_entities: Array
    neighbour_trace_permutations: Array
    periodic_face_mask: Array
    support_id: str = eqx.field(static=True)
    entity_set_id: str = eqx.field(static=True)
    selection_id: str | None = eqx.field(static=True)
    trace_map_id: str | None = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: str,
        entity_indices: ArrayLike,
        support_id: str,
        entity_set_id: str,
        /,
        *,
        owner_cells: ArrayLike | None = None,
        neighbour_cells: ArrayLike | None = None,
        owner_local_entities: ArrayLike | None = None,
        neighbour_local_entities: ArrayLike | None = None,
        neighbour_trace_permutations: ArrayLike | None = None,
        periodic_face_mask: ArrayLike | None = None,
        selection_id: str | None = None,
    ):
        kind_ = str(kind)
        if kind_ not in ("cell", "exterior_facet", "interior_facet"):
            raise ValueError("Unsupported integration-domain kind.")
        indices = np.asarray(entity_indices, dtype=np.int32)
        if (
            indices.ndim != 1
            or np.any(indices < 0)
            or np.unique(indices).size != indices.size
        ):
            raise ValueError(
                "Integration-domain indices must be unique non-negative IDs."
            )
        count = indices.size

        def route(name: str, value: ArrayLike | None, default: int) -> np.ndarray:
            result = (
                np.full((count,), default, dtype=np.int32)
                if value is None
                else np.asarray(value, dtype=np.int32)
            )
            if result.shape != (count,):
                raise ValueError(f"{name} must have shape {(count,)}.")
            return result

        owner = route("owner_cells", owner_cells, -1)
        neighbour = route("neighbour_cells", neighbour_cells, -1)
        owner_local = route("owner_local_entities", owner_local_entities, -1)
        neighbour_local = route("neighbour_local_entities", neighbour_local_entities, -1)
        if kind_ == "cell":
            if np.any(owner < 0):
                owner = indices.copy()
        elif np.any(owner < 0):
            raise ValueError("Facet integration domains require owner cells.")
        if kind_ == "interior_facet" and np.any(neighbour < 0):
            raise ValueError("Interior facets require neighbour cells.")
        if kind_ == "exterior_facet" and np.any(neighbour >= 0):
            raise ValueError("Exterior facets cannot carry neighbour cells.")
        support = str(support_id)
        entity_set = str(entity_set_id)
        if not support or not entity_set:
            raise ValueError("Integration-domain support/entity IDs must be non-empty.")
        selection = None if selection_id is None else str(selection_id)
        if selection is not None and not selection:
            raise ValueError("selection_id must be non-empty or None.")
        trace_permutations = (
            np.empty((count, 0), dtype=np.int32)
            if neighbour_trace_permutations is None
            else np.asarray(neighbour_trace_permutations, dtype=np.int32)
        )
        if trace_permutations.ndim != 2 or trace_permutations.shape[0] != count:
            raise ValueError(
                "Neighbour trace permutations require one row per domain entity."
            )
        if trace_permutations.shape[1]:
            expected = np.arange(trace_permutations.shape[1], dtype=np.int32)
            if any(
                not np.array_equal(np.sort(permutation), expected)
                for permutation in trace_permutations
            ):
                raise ValueError("Each neighbour trace map must be a permutation.")
        periodic = (
            np.zeros((count,), dtype=bool)
            if periodic_face_mask is None
            else np.asarray(periodic_face_mask, dtype=bool)
        )
        if periodic.shape != (count,):
            raise ValueError("periodic_face_mask must match the domain entities.")
        if np.any(periodic) and kind_ != "interior_facet":
            raise ValueError("Periodic face routes require an interior-facet domain.")
        trace_map = (
            None
            if trace_permutations.shape[1] == 0
            else canonical_fingerprint(
                {
                    "kind": "neighbour-trace-map",
                    "permutations": array_tree_fingerprint(trace_permutations),
                    "periodic": array_tree_fingerprint(periodic),
                }
            )
        )
        self.kind = kind_
        self.entity_indices = jnp.asarray(indices)
        self.owner_cells = jnp.asarray(owner)
        self.neighbour_cells = jnp.asarray(neighbour)
        self.owner_local_entities = jnp.asarray(owner_local)
        self.neighbour_trace_permutations = jnp.asarray(trace_permutations)
        self.periodic_face_mask = jnp.asarray(periodic)
        self.neighbour_local_entities = jnp.asarray(neighbour_local)
        self.support_id = support
        self.entity_set_id = entity_set
        self.selection_id = selection
        self.trace_map_id = trace_map
        self.domain_id = canonical_fingerprint(
            {
                "kind": "integration-domain",
                "entity_kind": kind_,
                "indices": array_tree_fingerprint(indices),
                "owner": array_tree_fingerprint(owner),
                "neighbour": array_tree_fingerprint(neighbour),
                "owner_local": array_tree_fingerprint(owner_local),
                "neighbour_local": array_tree_fingerprint(neighbour_local),
                "neighbour_trace_permutations": array_tree_fingerprint(
                    trace_permutations
                ),
                "periodic_face_mask": array_tree_fingerprint(periodic),
                "trace_map": trace_map,
                "support": support,
                "entity_set": entity_set,
                "selection": selection,
            }
        )


__all__ = ["IntegrationDomain"]
