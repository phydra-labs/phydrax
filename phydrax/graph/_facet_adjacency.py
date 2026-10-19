#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier, nonnegative_integer
from ..discretization.fem._generic import FiniteElementDiscretization
from ..discretization.finite_volume._unstructured import (
    UnstructuredFiniteVolumeDiscretization,
)
from ..sparse._relation import EdgeRelation
from ..typing import checked


def _integer_routes(name: str, value: ArrayLike, /) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iu":
        raise TypeError(
            f"FacetAdjacency.{name} must be an integer array; got dtype {array.dtype}"
        )
    if array.ndim != 1:
        raise ValueError(f"FacetAdjacency.{name} must be rank-1; got shape {array.shape}")
    return array.astype(np.int32)


class FacetAdjacency(StrictModule, NonTrainableState):
    """Cell adjacency through mesh facets with one owner-to-neighbor route per facet.

    `relation` routes follow `facet_ids`: each route starts at the facet owner cell
    and ends at its neighbor cell, so facet payloads such as fluxes or area
    vectors are route payloads without reindexing. Boundary facets (neighbor
    sentinel `-1`) and inactive facets keep their route slot with safe endpoint
    indices `0` and `relation.valid == False`; every native gather or reduction
    over the relation treats them as inert.

    `topology_id` identifies the scientific adjacency: the valid
    `(facet, owner, neighbor)` triples on the named cell and facet entity sets. It
    is independent of route order and boundary padding, so finite-volume and
    finite-element discretizations of one mesh share it. Use
    `GraphIR.from_edge_relation(adjacency.relation, ...)` for the learned model so
    the network and the physical residual reduce over one relation and one
    validity mask.
    """

    relation: EdgeRelation
    facet_ids: Array
    cell_entity_set_id: str = eqx.field(static=True)
    facet_entity_set_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        facet_ids: ArrayLike,
        owner_cells: ArrayLike,
        neighbor_cells: ArrayLike,
        /,
        *,
        cell_count: int,
        cell_entity_set_id: str,
        facet_entity_set_id: str,
        active_mask: ArrayLike | None = None,
    ) -> None:
        facets = _integer_routes("facet_ids", facet_ids)
        owners = _integer_routes("owner_cells", owner_cells)
        neighbors = _integer_routes("neighbor_cells", neighbor_cells)
        route_shape = facets.shape
        if owners.shape != route_shape or neighbors.shape != route_shape:
            raise ValueError(
                "FacetAdjacency.routes must align facet IDs, owner cells and neighbor cells; got mismatched shapes"
            )
        active = (
            np.ones(route_shape, dtype=np.bool_)
            if active_mask is None
            else np.asarray(active_mask)
        )
        if active.dtype.kind != "b" or active.shape != route_shape:
            raise ValueError(
                "FacetAdjacency.active_mask must be a boolean array with one entry per facet; got invalid mask dtype or shape"
            )
        count = nonnegative_integer(cell_count, "FacetAdjacency.cell_count")
        if np.any(facets < 0) or np.unique(facets).size != facets.size:
            raise ValueError(
                "FacetAdjacency.facet_ids must be unique nonnegative entity indices; got invalid facet IDs"
            )
        if np.any((owners < 0) | (owners >= count)):
            raise ValueError(
                f"FacetAdjacency.owner_cells must lie in [0, {count}); got out-of-range owners"
            )
        if np.any((neighbors < -1) | (neighbors >= count)):
            raise ValueError(
                f"FacetAdjacency.neighbor_cells must lie in [0, {count}) or use boundary sentinel -1; got out-of-range neighbors"
            )
        valid = active & (neighbors >= 0)
        if np.any(valid & (neighbors == owners)):
            raise ValueError(
                "FacetAdjacency.neighbor_cells must differ from owner cells on interior facets; got a self route"
            )
        cell_set = canonical_identifier(cell_entity_set_id, "cell_entity_set_id")
        facet_set = canonical_identifier(facet_entity_set_id, "facet_entity_set_id")
        order = np.argsort(facets[valid], kind="stable")
        routes = np.stack(
            (facets[valid][order], owners[valid][order], neighbors[valid][order]),
            axis=1,
        )
        self.relation = EdgeRelation(
            np.where(valid, owners, 0),
            np.where(valid, neighbors, 0),
            source_size=count,
            target_size=count,
            valid=valid,
        )
        self.facet_ids = jnp.asarray(facets)
        self.cell_entity_set_id = cell_set
        self.facet_entity_set_id = facet_set
        self.topology_id = canonical_fingerprint(
            {
                "kind": "facet-adjacency",
                "cell_entity_set": cell_set,
                "facet_entity_set": facet_set,
                "cell_count": count,
                "routes": array_tree_fingerprint(routes),
            }
        )


@checked
def facet_adjacency(
    discretization: UnstructuredFiniteVolumeDiscretization | FiniteElementDiscretization,
    /,
) -> FacetAdjacency:
    """Return the owner-to-neighbor facet adjacency of a prepared mesh discretization.

    Unstructured finite volumes route every face (boundary faces inert, inactive
    faces inert); finite elements route their interior-facet integration domain.
    """
    match discretization:
        case UnstructuredFiniteVolumeDiscretization():
            dimension = discretization.cell_dimension
            entity_sets = discretization.topology.entity_sets
            faces = discretization.face_block
            return FacetAdjacency(
                faces.face_ids,
                faces.owner_cells,
                faces.neighbor_cells,
                cell_count=discretization.cell_count,
                cell_entity_set_id=entity_sets[dimension].entity_set_id,
                facet_entity_set_id=entity_sets[dimension - 1].entity_set_id,
                active_mask=faces.active_mask,
            )
        case FiniteElementDiscretization():
            facets = discretization.interior_facet_domain
            cells = discretization.cell_domain
            return FacetAdjacency(
                facets.entity_indices,
                facets.owner_cells,
                facets.neighbor_cells,
                cell_count=cells.entity_indices.shape[0],
                cell_entity_set_id=cells.entity_set_id,
                facet_entity_set_id=facets.entity_set_id,
            )
        case _:
            raise TypeError(
                "facet_adjacency.discretization must be an UnstructuredFiniteVolumeDiscretization "
                "or FiniteElementDiscretization; got unsupported discretization"
            )


__all__ = ["FacetAdjacency", "facet_adjacency"]
