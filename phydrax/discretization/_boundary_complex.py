#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import AbstractLinearOperator, ArraySpace, HilbertComplex
from ..sparse import EdgeRelation, SparseCoordinateOperator
from ._topology import CellComplexTopology, EntitySet, OrientedIncidence


@final
class BoundaryComplex(StrictModule, NonTrainableState):
    """Closed facet selection with outward-first induced top-cell orientation.

    Lower-dimensional cells retain their parent orientation. The top-dimensional
    boundary cells carry their signed occurrence in the volume boundary chain.
    Inclusion and restriction use the same signs, so restriction commutes with d.
    At degree zero a point value is orientation independent; endpoint orientation
    is retained separately for integration of an oriented zero-dimensional chain.
    """

    topology: CellComplexTopology
    parent_indices: tuple[Array, ...]
    orientation_signs: tuple[Array, ...]
    parent_topology_id: str = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)

    def restriction_operators(
        self, source: HilbertComplex, target: HilbertComplex, /
    ) -> tuple[AbstractLinearOperator, ...]:
        if source.top_degree != self.topology.dimension + 1:
            raise ValueError("Source degree must equal the boundary degree plus one.")
        if target.top_degree != self.topology.dimension:
            raise ValueError("Target degree must equal the boundary degree.")
        operators: list[AbstractLinearOperator] = []
        for degree, indices in enumerate(self.parent_indices):
            parent_space = source.space(degree)
            boundary_space = target.space(degree)
            if not isinstance(parent_space, ArraySpace) or not isinstance(
                boundary_space, ArraySpace
            ):
                raise TypeError(
                    "Cell restriction requires array-valued coordinate spaces."
                )
            parent_size = parent_space.size
            count = self.topology.entities(degree).count
            if boundary_space.shape != (count,):
                raise ValueError(
                    "Boundary target does not use the selected cell coordinates."
                )
            signs = (
                self.orientation_signs[degree].astype(parent_space.dtype)
                if degree
                else jnp.ones((count,), dtype=parent_space.dtype)
            )
            operators.append(
                SparseCoordinateOperator(
                    EdgeRelation(
                        np.asarray(indices, dtype=np.int32),
                        np.arange(count, dtype=np.int32),
                        source_size=parent_size,
                        target_size=count,
                    ),
                    signs,
                    source=parent_space,
                    target=boundary_space,
                    operator_id=f"{self.boundary_id}:restriction:{degree}",
                    accumulation_dtype=parent_space.dtype,
                )
            )
        return tuple(operators)


def boundary_subcomplex(
    topology: CellComplexTopology, /, *, boundary_mask: ArrayLike
) -> BoundaryComplex:
    """Prepare the incidence closure of selected exterior facets.

    ``boundary_mask`` indexes degree n−1 cells. Each selected facet must have
    exactly one active incident volume cell. Interior/nonmanifold facets are
    refused rather than assigned an arbitrary outward normal. Geometry is not
    needed: the oriented volume incidence determines the outward-first sign.
    """
    if not isinstance(topology, CellComplexTopology):
        raise TypeError("topology must be CellComplexTopology.")
    dimension = topology.dimension
    if dimension == 0:
        raise ValueError("A zero-dimensional complex has no boundary complex.")
    facet_count = topology.entities(dimension - 1).count
    mask = np.asarray(boundary_mask)
    if mask.dtype != np.bool_ or mask.shape != (facet_count,):
        raise ValueError("boundary_mask must be a Boolean vector over volume facets.")
    top = topology.incidences[-1]
    valid = np.asarray(top.relation.valid, dtype=np.bool_)
    lower = np.asarray(top.relation.source_indices, dtype=np.int32)[valid]
    upper = np.asarray(top.relation.target_indices, dtype=np.int32)[valid]
    active = np.asarray(topology.entities(dimension).active_mask, dtype=np.bool_)[upper]
    lower = lower[active]
    coefficients = np.asarray(top.signs, dtype=np.float64)[valid][active]
    counts = np.bincount(lower, minlength=facet_count)
    if np.any(mask & (counts != 1)):
        raise ValueError(
            "Selected facets must have exactly one active incident volume cell."
        )
    if np.any(mask & ~np.asarray(topology.entities(dimension - 1).active_mask)):
        raise ValueError("Selected boundary facets must be active.")
    selected: list[np.ndarray] = [
        np.empty((0,), dtype=np.int32) for _ in range(dimension)
    ]
    selected[-1] = np.flatnonzero(mask).astype(np.int32)
    outward = np.zeros((facet_count,), dtype=np.float64)
    np.add.at(outward, lower, coefficients)
    signs = [np.ones((0,), dtype=np.float64) for _ in range(dimension)]
    signs[-1] = outward[selected[-1]]
    for degree in range(dimension - 1, 0, -1):
        incidence = topology.incidences[degree - 1]
        valid = np.asarray(incidence.relation.valid, dtype=np.bool_)
        src = np.asarray(incidence.relation.source_indices, dtype=np.int32)[valid]
        dst = np.asarray(incidence.relation.target_indices, dtype=np.int32)[valid]
        selected[degree - 1] = np.unique(src[np.isin(dst, selected[degree])])
        signs[degree - 1] = np.ones((selected[degree - 1].size,), dtype=np.float64)
    entities = tuple(
        EntitySet(
            f"boundary-{topology.entities(degree).name}",
            degree,
            np.asarray(topology.entities(degree).entity_ids)[indices],
        )
        for degree, indices in enumerate(selected)
    )
    incidences: list[OrientedIncidence] = []
    for degree in range(1, dimension):
        original = topology.incidences[degree - 1]
        valid = np.asarray(original.relation.valid, dtype=np.bool_)
        src = np.asarray(original.relation.source_indices, dtype=np.int32)[valid]
        dst = np.asarray(original.relation.target_indices, dtype=np.int32)[valid]
        retained = np.isin(dst, selected[degree])
        source_ids = np.searchsorted(selected[degree - 1], src[retained])
        target_ids = np.searchsorted(selected[degree], dst[retained])
        values = np.asarray(original.signs)[valid][retained]
        values = values * signs[degree][target_ids] * signs[degree - 1][source_ids]
        incidences.append(
            OrientedIncidence(
                degree,
                entities[degree - 1],
                entities[degree],
                EdgeRelation(
                    source_ids,
                    target_ids,
                    source_size=entities[degree - 1].count,
                    target_size=entities[degree].count,
                ),
                values,
            )
        )
    boundary_topology = CellComplexTopology(entities, tuple(incidences))
    boundary_id = canonical_fingerprint(
        {
            "kind": "induced-boundary-complex",
            "parent": topology.topology_id,
            "topology": boundary_topology.topology_id,
            "indices": [indices.tolist() for indices in selected],
            "orientations": [values.tolist() for values in signs],
        }
    )
    return BoundaryComplex(
        boundary_topology,
        tuple(jnp.asarray(indices) for indices in selected),
        tuple(jnp.asarray(values) for values in signs),
        topology.topology_id,
        boundary_id,
    )


__all__ = ["BoundaryComplex", "boundary_subcomplex"]
