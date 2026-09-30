#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conservative, finite-element, cochain, and reflux transfer on forest AMR.

Every route is prepared once on the host from two canonical forest topologies of
one plan and executed through :mod:`phydrax.sparse`.  Leaf transfers live on
fixed-capacity padded leaf storage so compiled consumers are reused across
adaptation cycles that stay inside one capacity bucket.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from typing import Any, final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    adjoint,
    ArraySpace,
    ComplexMap,
    ComplexMapEvidence,
    dual_transpose,
    HilbertComplex,
)
from ...sparse import (
    EdgeRelation,
    linear_apply,
    linear_transpose_apply,
    route_reduce,
    SparseCoordinateOperator,
)
from .._cochain_hodge import CochainHodge, DiagonalHodge
from .._transfer import TransferProperties
from ..fem._topology_transfer import (
    FiniteElementTopologyTransfer,
    vertex_interpolation_transfer,
)
from ._entities import _active_operator
from ._entity_transfer import (
    _pad_operator,
    _prolongation_routes,
    _scipy_matrix,
    CompatibleEntityTransfer,
)
from ._forest import (
    _locate,
    _node_keys,
    _point_keys,
    forest_capacity_bucket,
    forest_leaf_geometry,
    ForestHierarchyTopology,
    ForestLeafGeometry,
)
from ._reflux import FluxRegister


class ForestTransferResult(StrictModule):
    """Transferred padded leaf values with componentwise conservation evidence."""

    values: Array
    source_content: Array
    target_content: Array
    conservation_residual: Array
    successful: Array


class ForestTransferRoutes(StrictModule, NonTrainableState):
    """Bucket-stable padded leaf routes of one conservative forest transfer.

    Static structure is limited to capacities, so compiled consumers of
    ``apply``/``pullback``/``hilbert_adjoint`` are reused by every transition whose
    source, target, and route capacities fall in the same buckets.
    """

    relation: EdgeRelation
    coefficients: Array
    source_measures: Array
    target_measures: Array

    def _content(self, measures: Array, values: Array, /) -> Array:
        weights = measures.reshape(measures.shape + (1,) * (values.ndim - 1))
        return jnp.sum(weights * values, axis=0)

    def apply(self, values: ArrayLike, /) -> ForestTransferResult:
        """Transfer padded source leaf values ``(Cs, *T)`` to ``(Ct, *T)``.

        Padded source slots are ignored; success requires finite values and a
        content residual within ``100 eps max(|source content|, 1)``.
        """
        source = jnp.asarray(values)
        if source.ndim == 0 or source.shape[0] != self.relation.source_size:
            raise ValueError("Forest transfer values must match the source capacity.")
        active = (self.source_measures > 0.0).reshape(
            (self.relation.source_size,) + (1,) * (source.ndim - 1)
        )
        source = jnp.where(active, source, jnp.zeros((), dtype=source.dtype))
        target = linear_apply(self.relation, self.coefficients, source)
        source_content = self._content(self.source_measures, source)
        target_content = self._content(self.target_measures, target)
        residual = target_content - source_content
        precision = jnp.finfo(jnp.real(target).dtype).eps
        tolerance = 100.0 * precision * jnp.maximum(jnp.abs(source_content), 1.0)
        successful = jnp.all(jnp.isfinite(target)) & jnp.all(
            jnp.abs(residual) <= tolerance
        )
        return ForestTransferResult(
            values=target,
            source_content=source_content,
            target_content=target_content,
            conservation_residual=residual,
            successful=successful,
        )

    def pullback(self, dual: ArrayLike, /) -> Array:
        """Algebraic transpose action mapping target duals ``(Ct, *T)`` to source."""
        values = jnp.asarray(dual)
        if values.ndim == 0 or values.shape[0] != self.relation.target_size:
            raise ValueError("Forest transfer duals must match the target capacity.")
        return linear_transpose_apply(self.relation, self.coefficients, values)

    def hilbert_adjoint(self, target_values: ArrayLike, /) -> Array:
        """Adjoint under the physical leaf-volume pairings of source and target."""
        values = jnp.asarray(target_values)
        if values.ndim == 0 or values.shape[0] != self.relation.target_size:
            raise ValueError("Forest adjoint values must match the target capacity.")
        target_weights = self.target_measures.reshape(
            self.target_measures.shape + (1,) * (values.ndim - 1)
        )
        pulled = self.pullback(target_weights * values)
        source_weights = self.source_measures.reshape(
            self.source_measures.shape + (1,) * (values.ndim - 1)
        )
        safe = jnp.where(source_weights > 0.0, source_weights, 1.0)
        return jnp.where(source_weights > 0.0, pulled / safe, 0.0)


def _leaf_volumes(topology: ForestHierarchyTopology, geometry: Any, /) -> np.ndarray:
    geometry_ = forest_leaf_geometry(topology) if geometry is None else geometry
    if not isinstance(geometry_, ForestLeafGeometry):
        raise TypeError("Forest leaf geometry must be a ForestLeafGeometry.")
    if not bool(geometry_.valid):
        raise ValueError("Forest leaf geometry failed its metric evidence.")
    volumes = np.asarray(geometry_.volumes, dtype=np.float64)
    if volumes.shape != (topology.signature.leaf_capacity,):
        raise ValueError("Forest leaf geometry does not match the topology capacity.")
    active = volumes[: topology.leaf_count]
    if np.any(~np.isfinite(active)) or np.any(active <= 0.0):
        raise ValueError("Forest leaf volumes must be finite and positive.")
    return active


class ForestFieldTransition(StrictModule, NonTrainableState):
    """Conservative parent/child cell-average transfer between forest epochs.

    A target leaf inside a source leaf receives that leaf's average scaled so the
    source content splits exactly over its descendants; a target leaf covering
    several source leaves receives their volume-weighted average.  On affine roots
    both rules are also exactly constant preserving.
    """

    source: ForestHierarchyTopology
    target: ForestHierarchyTopology
    routes: ForestTransferRoutes
    properties: TransferProperties
    retained_leaves: int = eqx.field(static=True)
    prolonged_leaves: int = eqx.field(static=True)
    restricted_leaves: int = eqx.field(static=True)
    transition_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: ForestHierarchyTopology,
        target: ForestHierarchyTopology,
        /,
        *,
        source_geometry: ForestLeafGeometry | None = None,
        target_geometry: ForestLeafGeometry | None = None,
    ) -> None:
        if not isinstance(source, ForestHierarchyTopology) or not isinstance(
            target, ForestHierarchyTopology
        ):
            raise TypeError("Forest field transitions require forest topologies.")
        if source.plan.plan_id != target.plan.plan_id:
            raise ValueError("Forest field transitions cannot change the forest plan.")
        if target.epoch.index != source.epoch.index + 1:
            raise ValueError("Forest field transitions require consecutive epochs.")
        plan = source.plan
        source_volumes = _leaf_volumes(source, source_geometry)
        target_volumes = _leaf_volumes(target, target_geometry)
        source_levels = source.leaf_levels()
        source_keys = source.leaf_keys()
        target_levels = target.leaf_levels()
        target_keys = target.leaf_keys()
        target_count = target.leaf_count
        containing = _locate(source_keys, target_keys)
        inside = source_levels[containing] <= target_levels
        # Descendant targets split their ancestor's content by volume share.
        descendant_volume = np.bincount(
            containing[inside],
            weights=target_volumes[inside],
            minlength=source.leaf_count,
        )
        inside_targets = np.flatnonzero(inside)
        inside_sources = containing[inside]
        inside_weights = (
            source_volumes[inside_sources] / descendant_volume[inside_sources]
        )
        coarse_targets = np.flatnonzero(~inside)
        spans = np.int64(1) << (
            plan.dimension * (plan.maximum_level - target_levels[coarse_targets])
        )
        starts = np.searchsorted(source_keys, target_keys[coarse_targets], side="left")
        stops = np.searchsorted(
            source_keys, target_keys[coarse_targets] + spans, side="left"
        )
        widths = stops - starts
        merged_targets = np.repeat(coarse_targets, widths)
        merged_sources = (
            np.arange(widths.sum(), dtype=np.int64)
            - np.repeat(np.cumsum(widths) - widths, widths)
            + np.repeat(starts, widths)
        )
        merged_weights = source_volumes[merged_sources] / target_volumes[merged_targets]
        route_sources = np.concatenate((inside_sources, merged_sources))
        route_targets = np.concatenate((inside_targets, merged_targets))
        route_weights = np.concatenate((inside_weights, merged_weights))
        order = np.lexsort((route_sources, route_targets))
        route_sources = route_sources[order]
        route_targets = route_targets[order]
        route_weights = route_weights[order]
        route_count = route_sources.shape[0]
        capacity = forest_capacity_bucket(route_count, target.signature.leaf_capacity)
        valid = np.arange(capacity) < route_count
        padded_sources = np.zeros((capacity,), dtype=np.int32)
        padded_targets = np.zeros((capacity,), dtype=np.int32)
        padded_weights = np.zeros((capacity,), dtype=np.float64)
        padded_sources[:route_count] = route_sources
        padded_targets[:route_count] = route_targets
        padded_weights[:route_count] = route_weights
        source_measures = np.zeros((source.signature.leaf_capacity,), dtype=np.float64)
        source_measures[: source.leaf_count] = source_volumes
        target_measures = np.zeros((target.signature.leaf_capacity,), dtype=np.float64)
        target_measures[:target_count] = target_volumes
        routes = ForestTransferRoutes(
            relation=EdgeRelation(
                padded_sources,
                padded_targets,
                source_size=source.signature.leaf_capacity,
                target_size=target.signature.leaf_capacity,
                valid=valid,
            ),
            coefficients=jnp.asarray(padded_weights),
            source_measures=jnp.asarray(source_measures),
            target_measures=jnp.asarray(target_measures),
        )
        retained = int(
            np.count_nonzero(inside & (source_levels[containing] == target_levels))
        )
        self.source = source
        self.target = target
        self.routes = routes
        self.properties = TransferProperties(
            constant_preserving=plan.maps is None,
            conservative=True,
            positivity_preserving=True,
            nested=True,
            adjoint_paired=True,
            differentiable_geometry=False,
            exact_on=("nested-cell-average",),
        )
        self.retained_leaves = retained
        self.prolonged_leaves = int(inside_targets.size) - retained
        self.restricted_leaves = int(coarse_targets.size)
        self.transition_id = canonical_fingerprint(
            {
                "kind": "forest-field-transition",
                "source": source.epoch.epoch_id,
                "target": target.epoch.epoch_id,
                "routes": array_tree_fingerprint(
                    (padded_sources, padded_targets, padded_weights, valid)
                ),
                "measures": array_tree_fingerprint((source_measures, target_measures)),
            }
        )


class ForestRefluxRoutes(StrictModule, NonTrainableState):
    """Coarse/fine interface flux-register routes of one forest workset.

    Every hanging interior face routes to its coarse-side leaf.  ``signs`` orient
    fluxes along ``+face_axes`` into the coarse leaf's balance: ``+1`` when the
    coarse leaf is the plus side and ``-1`` when it is the minus side.
    """

    relation: EdgeRelation
    signs: Array

    def __init__(self, topology: ForestHierarchyTopology, /) -> None:
        if not isinstance(topology, ForestHierarchyTopology):
            raise TypeError("Forest reflux routes require a forest topology.")
        workset = topology.workset
        valid = workset.face_valid & workset.face_coarse_fine
        minus = jnp.where(workset.face_valid, workset.face_minus, 0)
        plus = jnp.where(workset.face_valid, workset.face_plus, 0)
        coarse_is_minus = workset.leaf_levels[minus] < workset.leaf_levels[plus]
        coarse = jnp.where(valid, jnp.where(coarse_is_minus, minus, plus), 0)
        self.relation = EdgeRelation(
            coarse,
            jnp.arange(topology.signature.face_capacity, dtype=jnp.int32),
            source_size=topology.signature.leaf_capacity,
            target_size=topology.signature.face_capacity,
            valid=valid,
        )
        self.signs = jnp.where(valid, jnp.where(coarse_is_minus, -1.0, 1.0), 0.0)

    def register(
        self,
        coarse_flux: ArrayLike,
        fine_flux: ArrayLike,
        /,
        *,
        accumulated_time: ArrayLike = 1.0,
    ) -> FluxRegister:
        """Flux register of time-integrated face fluxes ``(F, *T)`` along ``+axis``.

        ``coarse_flux`` is what the coarse leaves applied and ``fine_flux`` what the
        fine leaves applied through each hanging face over the same interval.
        """
        coarse = jnp.asarray(coarse_flux)
        fine = jnp.asarray(fine_flux)
        if coarse.ndim == 0 or coarse.shape[0] != self.relation.target_size:
            raise ValueError("Forest reflux fluxes must match the face capacity.")
        signs = self.signs.reshape(self.signs.shape + (1,) * (coarse.ndim - 1))
        return FluxRegister(
            signs * coarse,
            signs * fine,
            self.relation.valid,
            accumulated_time=accumulated_time,
            orientation=1,
            refinement_ratio=2,
        )

    def apply(
        self,
        register: FluxRegister,
        state: ArrayLike,
        leaf_volumes: ArrayLike,
        /,
    ) -> Array:
        """Add the register's coarse-leaf corrections to padded leaf state ``(C, *T)``."""
        if not isinstance(register, FluxRegister):
            raise TypeError("Forest reflux requires a FluxRegister.")
        values = jnp.asarray(state)
        volumes = jnp.asarray(leaf_volumes)
        if values.ndim == 0 or values.shape[0] != self.relation.source_size:
            raise ValueError("Forest reflux state must match the leaf capacity.")
        face_volumes = jnp.where(
            self.relation.valid, volumes[self.relation.source_indices], 1.0
        )
        correction = register.correction(face_volumes)
        return values + route_reduce(self.relation.transpose(), correction)


def _touching_leaf_rows(
    topology: ForestHierarchyTopology,
    leaf_vertices: np.ndarray,
    points: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Corner vertex ids and multilinear weights of the coarsest leaf at each point.

    ``points`` are finest-lattice vertex coordinates.  Among the leaves whose
    closure contains a point, the coarsest (lowest slot on ties) evaluates it, so a
    continuous vertex field is sampled consistently from either side.
    """
    plan = topology.plan
    dimension = plan.dimension
    depth = plan.maximum_level
    extent = np.asarray(plan.root_shape, dtype=np.int64) << depth
    periodic = np.asarray(plan.periodic_axes, dtype=np.bool_)
    levels = topology.leaf_levels()
    coordinates = topology.leaf_coordinates()
    keys = topology.leaf_keys()
    offsets = np.indices((2,) * dimension).reshape(dimension, -1).T.astype(np.int64)
    count = points.shape[0]
    cells = points[:, None, :] - offsets[None, :, :]
    wrapped = np.where(periodic, cells % extent, cells)
    inside = np.all((wrapped >= 0) & (wrapped < extent), axis=-1)
    slots = np.zeros(inside.shape, dtype=np.int64)
    slots[inside] = _locate(keys, _point_keys(plan, wrapped[inside]))
    candidate_levels = np.where(inside, levels[slots], plan.maximum_level + 1)
    rank = candidate_levels * (topology.leaf_count + 1) + np.where(inside, slots, 0)
    choice = np.argmin(rank, axis=1)
    rows = np.arange(count)
    slot = slots[rows, choice]
    span = np.int64(1) << (depth - levels[slot])
    # Undo the periodic wrap of the chosen candidate cell to place the anchor
    # in the point's unwrapped frame.
    shift = cells[rows, choice] - wrapped[rows, choice]
    anchor = (coordinates[slot] << (depth - levels[slot])[:, None]) + shift
    fraction = (points - anchor) / span[:, None].astype(np.float64)
    weights = np.prod(
        np.where(
            offsets[None, :, :] == 1, fraction[:, None, :], 1.0 - fraction[:, None, :]
        ),
        axis=-1,
    )
    return leaf_vertices[slot], weights


class ForestVertexLayout(StrictModule, NonTrainableState):
    """Continuous Q1 vertex layout of one forest with hanging-vertex constraints.

    Vertices are the unique leaf corners on the finest lattice (periodic axes
    wrapped).  A vertex is hanging when it is not a corner of the coarsest leaf
    touching it; ``constraint_*`` express every vertex through regular vertices
    (identity rows for regular vertices), recursively composed across levels.
    """

    topology: ForestHierarchyTopology
    vertex_coordinates: Array
    leaf_vertices: Array
    hanging: Array
    constraint_rows: Array
    constraint_weights: Array
    constraint_valid: Array
    vertex_count: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(self, topology: ForestHierarchyTopology, /) -> None:
        if not isinstance(topology, ForestHierarchyTopology):
            raise TypeError("Forest vertex layouts require a forest topology.")
        plan = topology.plan
        dimension = plan.dimension
        depth = plan.maximum_level
        extent = np.asarray(plan.root_shape, dtype=np.int64) << depth
        periodic = np.asarray(plan.periodic_axes, dtype=np.bool_)
        levels = topology.leaf_levels()
        span = (np.int64(1) << (depth - levels))[:, None, None]
        offsets = np.indices((2,) * dimension).reshape(dimension, -1).T.astype(np.int64)
        corners = (topology.leaf_coordinates() << (depth - levels)[:, None])[
            :, None, :
        ] + offsets[None, :, :] * span
        corners = np.where(periodic, corners % extent, corners)
        vertices, inverse = np.unique(
            corners.reshape(-1, dimension), axis=0, return_inverse=True
        )
        leaf_vertices = inverse.reshape(topology.leaf_count, -1).astype(np.int64)
        vertex_count = vertices.shape[0]
        rows, weights = _touching_leaf_rows(topology, leaf_vertices, vertices)
        own = np.arange(vertex_count)[:, None]
        hanging = ~np.any((rows == own) & np.isclose(weights, 1.0), axis=1)
        local = sp.csr_matrix(
            (
                np.where(hanging[:, None], weights, (rows == own) * 1.0).reshape(-1),
                (np.repeat(np.arange(vertex_count), rows.shape[1]), rows.reshape(-1)),
            ),
            shape=(vertex_count, vertex_count),
        )
        local.eliminate_zeros()
        constraint = local
        # Hanging masters are resolved from strictly coarser leaves, so repeated
        # substitution terminates within the forest depth.
        for _ in range(depth + 1):
            if not np.any(hanging[constraint.indices]):
                break
            constraint = (constraint @ local).tocsr()
            constraint.eliminate_zeros()
        if np.any(hanging[constraint.indices]):
            raise RuntimeError("Forest hanging-vertex constraints failed to resolve.")
        width = max(int(np.max(np.diff(constraint.indptr))), 1)
        constraint_rows, constraint_weights, constraint_valid = _padded_csr(
            constraint, width
        )
        capacity = topology.signature.leaf_capacity
        padded_leaf_vertices = np.full(
            (capacity, leaf_vertices.shape[1]), -1, dtype=np.int32
        )
        padded_leaf_vertices[: topology.leaf_count] = leaf_vertices
        self.topology = topology
        self.vertex_coordinates = jnp.asarray(vertices, dtype=jnp.int64)
        self.leaf_vertices = jnp.asarray(padded_leaf_vertices)
        self.hanging = jnp.asarray(hanging)
        self.constraint_rows = jnp.asarray(constraint_rows)
        self.constraint_weights = jnp.asarray(constraint_weights)
        self.constraint_valid = jnp.asarray(constraint_valid)
        self.vertex_count = vertex_count
        self.layout_id = canonical_fingerprint(
            {
                "kind": "forest-vertex-layout",
                "topology": topology.topology_id,
                "vertices": array_tree_fingerprint(vertices),
                "constraints": array_tree_fingerprint(
                    (constraint_rows, constraint_weights, constraint_valid)
                ),
            }
        )

    def reference_coordinates(self, /) -> np.ndarray:
        """Host ``(V, d)`` reference coordinates of the vertices."""
        plan = self.topology.plan
        spacing = np.asarray(plan.root_spacing, dtype=np.float64) / float(
            1 << plan.maximum_level
        )
        return np.asarray(plan.lower_bounds, dtype=np.float64) + spacing * np.asarray(
            self.vertex_coordinates, dtype=np.float64
        )

    def constrain(self, values: ArrayLike, /) -> Array:
        """Overwrite hanging-vertex values ``(V, *T)`` by their constraint rows."""
        array = jnp.asarray(values)
        if array.ndim == 0 or array.shape[0] != self.vertex_count:
            raise ValueError("Vertex values must match the forest vertex count.")
        gathered = array[jnp.where(self.constraint_valid, self.constraint_rows, 0)]
        weights = jnp.where(self.constraint_valid, self.constraint_weights, 0.0)
        return jnp.sum(
            weights.reshape(weights.shape + (1,) * (array.ndim - 1)) * gathered, axis=1
        )


def _padded_csr(
    matrix: sp.csr_matrix, width: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.diff(matrix.indptr)
    rows = np.zeros((matrix.shape[0], width), dtype=np.int32)
    weights = np.zeros((matrix.shape[0], width), dtype=np.float64)
    valid = np.arange(width)[None, :] < counts[:, None]
    rows[valid] = matrix.indices
    weights[valid] = matrix.data
    return rows, weights, valid


def forest_vertex_interpolation(
    source: ForestVertexLayout,
    target: ForestVertexLayout,
    /,
) -> FiniteElementTopologyTransfer:
    """Continuous Q1 interpolation from source to target forest vertices.

    Each target vertex evaluates the source field in the coarsest source leaf
    touching it, composed with source hanging constraints so rows reference only
    regular source vertices.  ``pullback`` is the matching restriction of duals.
    On affine roots linear reproduction is certified on reference coordinates.
    """
    if not isinstance(source, ForestVertexLayout) or not isinstance(
        target, ForestVertexLayout
    ):
        raise TypeError("Forest vertex interpolation requires vertex layouts.")
    if source.topology.plan.plan_id != target.topology.plan.plan_id:
        raise ValueError("Forest vertex interpolation cannot change the forest plan.")
    source_topology = source.topology
    leaf_vertices = np.asarray(source.leaf_vertices, dtype=np.int64)[
        : source_topology.leaf_count
    ]
    rows, weights = _touching_leaf_rows(
        source_topology,
        leaf_vertices,
        np.asarray(target.vertex_coordinates, dtype=np.int64),
    )
    target_count = target.vertex_count
    evaluation = sp.csr_matrix(
        (
            weights.reshape(-1),
            (np.repeat(np.arange(target_count), rows.shape[1]), rows.reshape(-1)),
        ),
        shape=(target_count, source.vertex_count),
    )
    constraint_valid = np.asarray(source.constraint_valid, dtype=np.bool_)
    constraint = sp.csr_matrix(
        (
            np.asarray(source.constraint_weights)[constraint_valid],
            (
                np.nonzero(constraint_valid)[0],
                np.asarray(source.constraint_rows)[constraint_valid],
            ),
        ),
        shape=(source.vertex_count, source.vertex_count),
    )
    composed = (evaluation @ constraint).tocsr()
    composed.eliminate_zeros()
    composed.sort_indices()
    width = max(int(np.max(np.diff(composed.indptr))), 1)
    padded_rows, padded_weights, valid = _padded_csr(composed, width)
    # Reference-linear fields are reproduced only on affine, aperiodic roots; a
    # periodic seam wraps vertex coordinates and a chart bends reference lines.
    linear = source_topology.plan.maps is None and not any(
        source_topology.plan.periodic_axes
    )
    return vertex_interpolation_transfer(
        padded_rows,
        padded_weights,
        valid,
        source_size=source.vertex_count,
        source_topology_id=source_topology.topology_id,
        target_topology_id=target.topology.topology_id,
        preserves_linear=linear,
        source_coordinates=source.reference_coordinates() if linear else None,
        target_coordinates=target.reference_coordinates() if linear else None,
    )


def _orientation(mask: int, dimension: int, /) -> tuple[int, ...]:
    return tuple(axis for axis in range(dimension) if (mask >> axis) & 1)


class _EntityKeyCodec(StrictModule, NonTrainableState):
    """Collision-free int64 keys of ``(level, orientation mask, coordinate)``."""

    dimension: int = eqx.field(static=True)
    coordinate_bits: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, topology: ForestHierarchyTopology, /) -> None:
        plan = topology.plan
        bits = tuple(
            int(extent << plan.maximum_level).bit_length() for extent in plan.root_shape
        )
        if int(plan.maximum_level).bit_length() + plan.dimension + sum(bits) > 62:
            raise ValueError("Forest cochain entity keys exceed the int64 budget.")
        self.dimension = plan.dimension
        self.coordinate_bits = bits

    def encode(self, levels: Any, masks: Any, coordinates: Any, /) -> np.ndarray:
        key = (np.asarray(levels, dtype=np.int64) << self.dimension) | np.asarray(
            masks, dtype=np.int64
        )
        for axis, bits in enumerate(self.coordinate_bits):
            key = (key << bits) | np.asarray(coordinates, dtype=np.int64)[:, axis]
        return key


def _canonical_entities(
    topology: Any, levels: Any, masks: Any, coordinates: Any, /
) -> np.ndarray:
    """Wrap periodic normal coordinates of level-lattice entities."""
    plan = topology.plan
    extent = np.asarray(plan.root_shape, dtype=np.int64)[None, :] << levels[:, None]
    periodic = np.asarray(plan.periodic_axes, dtype=np.bool_)[None, :]
    tangent = ((masks[:, None] >> np.arange(plan.dimension)[None, :]) & 1).astype(
        np.bool_
    )
    return np.where(periodic & ~tangent, coordinates % extent, coordinates)


def _entity_owners(
    topology: ForestHierarchyTopology,
    levels: np.ndarray,
    masks: np.ndarray,
    coordinates: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Coarsest leaf of level ``<=`` the entity level whose closure contains it.

    Returns ``(slot, level)`` with ``-1`` for entities touched only by finer
    leaves.  Candidates are the level cells sharing the entity along its normal
    axes; ties resolve to the lowest slot.
    """
    plan = topology.plan
    dimension = plan.dimension
    leaf_levels = topology.leaf_levels()
    keys = topology.leaf_keys()
    offsets = np.indices((2,) * dimension).reshape(dimension, -1).T.astype(np.int64)
    offset_masks = np.sum(offsets << np.arange(dimension)[None, :], axis=1)
    admissible = (offset_masks[None, :] & masks[:, None]) == 0
    cells = coordinates[:, None, :] - offsets[None, :, :]
    extent = (np.asarray(plan.root_shape, dtype=np.int64)[None, :] << levels[:, None])[
        :, None, :
    ]
    periodic = np.asarray(plan.periodic_axes, dtype=np.bool_)
    cells = np.where(periodic, cells % extent, cells)
    inside = admissible & np.all((cells >= 0) & (cells < extent), axis=-1)
    slots = np.full(inside.shape, -1, dtype=np.int64)
    candidate_levels = np.broadcast_to(levels[:, None], inside.shape)[inside]
    slots[inside] = _locate(keys, _node_keys(plan, candidate_levels, cells[inside]))
    owner_levels = np.where(
        inside, leaf_levels[slots.clip(min=0)], plan.maximum_level + 1
    )
    eligible = inside & (owner_levels <= levels[:, None])
    rank = np.where(
        eligible,
        owner_levels * (topology.leaf_count + 1) + slots,
        np.iinfo(np.int64).max,
    )
    choice = np.argmin(rank, axis=1)
    rows = np.arange(levels.shape[0])
    found = eligible[rows, choice]
    return (
        np.where(found, slots[rows, choice], -1),
        np.where(found, owner_levels[rows, choice], -1),
    )


def _local_entity_table(dimension: int, /) -> tuple[np.ndarray, np.ndarray]:
    """Orientation masks and offsets of the ``3**d`` closure entities of a cell."""
    masks = []
    offsets = []
    for mask in range(1 << dimension):
        normal = [axis for axis in range(dimension) if not (mask >> axis) & 1]
        for bits in product((0, 1), repeat=len(normal)):
            offset = np.zeros((dimension,), dtype=np.int64)
            offset[normal] = bits
            masks.append(mask)
            offsets.append(offset)
    return np.asarray(masks, dtype=np.int64), np.asarray(offsets, dtype=np.int64)


class _MasterIndex(StrictModule, NonTrainableState):
    """Sorted master entity keys of one forest, per cochain degree (host)."""

    levels: tuple[np.ndarray, ...]
    masks: tuple[np.ndarray, ...]
    coordinates: tuple[np.ndarray, ...]
    keys: tuple[np.ndarray, ...]


def _master_entities(topology: ForestHierarchyTopology, codec: _EntityKeyCodec, /) -> Any:
    """Leaf closure entities owned at their own level, sorted per degree."""
    dimension = topology.plan.dimension
    table_masks, table_offsets = _local_entity_table(dimension)
    leaf_levels = topology.leaf_levels()
    leaf_coordinates = topology.leaf_coordinates()
    entity_count = table_masks.shape[0]
    levels = np.repeat(leaf_levels, entity_count)
    masks = np.tile(table_masks, topology.leaf_count)
    coordinates = (leaf_coordinates[:, None, :] + table_offsets[None, :, :]).reshape(
        -1, dimension
    )
    coordinates = _canonical_entities(topology, levels, masks, coordinates)
    keys = codec.encode(levels, masks, coordinates)
    keys, first = np.unique(keys, return_index=True)
    levels, masks, coordinates = levels[first], masks[first], coordinates[first]
    _, owner_levels = _entity_owners(topology, levels, masks, coordinates)
    master = owner_levels == levels
    degrees = np.bitwise_count(masks.astype(np.uint64)).astype(np.int64)
    per_degree = tuple(master & (degrees == degree) for degree in range(dimension + 1))
    return _MasterIndex(
        levels=tuple(levels[selected] for selected in per_degree),
        masks=tuple(masks[selected] for selected in per_degree),
        coordinates=tuple(coordinates[selected] for selected in per_degree),
        keys=tuple(keys[selected] for selected in per_degree),
    )


_Entities = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _subdivided_entities(entities: _Entities, selected: np.ndarray, /) -> list[_Entities]:
    """Level-``l + 1`` tangent subdivision of selected weighted entities."""
    rows, levels, masks, coordinates, coefficients = entities
    dimension = coordinates.shape[1]
    offsets = np.indices((2,) * dimension).reshape(dimension, -1).T.astype(np.int64)
    offset_masks = np.sum(offsets << np.arange(dimension)[None, :], axis=1)
    parts = []
    for offset, offset_mask in zip(offsets, offset_masks, strict=True):
        chosen = selected & ((masks & offset_mask) == offset_mask)
        if np.any(chosen):
            parts.append(
                (
                    rows[chosen],
                    levels[chosen] + 1,
                    masks[chosen],
                    2 * coordinates[chosen] + offset[None, :],
                    coefficients[chosen],
                )
            )
    return parts


def _interpolated_entities(
    topology: ForestHierarchyTopology,
    entities: _Entities,
    owners: np.ndarray,
    selected: np.ndarray,
    /,
) -> list[_Entities]:
    """Owner-leaf closure entities weighted by the commuting prolongation.

    Entities sharing one owner and level form one call of the canonical
    tensor-product prolongation from :mod:`._entity_transfer`.
    """
    rows, levels, masks, coordinates, coefficients = entities
    plan = topology.plan
    dimension = plan.dimension
    table_masks, table_offsets = _local_entity_table(dimension)
    leaf_levels = topology.leaf_levels()
    leaf_coordinates = topology.leaf_coordinates()
    indices = np.flatnonzero(selected)
    indices = indices[np.lexsort((levels[indices], owners[indices]))]
    boundaries = np.flatnonzero(np.diff(owners[indices]) | np.diff(levels[indices])) + 1
    parts = []
    for group in np.split(indices, boundaries):
        owner = owners[group[0]]
        owner_level = leaf_levels[owner]
        local_levels = np.full(table_masks.shape, owner_level, dtype=np.int64)
        local_coordinates = _canonical_entities(
            topology,
            local_levels,
            table_masks,
            leaf_coordinates[owner][None, :] + table_offsets,
        )
        sources, targets, weights = _prolongation_routes(
            tuple(
                (_orientation(int(mask), dimension), tuple(coordinate.tolist()))
                for mask, coordinate in zip(table_masks, local_coordinates, strict=True)
            ),
            tuple(
                (
                    _orientation(int(masks[index]), dimension),
                    tuple(coordinates[index].tolist()),
                )
                for index in group
            ),
            1 << int(levels[group[0]] - owner_level),
            plan.level_shape(int(owner_level)),
            plan.periodic_axes,
        )
        sources = np.asarray(sources, dtype=np.int64)
        targets = group[np.asarray(targets, dtype=np.int64)]
        parts.append(
            (
                rows[targets],
                local_levels[sources],
                table_masks[sources],
                local_coordinates[sources],
                coefficients[targets] * np.asarray(weights, dtype=np.float64),
            )
        )
    return parts


def _extension_routes(
    topology: ForestHierarchyTopology,
    codec: _EntityKeyCodec,
    masters: _MasterIndex,
    degree: int,
    rows: np.ndarray,
    levels: np.ndarray,
    masks: np.ndarray,
    coordinates: np.ndarray,
    coefficients: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Express weighted degree-``k`` entities through the forest's master cochains.

    Master entities contribute directly.  An entity inside a coarser owner leaf is
    interpolated from that leaf's closure entities by the tensor-product commuting
    prolongation of :mod:`._entity_transfer`; an entity touched only by finer
    leaves is the sum of its level-``l + 1`` tangent subdivision.  Subdivision
    waves only go finer and interpolation waves only go coarser (an interpolated
    closure entity always has an owner), so at most ``2 L + 2`` waves run.
    Returns coalesced ``(row, master, coefficient)`` triplets.
    """
    dimension = topology.plan.dimension
    master_keys = masters.keys[degree]
    emitted: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    pending: list[_Entities] = [
        (
            np.asarray(rows, dtype=np.int64),
            np.asarray(levels, dtype=np.int64),
            np.asarray(masks, dtype=np.int64),
            np.asarray(coordinates, dtype=np.int64).reshape(-1, dimension),
            np.asarray(coefficients, dtype=np.float64),
        )
    ]
    for _ in range(2 * topology.plan.maximum_level + 3):
        if not pending:
            break
        rows_, levels_, masks_, coordinates_, coefficients_ = (
            np.concatenate(column) for column in zip(*pending, strict=True)
        )
        coordinates_ = _canonical_entities(topology, levels_, masks_, coordinates_)
        entities = (rows_, levels_, masks_, coordinates_, coefficients_)
        owners, owner_levels = _entity_owners(topology, levels_, masks_, coordinates_)
        direct = owner_levels == levels_
        if np.any(direct):
            keys = codec.encode(levels_[direct], masks_[direct], coordinates_[direct])
            position = np.searchsorted(master_keys, keys).clip(
                max=max(master_keys.size - 1, 0)
            )
            if master_keys.size == 0 or np.any(master_keys[position] != keys):
                raise RuntimeError("Forest cochain extension lost a master entity.")
            emitted.append((rows_[direct], position, coefficients_[direct]))
        pending = _subdivided_entities(entities, owners < 0)
        coarser = (owners >= 0) & ~direct
        if np.any(coarser):
            pending.extend(_interpolated_entities(topology, entities, owners, coarser))
    if pending:
        raise RuntimeError("Forest cochain extension did not terminate.")
    if not emitted:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float64),
        )
    all_rows, all_masters, all_coefficients = (
        np.concatenate(column) for column in zip(*emitted, strict=True)
    )
    pairs, inverse = np.unique(
        np.stack((all_rows, all_masters), axis=1), axis=0, return_inverse=True
    )
    summed = np.bincount(inverse.reshape(-1), weights=all_coefficients)
    keep = summed != 0.0
    return pairs[keep, 0], pairs[keep, 1], summed[keep]


def _sparse_operator(
    rows: np.ndarray,
    columns: np.ndarray,
    coefficients: np.ndarray,
    source_space: ArraySpace,
    target_space: ArraySpace,
    kind: str,
    /,
) -> SparseCoordinateOperator:
    capacity = forest_capacity_bucket(rows.size, 1)
    return _pad_operator(
        # ty: ignore[invalid-argument-type]
        columns,
        # ty: ignore[invalid-argument-type]
        rows,
        # ty: ignore[invalid-argument-type]
        coefficients,
        capacity,
        source_space,
        target_space,
        canonical_fingerprint(
            {
                "kind": kind,
                "source": source_space.space_id,
                "target": target_space.space_id,
                "routes": array_tree_fingerprint((rows, columns, coefficients)),
            }
        ),
    )


@final
class ForestCochainComplex(StrictModule, NonTrainableState):
    """Constrained cubical cochain complex of one 2:1-balanced forest.

    Degree-``k`` degrees of freedom live on master entities: leaf closure entities
    that are not interior to a coarser leaf's closure.  Entities hanging on coarser
    leaves are constrained by the commuting tensor-product prolongation, so the
    coboundaries ``d_k`` carry the constrained incidence and satisfy
    ``d_{k+1} d_k = 0`` (reported as ``nilpotency_defect``).
    """

    topology: ForestHierarchyTopology
    entity_levels: tuple[Array, ...]
    entity_orientations: tuple[Array, ...]
    entity_coordinates: tuple[Array, ...]
    entity_valid: tuple[Array, ...]
    spaces: tuple[ArraySpace, ...]
    coboundaries: tuple[SparseCoordinateOperator, ...]
    active_coboundaries: tuple[SparseCoordinateOperator, ...]
    entity_counts: tuple[int, ...] = eqx.field(static=True)
    nilpotency_defect: float = eqx.field(static=True)
    complex_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: ForestHierarchyTopology,
        /,
        *,
        dtype: DTypeLike = jnp.float64,
    ) -> None:
        if not isinstance(topology, ForestHierarchyTopology):
            raise TypeError("Forest cochain complexes require a forest topology.")
        dtype_ = jnp.dtype(dtype)
        if not jnp.issubdtype(dtype_, jnp.inexact):
            raise TypeError("Forest cochain dtype must be inexact.")
        dimension = topology.plan.dimension
        codec = _EntityKeyCodec(topology)
        masters = _master_entities(topology, codec)
        counts = tuple(keys.size for keys in masters.keys)
        capacities = tuple(forest_capacity_bucket(count, 1) for count in counts)
        spaces = tuple(
            ArraySpace(
                (capacity,),
                dtype=dtype_,
                space_id=canonical_fingerprint(
                    {
                        "kind": "forest-storage-coordinates",
                        "topology": topology.topology_id,
                        "degree": degree,
                        "capacity": capacity,
                        "dtype": dtype_.str,
                        "masters": array_tree_fingerprint(masters.keys[degree]),
                    }
                ),
            )
            for degree, capacity in enumerate(capacities)
        )
        coboundaries = []
        for degree in range(dimension):
            upper_levels = masters.levels[degree + 1]
            upper_masks = masters.masks[degree + 1]
            upper_coordinates = masters.coordinates[degree + 1]
            face_rows, face_levels, face_masks, face_coordinates, face_signs = (
                [],
                [],
                [],
                [],
                [],
            )
            for axis in range(dimension):
                carries = ((upper_masks >> axis) & 1).astype(np.bool_)
                if not np.any(carries):
                    continue
                selected = np.flatnonzero(carries)
                position = np.bitwise_count(
                    (upper_masks[selected] & ((1 << axis) - 1)).astype(np.uint64)
                ).astype(np.int64)
                lower_mask = upper_masks[selected] & ~(1 << axis)
                shifted = upper_coordinates[selected].copy()
                shifted[:, axis] += 1
                for coordinate, sign in (
                    (upper_coordinates[selected], -((-1.0) ** position)),
                    (shifted, (-1.0) ** position),
                ):
                    face_rows.append(selected)
                    face_levels.append(upper_levels[selected])
                    face_masks.append(lower_mask)
                    face_coordinates.append(coordinate)
                    face_signs.append(sign)
            rows, columns, coefficients = _extension_routes(
                topology,
                codec,
                masters,
                degree,
                np.concatenate(face_rows),
                np.concatenate(face_levels),
                np.concatenate(face_masks),
                np.concatenate(face_coordinates),
                np.concatenate(face_signs),
            )
            coboundaries.append(
                _sparse_operator(
                    rows,
                    columns,
                    coefficients,
                    spaces[degree],
                    spaces[degree + 1],
                    "forest-cochain-coboundary",
                )
            )
        matrices = tuple(_scipy_matrix(operator) for operator in coboundaries)
        nilpotency = max(
            (
                float(np.max(np.abs((upper @ lower).data), initial=0.0))
                for lower, upper in zip(matrices[:-1], matrices[1:], strict=True)
            ),
            default=0.0,
        )
        if nilpotency > 1.0e-12:
            raise ValueError("Forest cochain coboundaries violate d o d = 0.")

        def padded(values: Any, capacity: Any, fill: Any, dtype_host: Any) -> Any:
            array = np.asarray(values, dtype=dtype_host)
            result = np.full((capacity,) + array.shape[1:], fill, dtype=dtype_host)
            result[: array.shape[0]] = array
            return jnp.asarray(result)

        self.topology = topology
        self.entity_levels = tuple(
            padded(values, capacity, -1, np.int32)
            for values, capacity in zip(masters.levels, capacities, strict=True)
        )
        self.entity_orientations = tuple(
            padded(values, capacity, -1, np.int32)
            for values, capacity in zip(masters.masks, capacities, strict=True)
        )
        self.entity_coordinates = tuple(
            padded(values, capacity, -1, np.int32)
            for values, capacity in zip(masters.coordinates, capacities, strict=True)
        )
        self.entity_valid = tuple(
            jnp.asarray(np.arange(capacity) < count)
            for count, capacity in zip(counts, capacities, strict=True)
        )
        self.spaces = spaces
        self.coboundaries = tuple(coboundaries)
        self.entity_counts = counts
        self.nilpotency_defect = nilpotency
        self.complex_id = canonical_fingerprint(
            {
                "kind": "forest-cochain-complex",
                "topology": topology.topology_id,
                "entities": [array_tree_fingerprint(keys) for keys in masters.keys],
                "coboundaries": [operator.operator_id for operator in coboundaries],
            }
        )
        active_spaces = tuple(
            ArraySpace(
                (count,),
                dtype=dtype_,
                space_id=canonical_fingerprint(
                    {
                        "kind": "forest-active-coordinates",
                        "complex": self.complex_id,
                        "degree": degree,
                        "active_indices": np.arange(count, dtype=np.int32),
                    }
                ),
            )
            for degree, count in enumerate(counts)
        )
        self.active_coboundaries = tuple(
            _active_operator(operator, active_spaces[degree], active_spaces[degree + 1])
            for degree, operator in enumerate(coboundaries)
        )

    def _masters(self, /) -> _MasterIndex:
        codec = _EntityKeyCodec(self.topology)
        levels = tuple(
            np.asarray(values, dtype=np.int64)[:count]
            for values, count in zip(self.entity_levels, self.entity_counts, strict=True)
        )
        masks = tuple(
            np.asarray(values, dtype=np.int64)[:count]
            for values, count in zip(
                self.entity_orientations, self.entity_counts, strict=True
            )
        )
        coordinates = tuple(
            np.asarray(values, dtype=np.int64)[:count]
            for values, count in zip(
                self.entity_coordinates, self.entity_counts, strict=True
            )
        )
        return _MasterIndex(
            levels=levels,
            masks=masks,
            coordinates=coordinates,
            keys=tuple(
                codec.encode(level, mask, coordinate)
                for level, mask, coordinate in zip(
                    levels, masks, coordinates, strict=True
                )
            ),
        )

    def coboundary(self, degree: int, /) -> SparseCoordinateOperator:
        """Constrained discrete exterior derivative ``d_k : C^k -> C^{k+1}``."""
        index = int(degree)
        if index < 0 or index >= len(self.coboundaries):
            raise ValueError("Forest cochain degree is out of range.")
        return self.coboundaries[index]

    def hilbert_complex(self, hodges: Sequence[CochainHodge], /) -> HilbertComplex:
        """Metric Hilbert realization of the constrained master-entity complex.

        Only master coordinates enter the spaces. Capacity padding therefore
        cannot appear as harmonic modes, and a non-diagonal Hodge is restricted
        before its inverse is prepared.
        """
        metrics = tuple(hodges)
        if len(metrics) != len(self.spaces):
            raise ValueError("Forest Hodges must cover every cochain degree.")
        active_spaces: list[ArraySpace] = []
        for degree, (metric, storage, count) in enumerate(
            zip(metrics, self.spaces, self.entity_counts, strict=True)
        ):
            if metric.size != storage.size:
                raise ValueError("Forest Hodge size must match its storage capacity.")
            indices = np.arange(count, dtype=np.int32)
            mask_id = canonical_fingerprint(
                {
                    "complex": self.complex_id,
                    "degree": degree,
                    "active_indices": indices,
                }
            )
            space, _ = metric.restrict(indices).make_space(
                space_id=mask_id,
                dtype=storage.dtype,
            )
            active_spaces.append(space)
        differentials = tuple(
            SparseCoordinateOperator(
                operator.relation,
                operator.coefficients,
                source=active_spaces[degree],
                target=active_spaces[degree + 1],
                operator_id=operator.operator_id,
            )
            for degree, operator in enumerate(self.active_coboundaries)
        )
        return HilbertComplex(
            tuple(active_spaces),
            differentials,
            complex_id=canonical_fingerprint(
                {
                    "kind": "forest-hilbert-complex",
                    "topology": self.complex_id,
                    "spaces": [space.space_id for space in active_spaces],
                    "pairings": [space.pairing.pairing_id for space in active_spaces],
                }
            ),
        )


def _cochain_extension(
    source: ForestCochainComplex,
    target: ForestCochainComplex,
    degree: int,
    /,
) -> SparseCoordinateOperator:
    """Source-to-target master map of one degree by extension over the source."""
    source_masters = source._masters()
    target_masters = target._masters()
    count = target.entity_counts[degree]
    rows, columns, coefficients = _extension_routes(
        source.topology,
        _EntityKeyCodec(source.topology),
        source_masters,
        degree,
        np.arange(count, dtype=np.int64),
        target_masters.levels[degree],
        target_masters.masks[degree],
        target_masters.coordinates[degree],
        np.ones((count,), dtype=np.float64),
    )
    return _sparse_operator(
        rows,
        columns,
        coefficients,
        source.spaces[degree],
        target.spaces[degree],
        "forest-cochain-transfer",
    )


@final
class ForestCochainTransfer(StrictModule, NonTrainableState):
    """Commuting cochain transfer between a forest complex and its refinement.

    Mirrors :class:`CompatibleEntityTransferFamily`: ``prolongation`` maps coarse to
    fine masters by cochain extension (the commuting tensor-product prolongation
    inside every coarse leaf), and ``restriction`` samples the fine cochain on the
    coarse masters, its exact left inverse.  Each degree qualifies the constant
    (degree zero), round-trip ``R P = I``, and commuting ``d P = P d`` defects to
    ``1e-12``.  Epochs that both refine and coarsen compose a prolongation onto
    :func:`forest_common_refinement` with a restriction from it.
    """

    coarse: ForestCochainComplex
    fine: ForestCochainComplex
    transfers: tuple[CompatibleEntityTransfer, ...]
    complex_map: ComplexMap
    evidence: ComplexMapEvidence
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self, coarse: ForestCochainComplex, fine: ForestCochainComplex, /
    ) -> None:
        if not isinstance(coarse, ForestCochainComplex) or not isinstance(
            fine, ForestCochainComplex
        ):
            raise TypeError("Forest cochain transfers require cochain complexes.")
        coarse_topology = coarse.topology
        fine_topology = fine.topology
        if coarse_topology.plan.plan_id != fine_topology.plan.plan_id:
            raise ValueError("Forest cochain transfers cannot change the forest plan.")
        containing = _locate(coarse_topology.leaf_keys(), fine_topology.leaf_keys())
        if np.any(
            coarse_topology.leaf_levels()[containing] > fine_topology.leaf_levels()
        ):
            raise ValueError("The fine forest must refine the coarse forest.")
        dimension = coarse_topology.plan.dimension
        prolongations = tuple(
            _cochain_extension(coarse, fine, degree) for degree in range(dimension + 1)
        )
        restrictions = tuple(
            _cochain_extension(fine, coarse, degree) for degree in range(dimension + 1)
        )
        prolongation_matrices = tuple(
            _scipy_matrix(operator) for operator in prolongations
        )
        restriction_matrices = tuple(_scipy_matrix(operator) for operator in restrictions)
        coarse_d = tuple(_scipy_matrix(operator) for operator in coarse.coboundaries)
        fine_d = tuple(_scipy_matrix(operator) for operator in fine.coboundaries)
        tolerance = 1.0e-12
        commuting_defects: list[float] = []
        transfers = []
        for degree in range(dimension + 1):
            count = coarse.entity_counts[degree]
            active = np.flatnonzero(np.arange(coarse.spaces[degree].size) < count)
            roundtrip = (
                restriction_matrices[degree] @ prolongation_matrices[degree]
            ).tocsr() - sp.identity(coarse.spaces[degree].size, format="csr")
            roundtrip_defect = float(np.max(np.abs(roundtrip[active].data), initial=0.0))
            if degree == 0:
                ones = np.zeros((coarse.spaces[0].size,), dtype=np.float64)
                ones[:count] = 1.0
                image = prolongation_matrices[0] @ ones
                constant_defect = float(
                    np.max(np.abs(image[: fine.entity_counts[0]] - 1.0), initial=0.0)
                )
            else:
                constant_defect = 0.0
            if degree < dimension:
                commutator = (
                    fine_d[degree] @ prolongation_matrices[degree]
                    - prolongation_matrices[degree + 1] @ coarse_d[degree]
                )
                commuting_defect = float(np.max(np.abs(commutator.data), initial=0.0))
            else:
                commuting_defect = 0.0
            if degree < dimension:
                commuting_defects.append(commuting_defect)
            if max(constant_defect, roundtrip_defect, commuting_defect) > tolerance:
                raise ValueError(
                    "Forest cochain transfer failed constant/roundtrip/commuting qualification."
                )
            prolongation = prolongations[degree]
            restriction = restrictions[degree]
            transfers.append(
                CompatibleEntityTransfer(
                    degree,
                    prolongation,
                    restriction,
                    dual_transpose(prolongation),
                    adjoint(prolongation),
                    constant_defect,
                    roundtrip_defect,
                    canonical_fingerprint(
                        {
                            "kind": "forest-cochain-degree-transfer",
                            "degree": degree,
                            "prolongation": prolongation.operator_id,
                            "restriction": restriction.operator_id,
                        }
                    ),
                )
            )
        self.coarse = coarse
        self.fine = fine
        self.transfers = tuple(transfers)
        source = coarse.hilbert_complex(
            tuple(
                DiagonalHodge(jnp.ones(space.shape, dtype=jnp.float64))
                for space in coarse.spaces
            )
        )
        target = fine.hilbert_complex(
            tuple(
                DiagonalHodge(jnp.ones(space.shape, dtype=jnp.float64))
                for space in fine.spaces
            )
        )
        maps = tuple(
            _active_operator(
                value.prolongation, source.space(degree), target.space(degree)
            )
            for degree, value in enumerate(transfers)
        )
        self.complex_map = ComplexMap(
            source,
            target,
            maps,
            map_id=canonical_fingerprint(
                {
                    "kind": "forest-cochain-complex-map",
                    "maps": [operator.operator_id for operator in maps],
                }
            ),
        )
        self.evidence = ComplexMapEvidence(
            jnp.asarray(commuting_defects, dtype=jnp.float64),
            jnp.asarray(True, dtype=jnp.bool_),
        )
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "forest-cochain-transfer",
                "coarse": coarse.complex_id,
                "fine": fine.complex_id,
                "transfers": [transfer.transfer_id for transfer in transfers],
            }
        )

    def transfer(self, degree: int, /) -> CompatibleEntityTransfer:
        index = int(degree)
        if index < 0 or index >= len(self.transfers):
            raise ValueError("Forest cochain transfer degree is out of range.")
        return self.transfers[index]


__all__ = [
    "ForestCochainComplex",
    "ForestCochainTransfer",
    "ForestFieldTransition",
    "ForestRefluxRoutes",
    "ForestTransferResult",
    "ForestTransferRoutes",
    "ForestVertexLayout",
    "forest_vertex_interpolation",
]
