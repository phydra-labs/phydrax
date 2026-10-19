#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Certify target maps from sharded dyadic restriction evidence.

Mesh certificate checks
-----------------------
[
  {
    "checks": [
      "cell_validity"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Every process reports certified status from the target cell-geometry validity certificate."
      },
      {
        "tag": "X",
        "statement": "Owned global cell counts form the declared inventory with one storage partition per process."
      }
    ],
    "conclusion": "All-owner validity is accepted only when every process certifies its owned target maps; otherwise collective validity is unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._collective_domain_coverage._collective_validity",
      "locator": "phydrax.geometry._collective_domain_coverage._collective_validity",
      "url": ""
    },
    "failure_statuses": []
  },
  {
    "checks": [
      "collective_cell_validity"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Every process reports certified status from the target cell-geometry validity certificate."
      },
      {
        "tag": "X",
        "statement": "Owned global cell counts form the declared inventory with one storage partition per process."
      }
    ],
    "conclusion": "All-owner validity is accepted only when every process certifies its owned target maps; otherwise collective validity is unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._collective_domain_coverage._collective_validity",
      "locator": "phydrax.geometry._collective_domain_coverage._collective_validity",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "facet_pairing"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Each recomputed facet vertex record matches a logical facet and has incidence multiplicity one or two."
      }
    ],
    "conclusion": "Target interior facets are paired and boundary facets have one incident target cell.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._collective_domain_coverage._facet_incidence",
      "locator": "phydrax.geometry._collective_domain_coverage._facet_incidence",
      "url": ""
    },
    "failure_statuses": []
  },
  {
    "checks": [
      "boundary_contact"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Target barycentric charts follow the disjoint dyadic root partition; each unshared facet has zero root barycentric coordinate on a root boundary face."
      }
    ],
    "conclusion": "Every unshared target facet lies on the root boundary.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._collective_domain_coverage._boundary_contacts",
      "locator": "phydrax.geometry._collective_domain_coverage._boundary_contacts",
      "url": ""
    },
    "failure_statuses": []
  },
  {
    "checks": [
      "embedding_premise"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "The newly bound target embedding certificate reports certified status before coverage is assembled."
      }
    ],
    "conclusion": "An uncertified target embedding leaves target domain coverage unresolved.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._collective_domain_coverage.certify_collective_premises",
      "locator": "phydrax.geometry._collective_domain_coverage.certify_collective_premises",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  },
  {
    "checks": [
      "mapped_coverage_resource_budget"
    ],
    "premises": [
      {
        "tag": "X",
        "statement": "Independent affine root integrals fit the coordinate request work and storage limits."
      }
    ],
    "conclusion": "Resource exhaustion produces unresolved coverage with absent achieved integrals.",
    "citation": {
      "kind": "definition",
      "reference": "phydrax.geometry._collective_domain_coverage.certify_collective_premises",
      "locator": "phydrax.geometry._collective_domain_coverage.certify_collective_premises",
      "url": ""
    },
    "failure_statuses": [
      "unresolved"
    ]
  }
]
End mesh certificate checks
"""

from __future__ import annotations

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental import multihost_utils

import phydrax.discretization._coordinate_budget as coordinate_budget_owner
from phydrax.geometry._embedding_state import (
    coordinate_certificate_request,
    EmbeddingState,
    MeshCertificateLimits,
)

from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    logical_array_value_collection_digest,
)
from ..discretization._adaptive_simplex import AdaptiveSimplexState
from ..discretization._cell_geometry import CellGeometrySpec, coordinate_lagrange_element
from ..discretization._cell_geometry_validity import (
    cell_geometry_id,
    CellValidityCertificate,
    CellValidityPolicy,
    certify_cell_geometry_validity,
)
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_budget import CoordinateEnclosureBudget
from ..discretization._coordinate_enclosure import coordinate_source_signature
from ..typing import checked
from ._domain_coverage import (
    coverage_result,
    declared_coverage_measures,
    DomainCoverageCertificate,
    mapped_coverage_measures,
    PiecewiseLinearDomain,
)
from ._global_embedding import EmbeddingCounts, GlobalEmbeddingCertificate
from ._mesh_certificates import active_expression_work, MeshCertificateBinding


def _require(predicate: Array, message: str) -> None:
    if not bool(jax.device_get(jnp.all(predicate))):
        raise ValueError(message)


def _bank(arrays: dict[str, Array], name: str) -> Array:
    value = arrays.get(name)
    if value is None:
        raise ValueError(
            f"collective.logical_arrays must contain bank {name!r}; got absent bank"
        )
    return value


@jax.jit
def _lookup(keys: Array, query: Array) -> tuple[Array, Array]:
    order = jnp.argsort(keys, stable=True)
    sorted_keys = keys[order]
    position = jnp.minimum(jnp.searchsorted(sorted_keys, query), sorted_keys.shape[0] - 1)
    rows = order[position]
    return rows, sorted_keys[position] == query


def _source_identity(value: str) -> Array:
    return jnp.asarray(
        np.frombuffer(bytes.fromhex(canonical_fingerprint(value)), dtype=np.uint8).copy()
    )


def _raw_partition(states: AdaptiveSimplexState, initial: AdaptiveSimplexState) -> Array:
    """Check living binary families and their root partition.

    Child vertex sets are exactly the two halves of the parent split edge.
    Parent/child reciprocity and strictly increasing allocation imply a forest;
    every living parent is either an active leaf or has both complementary halves.
    Coarsened history is allowed only when both historical children are retired.
    """
    if states.mesh.cells.ndim != 3 or initial.mesh.cells.shape != states.mesh.cells.shape:
        raise ValueError(
            "collective.states.mesh.cells must match partition-leading simplex shapes; got incompatible arrays"
        )
    return _raw_partition_values(states, initial)


@jax.jit
def _raw_partition_values(
    states: AdaptiveSimplexState, initial: AdaptiveSimplexState
) -> Array:
    parts, capacity, width = states.mesh.cells.shape
    vertices = states.mesh.vertex_ids.shape[1]
    lane = jnp.arange(capacity, dtype=jnp.int32)[None, :]
    vertex_lane = jnp.arange(vertices, dtype=jnp.int32)[None, :]
    allocated = lane < states.cursors[:, 1:2]
    original = lane < initial.cursors[:, 1:2]
    original_vertices = vertex_lane < initial.cursors[:, 0:1]
    living = allocated & ~states.retired
    child = jnp.clip(states.children, 0, capacity - 1)
    first, second = child[:, :, 0], child[:, :, 1]
    part = jnp.arange(parts, dtype=jnp.int32)[:, None]
    has_children = jnp.all(states.children >= 0, axis=2)
    split = living & ~states.mesh.cell_active
    history = living & states.mesh.cell_active & has_children
    family = split | history
    parents = jnp.clip(states.parents // 2, 0, capacity - 1)
    ordinal = jnp.mod(states.parents, 2)
    roots = jnp.all(
        ~original
        | (
            (states.mesh.cell_ids == initial.mesh.cell_ids)
            & jnp.all(states.mesh.cells == initial.mesh.cells, axis=2)
            & jnp.all(states.tuples == initial.tuples, axis=2)
            & (states.tags == initial.tags)
            & (states.parents == initial.parents)
            & (states.blocks == initial.blocks)
            & (states.cell_classes == initial.cell_classes)
            & jnp.all(states.facet_classes == initial.facet_classes, axis=2)
            & (states.generations == initial.generations)
            & (~initial.retired | (states.retired & ~states.mesh.cell_active))
        )
    ) & jnp.all(
        ~original_vertices
        | (
            (states.mesh.vertex_ids == initial.mesh.vertex_ids)
            & jnp.all(states.mesh.coordinates == initial.mesh.coordinates, axis=2)
            & jnp.all(states.vertex_parents == initial.vertex_parents, axis=2)
        )
    )
    mid = jnp.clip(states.bisection_vertices, 0, vertices - 1)
    tag = jnp.clip(states.tags, 1, width - 1)
    endpoint0 = states.tuples[:, :, 0]
    endpoint1 = jnp.take_along_axis(states.tuples, tag[:, :, None], axis=2)[:, :, 0]
    split_parents = states.vertex_parents[part, mid]
    parent_sets = jnp.sort(states.mesh.cells, axis=2)
    first_expected = jnp.sort(
        jnp.where(
            states.mesh.cells == endpoint1[:, :, None], mid[:, :, None], states.mesh.cells
        ),
        axis=2,
    )
    second_expected = jnp.sort(
        jnp.where(
            states.mesh.cells == endpoint0[:, :, None], mid[:, :, None], states.mesh.cells
        ),
        axis=2,
    )
    families = jnp.all(
        ~family
        | (
            has_children
            & (states.children[:, :, 0] > lane)
            & (states.children[:, :, 1] > lane)
            & (states.children[:, :, 0] < states.cursors[:, 1:2])
            & (states.children[:, :, 1] < states.cursors[:, 1:2])
            & (states.children[:, :, 0] != states.children[:, :, 1])
            & (states.parents[part, first] == 2 * lane)
            & (states.parents[part, second] == 2 * lane + 1)
            & jnp.all(
                jnp.sort(states.mesh.cells[part, first], axis=2) == first_expected, axis=2
            )
            & jnp.all(
                jnp.sort(states.mesh.cells[part, second], axis=2) == second_expected,
                axis=2,
            )
            & jnp.any(parent_sets == endpoint0[:, :, None], axis=2)
            & jnp.any(parent_sets == endpoint1[:, :, None], axis=2)
            & (endpoint0 != endpoint1)
            & jnp.all(
                jnp.sort(split_parents, axis=2)
                == jnp.sort(jnp.stack((endpoint0, endpoint1), axis=2), axis=2),
                axis=2,
            )
            & jnp.where(
                split,
                ~states.retired[part, first] & ~states.retired[part, second],
                states.retired[part, first] & states.retired[part, second],
            )
            & (states.cell_classes[part, first] == states.cell_classes)
            & (states.cell_classes[part, second] == states.cell_classes)
        )
    )
    branches = jnp.all(
        ~living
        | (
            (states.mesh.cell_active == ~split)
            & ((states.parents < 0) == (original & (initial.parents < 0)))
            & (
                (states.parents < 0)
                | (
                    (parents < lane)
                    & (states.children[part, parents, ordinal] == lane)
                    & (states.generations == states.generations[part, parents] + 1)
                )
            )
        )
    )
    retired = ~jnp.any(states.retired & states.mesh.cell_active) & ~jnp.any(
        allocated & (states.parents < 0) & states.retired
    )
    cursors = jnp.all(states.cursors[:, :2] >= initial.cursors[:, :2])
    return roots & families & branches & retired & cursors


def _ancestry(states: AdaptiveSimplexState) -> tuple[Array, Array]:
    """Return dyadic support on immutable root vertices for each numerical slot."""
    identifiers, weights, accepted = _ancestry_values(states)
    _require(
        accepted,
        "collective.vertex_ancestry must form an ordered dyadic source DAG; got inconsistent ancestry",
    )
    return identifiers, weights


@jax.jit
def _ancestry_values(states: AdaptiveSimplexState) -> tuple[Array, Array, Array]:
    parts, vertices = states.mesh.vertex_ids.shape
    width = states.mesh.cells.shape[2]
    sentinel = jnp.iinfo(jnp.int64).max
    identifiers = jnp.full((parts, vertices, width), -1, dtype=jnp.int64)
    weights = jnp.zeros((parts, vertices, width), dtype=jnp.float64)
    accepted = jnp.asarray(True)

    def step(
        slot: Array, carry: tuple[Array, Array, Array]
    ) -> tuple[Array, Array, Array]:
        ids, values, passed = carry
        parent = states.vertex_parents[:, slot]
        root = jnp.all(parent < 0, axis=1)
        safe = jnp.clip(parent, 0, vertices - 1)
        part = jnp.arange(parts, dtype=jnp.int32)
        keys = jnp.concatenate((ids[part, safe[:, 0]], ids[part, safe[:, 1]]), axis=1)
        amount = 0.5 * jnp.concatenate(
            (values[part, safe[:, 0]], values[part, safe[:, 1]]), axis=1
        )
        keys = jnp.where(amount != 0, keys, sentinel)
        order = jnp.argsort(keys, axis=1, stable=True)
        keys, amount = (
            jnp.take_along_axis(keys, order, axis=1),
            jnp.take_along_axis(amount, order, axis=1),
        )
        fresh = (keys != sentinel) & jnp.concatenate(
            (jnp.ones((parts, 1), dtype=jnp.bool_), keys[:, 1:] != keys[:, :-1]), axis=1
        )
        destination = jnp.where(
            keys != sentinel, jnp.cumsum(fresh, axis=1, dtype=jnp.int32) - 1, width
        )
        combined_ids = (
            jnp.full((parts, width + 1), sentinel, dtype=jnp.int64)
            .at[part[:, None], destination]
            .min(keys)
        )
        combined = (
            jnp.zeros((parts, width + 1), dtype=jnp.float64)
            .at[part[:, None], destination]
            .add(amount)
        )
        combined_ids = jnp.where(combined[:, :width] != 0, combined_ids[:, :width], -1)
        root_ids = (
            jnp.full((parts, width), -1, dtype=jnp.int64)
            .at[:, 0]
            .set(states.mesh.vertex_ids[:, slot])
        )
        root_values = jnp.zeros((parts, width), dtype=jnp.float64).at[:, 0].set(1.0)
        valid = root | (
            jnp.all(parent >= 0, axis=1)
            & jnp.all(parent < slot, axis=1)
            & (jnp.sum(fresh, axis=1) <= width)
            & (combined[:, width] == 0)
        )
        active = slot < states.cursors[:, 0]
        ids = ids.at[:, slot].set(jnp.where(root[:, None], root_ids, combined_ids))
        values = values.at[:, slot].set(
            jnp.where(root[:, None], root_values, combined[:, :width])
        )
        return ids, values, passed & jnp.all(~active | valid)

    return jax.lax.fori_loop(0, vertices, step, (identifiers, weights, accepted))


@jax.jit
def _epoch_cell_inventory(
    identifiers: Array, active: Array, expected_count: int
) -> tuple[Array, Array]:
    order = jnp.argsort(
        jnp.where(active, identifiers, jnp.iinfo(jnp.int64).max), stable=True
    )
    return order, jnp.sum(active, dtype=jnp.int64) == expected_count


def _global_epoch_banks(
    states: AdaptiveSimplexState, arrays: dict[str, Array], count: int
) -> tuple[Array, Array]:
    """Root-vertex support and dyadic weights of every active target corner."""
    ids = states.mesh.cell_ids.reshape(-1)
    active = states.mesh.cell_active.reshape(-1)
    full_order, complete_inventory = _epoch_cell_inventory(ids, active, count)
    order = full_order[:count]
    selected_ids = ids[order]
    _require(
        complete_inventory,
        "collective.cell_count must equal the active forest count; got differing counts",
    )
    _require(
        selected_ids[1:] > selected_ids[:-1],
        "collective.cell_global_ids must be strictly increasing; got duplicate identities",
    )
    _require(
        selected_ids == _bank(arrays, "cell_global_ids")[:count],
        "collective.cell_global_ids must match active epoch cells; got different identities",
    )
    part_count, _, width = states.mesh.cells.shape
    global_vertices = jnp.take_along_axis(
        states.mesh.vertex_ids, states.mesh.cells.reshape((part_count, -1)), axis=1
    ).reshape((-1, width))[order]
    _require(
        global_vertices == _bank(arrays, "cell_vertices")[:count],
        "collective.cell_vertices must match the epoch forest; got different corner identities",
    )
    source_ids, weights = _ancestry(states)
    # Gather per partition, flatten the partition-leading slot axis, then select
    # the active cells in increasing ID order.
    slots = (ids.shape[0], width, width)
    gathered_ids = jax.vmap(lambda source, rows: source[rows])(
        source_ids, states.mesh.cells
    ).reshape(slots)[order]
    gathered_weights = jax.vmap(lambda source, rows: source[rows])(
        weights, states.mesh.cells
    ).reshape(slots)[order]
    return gathered_ids, gathered_weights


def _exact_action_chart(actions: np.ndarray, width: int, /) -> tuple[Fraction, ...]:
    """Return root barycentrics of a rational coefficient action stack.

    Stack row 0 is the outermost action, so target corners are the rows of
    ``actions[0] @ ... @ actions[-1]``, as the serial
    ``BarycentricCellGeometryElement`` tabulation composes them. Each binary64
    weight is represented as a Fraction; an empty stack is the root identity.
    """
    chart = [
        [Fraction(int(row == column)) for column in range(width)] for row in range(width)
    ]
    for action in actions:
        weights = [[Fraction(float(value)) for value in row] for row in action]
        chart = [
            [
                sum(
                    (
                        chart[row][inner] * weights[inner][column]
                        for inner in range(width)
                    ),
                    Fraction(0),
                )
                for column in range(width)
            ]
            for row in range(width)
        ]
    return tuple(value for row in chart for value in row)


def _require_action_charts(
    counts: Array, actions: Array, valid: Array, barycentric: Array, width: int, /
) -> None:
    """Check coefficient action charts against dyadic ancestry.

    Every owner gathers the same global banks before rational comparison.
    """
    host_counts, host_actions, host_valid, expected = (
        np.asarray(value)
        for value in jax.device_get(
            multihost_utils.process_allgather(
                (counts, actions, valid, barycentric), tiled=True
            )
        )
    )
    host_valid = host_valid.astype(np.bool_)
    if (
        not np.issubdtype(host_counts.dtype, np.integer)
        or host_actions.dtype != np.float64
        or host_counts.shape != host_valid.shape
        or host_actions.ndim != 4
        or host_actions.shape[0] != host_counts.shape[0]
        or host_actions.shape[2:] != (width, width)
        or np.any(host_counts[host_valid] < 0)
        or np.any(host_counts[host_valid] > host_actions.shape[1])
        or not np.all(np.isfinite(host_actions[host_valid]))
    ):
        raise ValueError(
            "collective.geometry.action_weights must match declared depths and source width; got malformed stacks"
        )
    expected = expected.astype(np.float64)
    decided: set[tuple[bytes, bytes]] = set()
    for row in np.flatnonzero(host_valid):
        stack = host_actions[row, : host_counts[row]]
        key = (stack.tobytes(), expected[row].tobytes())
        if key in decided:
            continue
        if _exact_action_chart(stack, width) != tuple(
            Fraction(float(value)) for value in expected[row].reshape(-1)
        ):
            raise ValueError(
                "collective.geometry.action_weights must equal dyadic ancestry charts; got different coefficients"
            )
        decided.add(key)


@jax.jit
def _root_barycentric_values(
    source_vertex_ids: Array,
    source_weights: Array,
    source_corners: Array,
    root_rows: Array,
) -> tuple[Array, Array, Array]:
    root_vertices = source_corners[root_rows]
    barycentric = jnp.sum(
        jnp.where(
            source_vertex_ids[:, :, :, None] == root_vertices[:, None, None, :],
            source_weights[:, :, :, None],
            0.0,
        ),
        axis=2,
    )
    return root_vertices, barycentric, jnp.all(jnp.sum(barycentric, axis=2) == 1.0)


@jax.jit
def _root_bank_values_match(first: Array, second: Array) -> Array:
    return jnp.all(first == second)


@jax.jit
def _root_bank_masked_values_match(valid: Array, first: Array, second: Array) -> Array:
    return jnp.all(~valid[:, None] | (first == second))


@jax.jit
def _root_bank_rows_match(
    valid: Array, actual: Array, expected: Array, rows: Array
) -> Array:
    return jnp.all(~valid[:, None] | (actual == expected[rows]))


@jax.jit
def _root_parent_rows_match(
    valid: Array,
    present: Array,
    block_present: Array,
    target_sources: Array,
    parent_ids: Array,
    positions: Array,
) -> Array:
    return jnp.all(
        ~valid | (present & block_present & (target_sources[positions] == parent_ids))
    )


def _geometry_banks(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    states: AdaptiveSimplexState,
    arrays: dict[str, Array],
    count: int,
) -> tuple[Array, Array, Array]:
    """Return dyadic target roots, root rows and corner barycentrics."""
    elements, routes, coefficients = source_geometry.resolve(source_mesh)
    _require(
        _root_bank_values_match(
            _bank(arrays, "geometry/source_geometry_id"),
            _source_identity(cell_geometry_id(source_geometry)),
        ),
        "collective.geometry.source_geometry_id must match the root geometry; got different identity",
    )
    _require(
        _root_bank_values_match(
            _bank(arrays, "geometry/source_topology_id"),
            _source_identity(source_mesh.topology_id),
        ),
        "collective.geometry.source_topology_id must match the root topology; got different identity",
    )
    coordinate_count = coefficients.shape[0]
    _require(
        _root_bank_values_match(
            _bank(arrays, "geometry/coordinate_ids")[:coordinate_count],
            jnp.arange(coordinate_count, dtype=jnp.int64),
        ),
        "collective.geometry.coordinate_ids must match the root coefficient namespace; got different identities",
    )
    _require(
        _root_bank_values_match(
            _bank(arrays, "geometry/coordinates")[:coordinate_count], coefficients
        ),
        "collective.geometry.coordinates must equal root coefficients; got changed values",
    )
    source_vertex_ids, source_weights = _global_epoch_banks(states, arrays, count)
    target_sources = _bank(arrays, "organization/cell/source_ids")[:count]
    source_cell_ids = jnp.concatenate(
        tuple(jnp.asarray(block.global_ids) for block in source_mesh.blocks)
    )
    source_corners = jnp.concatenate(
        tuple(
            jnp.asarray(source_mesh.vertex_global_ids)[jnp.asarray(block.vertices)]
            for block in source_mesh.blocks
        )
    )
    root_rows, root_present = _lookup(source_cell_ids, target_sources)
    _require(
        root_present,
        "collective.organization.source_ids must identify root cells; got absent root",
    )
    root_vertices, barycentric, valid_barycentric = _root_barycentric_values(
        source_vertex_ids, source_weights, source_corners, root_rows
    )
    _require(
        valid_barycentric,
        "collective.geometry.barycentric must sum to one on the declared root; got exterior support",
    )
    covered = jnp.zeros((count,), dtype=jnp.int32)
    target_ids = _bank(arrays, "cell_global_ids")[:count]
    for block, element, route in zip(source_mesh.blocks, elements, routes, strict=True):
        if (
            block.cell_kind not in ("triangle", "tetrahedron")
            or element.element_id
            != coordinate_lagrange_element(block.cell_kind, 1).element_id
            or not np.array_equal(np.asarray(route), np.asarray(block.vertices))
        ):
            raise ValueError(
                "collective.source_geometry must be affine simplex geometry with vertex routes; got another basis or route"
            )
        prefix = f"geometry/cell_ids/{block.name}"
        cell_ids = _bank(arrays, prefix)
        valid = cell_ids >= 0
        positions, present = _lookup(target_ids, jnp.maximum(cell_ids, 0))
        parent_ids = _bank(arrays, f"geometry/parent_cell_ids/{block.name}")
        block_rows, block_present = _lookup(
            jnp.asarray(block.global_ids), jnp.maximum(parent_ids, 0)
        )
        _require(
            _root_parent_rows_match(
                valid, present, block_present, target_sources, parent_ids, positions
            ),
            "collective.organization.source_ids must identify root cells; got absent root",
        )
        _require(
            _root_bank_rows_match(
                valid,
                _bank(arrays, f"geometry/parent_vertex_ids/{block.name}"),
                root_vertices,
                positions,
            ),
            "collective.geometry.parent_vertex_ids must match ordered root corners; got different identities",
        )
        _require(
            _root_bank_rows_match(
                valid,
                _bank(arrays, f"geometry/routes/{block.name}"),
                jnp.asarray(route),
                block_rows,
            ),
            "collective.geometry.routes must match the root coefficient route; got changed route",
        )
        digest = canonical_fingerprint(
            {
                "signature": coordinate_source_signature(element),
                "arrays": array_tree_fingerprint(element),
            }
        )
        basis = jnp.asarray(np.frombuffer(bytes.fromhex(digest), dtype=np.uint8).copy())
        _require(
            _root_bank_masked_values_match(
                valid, _bank(arrays, f"geometry/source_basis/{block.name}"), basis
            ),
            "collective.geometry.source_basis must match the root basis fingerprint; got changed definition",
        )
        _require_action_charts(
            _bank(arrays, f"geometry/action_counts/{block.name}"),
            _bank(arrays, f"geometry/action_weights/{block.name}"),
            valid,
            barycentric[positions],
            block.vertices.shape[1],
        )
        covered = covered.at[jnp.where(valid, positions, count)].add(
            valid.astype(jnp.int32), mode="drop"
        )
    _require(
        covered == 1,
        "collective.geometry.cell_ids must cover each target cell once; got missing or repeated cells",
    )
    return target_sources, root_rows, barycentric


@jax.jit
def _facet_incidence_values(
    cell_vertices: Array, keys: Array, columns: Array
) -> tuple[Array, Array]:
    dimension = columns.shape[1]
    facets = keys.shape[0]
    packet = jnp.sort(cell_vertices[:, columns], axis=2).reshape((-1, dimension))
    joined = jnp.concatenate((keys, packet), axis=0)
    order = jnp.lexsort(tuple(joined[:, axis] for axis in reversed(range(dimension))))
    sorted_keys = joined[order]
    fresh = jnp.concatenate(
        (
            jnp.ones((1,), dtype=jnp.bool_),
            jnp.any(sorted_keys[1:] != sorted_keys[:-1], axis=1),
        )
    )
    group = jnp.cumsum(fresh, dtype=jnp.int32) - 1
    group_ids = jnp.empty_like(group).at[order].set(group)
    destination = (
        jnp.full((joined.shape[0],), -1, dtype=jnp.int32)
        .at[group_ids[:facets]]
        .set(jnp.arange(facets, dtype=jnp.int32))
    )
    incidence = destination[group_ids[facets:]]
    return incidence, jnp.all(incidence >= 0)


@jax.jit
def _facet_multiplicity_values(
    incidence: Array, facet_rows: Array
) -> tuple[Array, Array]:
    counts = jnp.zeros((facet_rows.shape[0],), dtype=jnp.int32).at[incidence].add(1)
    return counts, jnp.all((counts == 1) | (counts == 2))


def _facet_incidence(
    mesh: CellMesh, arrays: dict[str, Array], count: int
) -> tuple[Array, Array]:
    """Logical facet row of every (cell, opposite corner) and facet multiplicities.

    Facet incidence comes from logical cell vertex records. Facet IDs match
    lexicographic vertex records.
    """
    storage = mesh.storage
    if storage is None:
        raise ValueError(
            "collective.mesh.storage must describe facet incidence; got None"
        )
    dimension = mesh.topological_dimension
    facets = storage.global_entity_counts[dimension - 1]
    cell_vertices = _bank(arrays, "cell_vertices")[:count]
    columns = np.asarray(
        tuple(
            tuple(i for i in range(dimension + 1) if i != opposite)
            for opposite in range(dimension + 1)
        ),
        dtype=np.int32,
    )
    keys = _bank(arrays, f"entity_vertices_{dimension - 1}")[:facets]
    incidence, valid = _facet_incidence_values(cell_vertices, keys, columns)
    _require(
        valid,
        "collective.facet_incidence must identify every target facet; got absent facet",
    )
    counts, valid = _facet_multiplicity_values(incidence, keys)
    _require(
        valid,
        "collective.facet_multiplicity must equal one or two; got another multiplicity",
    )
    return incidence, counts


def _source_boundary_faces(source_mesh: CellMesh, /) -> np.ndarray:
    """Root faces (source cell order, opposite corner) with one incident root."""
    corners = np.concatenate(
        tuple(
            np.asarray(source_mesh.vertex_global_ids, dtype=np.int64)[
                np.asarray(block.vertices, dtype=np.int64)
            ]
            for block in source_mesh.blocks
        )
    )
    width = corners.shape[1]
    columns = np.asarray(
        tuple(
            tuple(i for i in range(width) if i != opposite) for opposite in range(width)
        ),
        dtype=np.int64,
    )
    faces = np.sort(corners[:, columns], axis=2).reshape((-1, width - 1))
    _, inverse, multiplicity = np.unique(
        faces, axis=0, return_inverse=True, return_counts=True
    )
    return (multiplicity[inverse.reshape(-1)] == 1).reshape((corners.shape[0], width))


@jax.jit
def _boundary_contact_values(boundary: Array, single: Array, barycentric: Array) -> Array:
    width = barycentric.shape[1]
    others = jnp.asarray(~np.eye(width, dtype=np.bool_))[None, :, :, None]
    on_face = jnp.all((barycentric == 0.0)[:, None, :, :] | ~others, axis=2)
    exterior = jnp.any(on_face & boundary[:, None, :], axis=2)
    return jnp.all(~single | exterior)


def _boundary_contacts(
    source_mesh: CellMesh,
    root_rows: Array,
    barycentric: Array,
    incidence: Array,
    counts: Array,
    count: int,
) -> None:
    """Every unshared target facet lies in an unshared face of its source root.

    A facet of a dyadic child lies in root face ``k`` exactly when every corner
    of that facet has zero barycentric weight on root corner ``k``. With the
    partition of every root and root embedding, this proves the target boundary is
    the source boundary and every other contact is a shared facet.
    """
    width = barycentric.shape[1]
    boundary = jnp.asarray(_source_boundary_faces(source_mesh))[root_rows]
    single = (counts[incidence] == 1).reshape((count, width))
    _require(
        _boundary_contact_values(boundary, single, barycentric),
        "collective.boundary_facets must lie on root boundary faces; got exterior unshared facet",
    )


# Published target material organization. A carrier without these banks has
# no published labels to check; coverage then uses only root inheritance.
_REGION_BANKS = (
    "organization/region/cell_indices",
    "organization/region/facet_cell_regions",
    "organization/region/facet_indices",
    "organization/region/facet_orientations",
)


def _region_banks(
    domain: PiecewiseLinearDomain,
    source_mesh: CellMesh,
    source_regions: np.ndarray,
    target_sources: Array,
    arrays: dict[str, Array],
    incidence: Array,
    counts: Array,
) -> None:
    source_ids = jnp.concatenate(
        tuple(jnp.asarray(block.global_ids) for block in source_mesh.blocks)
    )
    positions, present = _lookup(source_ids, target_sources)
    _require(
        present,
        "collective.organization.region.cell_indices must identify root occurrences; got absent root",
    )
    regions = _bank(arrays, "organization/region/cell_indices")[: target_sources.shape[0]]
    _require(
        regions == jnp.asarray(source_regions)[positions],
        "collective.organization.region.cell_indices must match root labels; got different assignments",
    )
    facets = counts.shape[0]
    neighbors = _bank(arrays, "organization/region/facet_cell_regions")[:facets]
    interface = _bank(arrays, "organization/region/facet_indices")[:facets]
    orientations = _bank(arrays, "organization/region/facet_orientations")[:facets]
    cross = jnp.all(neighbors >= 0, axis=1) & (neighbors[:, 0] != neighbors[:, 1])
    _require(
        cross == (interface >= 0),
        "collective.organization.region.facet_indices must identify cross-material facets; got inconsistent interfaces",
    )
    _require(
        jnp.where(cross, jnp.abs(orientations) == 1, orientations == 0),
        "collective.organization.region.facet_orientations must be signed units on interfaces and zero elsewhere; got invalid orientation",
    )
    _require(
        jnp.all((neighbors >= -1) & (neighbors < len(domain.region_ids))),
        "collective.organization.region.facet_cell_regions must contain declared labels or exterior sentinel; got unknown labels",
    )
    # Reciprocal cell incidence and inherited labels determine facet materials.
    material = jnp.repeat(regions, incidence.shape[0] // regions.shape[0])
    low = (
        jnp.full((facets,), len(domain.region_ids), dtype=jnp.int32)
        .at[incidence]
        .min(material)
    )
    high = jnp.full((facets,), -1, dtype=jnp.int32).at[incidence].max(material)
    declared_low = jnp.min(
        jnp.where(neighbors >= 0, neighbors, len(domain.region_ids)), axis=1
    )
    declared_high = jnp.max(neighbors, axis=1)
    _require(
        (low == declared_low)
        & (high == declared_high)
        & ((counts == 1) == jnp.any(neighbors < 0, axis=1)),
        "collective.organization.region.facet_cell_regions must match reciprocal incidence; got different materials or exterior status",
    )


def _collective_validity(
    mesh: CellMesh, geometry: CellGeometrySpec, policy: CellValidityPolicy, /
) -> tuple[CellValidityCertificate, bool]:
    """Return target-map validity and the all-owner verdict.

    Each partition owns disjoint logical cells. Gathered owned counts establish
    inventory coverage before the validity verdict is accepted.
    """
    storage = mesh.storage
    if storage is None:
        raise ValueError(
            "collective.mesh.storage must describe owner-local validity; got None"
        )
    if (
        storage.partition_count != jax.process_count()
        or storage.partition_index != jax.process_index()
    ):
        raise ValueError(
            "collective.mesh.storage.partition must match one partition per process; got incompatible topology"
        )
    validity = certify_cell_geometry_validity(geometry, mesh=mesh, policy=policy)
    dimension = mesh.topological_dimension
    owned = np.asarray(jax.device_get(storage.entity_owned[dimension]), dtype=np.bool_)
    local_ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
    )
    owned_ids = np.asarray(
        jax.device_get(storage.entity_global_ids[dimension]), dtype=np.int64
    )[owned]
    packet = np.asarray(
        (
            owned_ids.size,
            int(validity.all_certified and np.all(np.isin(owned_ids, local_ids))),
        ),
        dtype=np.int64,
    )
    gathered = np.asarray(
        multihost_utils.process_allgather(packet, tiled=False), dtype=np.int64
    ).reshape((-1, 2))
    if int(np.sum(gathered[:, 0])) != storage.global_entity_counts[dimension]:
        raise ValueError(
            "collective.owned_cells must partition the logical cell inventory; got differing global count"
        )
    return validity, bool(np.all(gathered[:, 1] == 1))


@checked
def certify_collective_premises(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    domain: PiecewiseLinearDomain,
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    source_embedding: GlobalEmbeddingCertificate,
    source_coverage: DomainCoverageCertificate,
    source_cell_regions: np.ndarray,
    states: AdaptiveSimplexState,
    initial_states: AdaptiveSimplexState,
    /,
    *,
    logical_arrays: tuple[tuple[str, Array], ...],
    source_partition_id: str,
    validity_policy: CellValidityPolicy | None = None,
    limits: MeshCertificateLimits | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[
    CellValidityCertificate, GlobalEmbeddingCertificate, DomainCoverageCertificate
]:
    """Certify target validity, embedding and domain coverage.

    Every owner supplies its local target and the serial affine root mesh.
    The disjoint dyadic partition and root restriction charts transport the
    root embedding and region integrals; reciprocal incidence identifies the
    boundary. Source certificates must certify the declared root domain.
    """
    if mesh.storage is None:
        raise TypeError(
            "collective.mesh.storage must describe owner-local storage; got None"
        )
    if source_mesh.storage is not None:
        raise ValueError(
            "collective.source_mesh.storage must be None for serial roots; got distributed storage"
        )
    source_embedding.binding.require(source_mesh, source_geometry)
    source_coverage.binding.require(source_mesh, source_geometry)
    if (
        source_embedding.status != "certified"
        or source_embedding.binding.junction_vertices
        or source_coverage.status != "certified"
        or source_coverage.domain_id != domain.domain_id
        or source_coverage.embedding_certificate_id != source_embedding.certificate_id
        or source_coverage.covered_source_facet_count
        != source_coverage.source_facet_count
    ):
        raise ValueError(
            "collective.source_certificates must certify embedding and full domain coverage with matching bindings; got incompatible evidence"
        )
    storage = mesh.storage
    if (
        source_partition_id != storage.evidence_id
        or cell_geometry_id(geometry) != storage.logical_coordinate_geometry_id
    ):
        raise ValueError(
            "collective.target_binding must match storage and logical geometry identities; got incompatible binding"
        )
    limits_ = MeshCertificateLimits() if limits is None else limits
    policy = CellValidityPolicy() if validity_policy is None else validity_policy
    arrays = dict(logical_arrays)
    owned = dict(storage.logical_arrays)
    if set(arrays) != set(owned) or logical_array_value_collection_digest(
        arrays
    ) != logical_array_value_collection_digest(owned):
        raise ValueError(
            "collective.logical_arrays must match owner-local storage content; got different banks"
        )
    _require(
        _raw_partition(states, initial_states),
        "collective.states must form a disjoint root partition; got inconsistent dyadic forest",
    )
    dimension = mesh.topological_dimension
    count = storage.global_entity_counts[dimension]
    target_sources, root_rows, barycentric = _geometry_banks(
        source_mesh, source_geometry, states, arrays, count
    )
    incidence, counts = _facet_incidence(mesh, arrays, count)
    _boundary_contacts(source_mesh, root_rows, barycentric, incidence, counts, count)
    regions = np.asarray(source_cell_regions, dtype=np.int64)
    if regions.shape != (sum(block.cell_count for block in source_mesh.blocks),):
        raise ValueError(
            "collective.source_cell_regions must cover every root cell; got incompatible shape"
        )
    region_banks = tuple(name for name in _REGION_BANKS if name in arrays)
    if region_banks == _REGION_BANKS:
        _region_banks(
            domain, source_mesh, regions, target_sources, arrays, incidence, counts
        )
    elif region_banks:
        raise ValueError(
            "collective.organization.region must publish all region banks; got incomplete bank set"
        )
    validity, valid = _collective_validity(mesh, geometry, policy)
    embedding_state = EmbeddingState(
        [],
        [
            "cell_validity",
            "collective_cell_validity",
            "source_embedding_premise",
            "complete_dyadic_source_partition",
            "actual_source_geometry_banks",
            "facet_pairing",
            "boundary_contact",
        ],
    )
    if not valid:
        embedding_state.add("collective_cell_validity", "unresolved", "mesh")
    # The boundary-contact proof makes the target boundary the source boundary
    # as a point set, so the subdivision preserves the source shell structure.
    embedding = GlobalEmbeddingCertificate(
        MeshCertificateBinding(mesh, geometry, "mapped", limits_),
        validity.certificate_id,
        tuple(embedding_state.findings),
        tuple(embedding_state.checks),
        counts=EmbeddingCounts(
            cell_count=count,
            boundary_facet_count=int(
                jax.device_get(jnp.sum(counts == 1, dtype=jnp.int64))
            ),
            shell_count=source_embedding.shell_count,
            candidate_pair_count=0,
            ray_test_count=0,
            subdivision_piece_count=0,
        ),
    )
    binding = MeshCertificateBinding(
        mesh,
        geometry,
        "mapped",
        limits_,
        source_id=domain.source_id,
        source_revision=domain.source_revision,
    )
    state = EmbeddingState(
        [],
        [
            "collective_domain_coverage",
            "complete_dyadic_source_partition",
            "actual_source_geometry_banks",
            *(("global_material_interface_incidence",) if region_banks else ()),
        ],
    )
    if embedding.status != "certified":
        state.add("embedding_premise", "unresolved", "mesh")
    # Integrate affine roots independently, then transport through their disjoint
    # dyadic partition to target regions under the request's resource limits.

    with coordinate_certificate_request(limits_, budget=budget) as budget:
        try:
            with budget.measurement(), budget.activate():
                measured, _ = mapped_coverage_measures(
                    state, source_mesh, source_geometry, regions, budget=budget
                )
            achieved = (
                *measured,
                *(Fraction(0) for _ in range(len(domain.region_ids) - len(measured))),
            )
        except coordinate_budget_owner.CoordinateEnclosureResourceError:
            state.add("mapped_coverage_resource_budget", "unresolved", "mesh")
            achieved = tuple(None for _ in domain.region_ids)
        coverage = coverage_result(
            binding,
            embedding,
            domain,
            state,
            declared_coverage_measures(domain),
            achieved,
            source_coverage.source_facet_count,
            {},
            (source_coverage.certificate_id, source_partition_id),
            expression_work=active_expression_work(budget=budget),
        )
        return validity, embedding, coverage
