#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Target premises from actual sharded dyadic restriction witnesses.

The caller restores and validates the neutral numerical epoch before publication.
This owner independently checks the root partition, exact coefficient/chart
banks, reciprocal facet incidence, source-boundary contact and authoritative
region/interface ancestry, and certifies the actual target maps with the
canonical validity owner on every partition. No accepted target result,
caller-supplied positive flags or materialized global mesh is consumed.
"""

from __future__ import annotations

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental import multihost_utils

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
from ..discretization._coordinate_enclosure import coordinate_source_signature
from ._mesh_certificates import (
    _coverage_result,
    _declared_coverage_measures,
    _EmbeddingState,
    _mapped_coverage_measures,
    DomainCoverageCertificate,
    GlobalEmbeddingCertificate,
    MeshCertificateBinding,
    MeshCertificateLimits,
    PiecewiseLinearDomain,
)


def _require(predicate: Array, message: str) -> None:
    if not bool(jax.device_get(jnp.all(predicate))):
        raise ValueError(message)


def _bank(arrays: dict[str, Array], name: str) -> Array:
    value = arrays.get(name)
    if value is None:
        raise ValueError(
            f"Collective proof banks lack the required logical array {name!r}."
        )
    return value


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
    """Complete living binary families, not a sum-only volume witness.

    Child vertex sets are exactly the two halves of the parent split edge.
    Parent/child reciprocity and strictly increasing allocation imply a forest;
    every living parent is either an active leaf or has both complete halves.
    Coarsened history is allowed only when both historical children are retired.
    """
    if states.mesh.cells.ndim != 3 or initial.mesh.cells.shape != states.mesh.cells.shape:
        raise ValueError(
            "Collective partitions require aligned partition-leading simplex arrays."
        )
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
    """Exact dyadic support on immutable root vertices for each numerical slot."""
    parts, vertices = states.mesh.vertex_ids.shape
    width = states.mesh.cells.shape[2]
    sentinel = jnp.iinfo(jnp.int64).max
    identifiers = jnp.full((parts, vertices, width), -1, dtype=jnp.int64)
    weights = jnp.zeros((parts, vertices, width), dtype=jnp.float64)
    accepted = jnp.asarray(True)

    def step(slot: int, carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
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

    identifiers, weights, accepted = jax.lax.fori_loop(
        0, vertices, step, (identifiers, weights, accepted)
    )
    _require(accepted, "Collective vertex ancestry is not an ordered dyadic source DAG.")
    return identifiers, weights


def _global_epoch_banks(
    states: AdaptiveSimplexState, arrays: dict[str, Array], count: int
) -> tuple[Array, Array]:
    """Root-vertex support and dyadic weights of every active target corner."""
    ids = states.mesh.cell_ids.reshape(-1)
    active = states.mesh.cell_active.reshape(-1)
    order = jnp.argsort(jnp.where(active, ids, jnp.iinfo(jnp.int64).max), stable=True)[
        :count
    ]
    selected_ids = ids[order]
    _require(
        jnp.sum(active, dtype=jnp.int64) == count,
        "Logical target cell count does not match the actual active forest.",
    )
    _require(
        selected_ids[1:] > selected_ids[:-1],
        "Actual collective target cell identities overlap.",
    )
    _require(
        selected_ids == _bank(arrays, "cell_global_ids")[:count],
        "Logical target cell IDs differ from actual active epoch cells.",
    )
    part_count, _, width = states.mesh.cells.shape
    global_vertices = jnp.take_along_axis(
        states.mesh.vertex_ids, states.mesh.cells.reshape((part_count, -1)), axis=1
    ).reshape((-1, width))[order]
    _require(
        global_vertices == _bank(arrays, "cell_vertices")[:count],
        "Logical cell corner identities differ from the numerical forest.",
    )
    source_ids, weights = _ancestry(states)
    # Gather per partition, flatten the partition-leading slot axis, then select
    # the active cells in canonical ID order.
    slots = (ids.shape[0], width, width)
    gathered_ids = jax.vmap(lambda source, rows: source[rows])(
        source_ids, states.mesh.cells
    ).reshape(slots)[order]
    gathered_weights = jax.vmap(lambda source, rows: source[rows])(
        weights, states.mesh.cells
    ).reshape(slots)[order]
    return gathered_ids, gathered_weights


def _exact_action_chart(actions: np.ndarray, width: int, /) -> tuple[Fraction, ...]:
    """Exact root barycentrics of a composed coefficient action stack.

    Stack row 0 is the outermost action, so target corners are the rows of
    ``actions[0] @ ... @ actions[-1]``, as the serial
    ``BarycentricCellGeometryElement`` tabulation composes them. Each binary64
    weight is its own exact rational; an empty stack is the root identity.
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
    """Exact equality of witnessed action charts and the actual dyadic ancestry.

    Every owner gathers the complete global banks, so each decides the same
    all-cell verdict.
    """
    host_counts, host_actions, host_valid, expected = (
        np.asarray(value)
        for value in multihost_utils.process_allgather(
            (counts, actions, valid, barycentric), tiled=True
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
            "Collective coefficient action stacks lost their source columns or declared depth."
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
                "Collective coefficient actions differ from the actual dyadic ancestry."
            )
        decided.add(key)


def _geometry_banks(
    source_mesh: CellMesh,
    source_geometry: CellGeometrySpec,
    states: AdaptiveSimplexState,
    arrays: dict[str, Array],
    count: int,
) -> tuple[Array, Array, Array]:
    """Exact dyadic root charts: target roots, root rows and corner barycentrics."""
    elements, routes, coefficients = source_geometry.resolve(source_mesh)
    _require(
        _bank(arrays, "geometry/source_geometry_id")
        == _source_identity(cell_geometry_id(source_geometry)),
        "Collective source geometry identity is not its actual original map.",
    )
    _require(
        _bank(arrays, "geometry/source_topology_id")
        == _source_identity(source_mesh.topology_id),
        "Collective source topology identity is not its actual original topology.",
    )
    coordinate_count = coefficients.shape[0]
    _require(
        _bank(arrays, "geometry/coordinate_ids")[:coordinate_count]
        == jnp.arange(coordinate_count, dtype=jnp.int64),
        "Scientific coordinate IDs lost their original coefficient namespace.",
    )
    _require(
        _bank(arrays, "geometry/coordinates")[:coordinate_count] == coefficients,
        "Collective target maps do not retain their actual original coefficients.",
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
    _require(root_present, "A collective target cell has no authoritative original root.")
    root_vertices = source_corners[root_rows]
    barycentric = jnp.sum(
        jnp.where(
            source_vertex_ids[:, :, :, None] == root_vertices[:, None, None, :],
            source_weights[:, :, :, None],
            0.0,
        ),
        axis=2,
    )
    _require(
        jnp.sum(barycentric, axis=2) == 1.0,
        "A target chart has source support outside its declared root.",
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
                "Collective inheritance requires the actual canonical affine simplex source."
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
            ~valid
            | (present & block_present & (target_sources[positions] == parent_ids)),
            "A collective target cell has no authoritative original root.",
        )
        _require(
            ~valid[:, None]
            | (
                _bank(arrays, f"geometry/parent_vertex_ids/{block.name}")
                == root_vertices[positions]
            ),
            "A collective target changed the ordered scientific root corner identities.",
        )
        _require(
            ~valid[:, None]
            | (
                _bank(arrays, f"geometry/routes/{block.name}")
                == jnp.asarray(route)[block_rows]
            ),
            "A collective target changed its original scientific coefficient route.",
        )
        digest = canonical_fingerprint(
            {
                "signature": coordinate_source_signature(element),
                "arrays": array_tree_fingerprint(element),
            }
        )
        basis = jnp.asarray(np.frombuffer(bytes.fromhex(digest), dtype=np.uint8).copy())
        _require(
            ~valid[:, None]
            | (_bank(arrays, f"geometry/source_basis/{block.name}") == basis),
            "A collective target changed its actual source basis definition.",
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
        "Every logical target cell must occur exactly once in the actual source map banks.",
    )
    return target_sources, root_rows, barycentric


def _facet_incidence(
    mesh: CellMesh, arrays: dict[str, Array], count: int
) -> tuple[Array, Array]:
    """Logical facet row of every (cell, opposite corner) and facet multiplicities.

    Facet incidence is recomputed from actual logical cell vertices; scientific
    facet IDs are matched by lexicographic vertex records, not row positions.
    """
    storage = mesh.storage
    if storage is None:
        raise ValueError("Collective facet incidence requires canonical mesh storage.")
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
    packet = jnp.sort(cell_vertices[:, columns], axis=2).reshape((-1, dimension))
    keys = _bank(arrays, f"entity_vertices_{dimension - 1}")[:facets]
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
    _require(
        incidence >= 0,
        "Actual target facets are absent from the logical facet inventory.",
    )
    counts = jnp.zeros((facets,), dtype=jnp.int32).at[incidence].add(1)
    _require(
        (counts == 1) | (counts == 2),
        "Global reciprocal facet multiplicity is not one or two.",
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
    complete partition and root embedding, this proves the target boundary is
    the source boundary and every other contact is a shared facet.
    """
    width = barycentric.shape[1]
    boundary = jnp.asarray(_source_boundary_faces(source_mesh))[root_rows]
    single = (counts[incidence] == 1).reshape((count, width))
    others = jnp.asarray(~np.eye(width, dtype=np.bool_))[None, :, :, None]
    on_face = jnp.all((barycentric == 0.0)[:, None, :, :] | ~others, axis=2)
    exterior = jnp.any(on_face & boundary[:, None, :], axis=2)
    _require(
        ~single | exterior, "An unshared target facet is not part of the source boundary."
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
    _require(present, "Target material labels have no actual original cell occurrence.")
    regions = _bank(arrays, "organization/region/cell_indices")[: target_sources.shape[0]]
    _require(
        regions == jnp.asarray(source_regions)[positions],
        "Target material assignments differ from their scientific source roots.",
    )
    facets = counts.shape[0]
    neighbors = _bank(arrays, "organization/region/facet_cell_regions")[:facets]
    interface = _bank(arrays, "organization/region/facet_indices")[:facets]
    orientations = _bank(arrays, "organization/region/facet_orientations")[:facets]
    cross = jnp.all(neighbors >= 0, axis=1) & (neighbors[:, 0] != neighbors[:, 1])
    _require(
        cross == (interface >= 0),
        "Global material interfaces are missing or falsely created.",
    )
    _require(
        jnp.where(cross, jnp.abs(orientations) == 1, orientations == 0),
        "Global material interface orientation is absent.",
    )
    _require(
        jnp.all((neighbors >= -1) & (neighbors < len(domain.region_ids))),
        "Global facet-region inventory has unknown source material labels.",
    )
    # Facet materials are checked against actual reciprocal cell incidence and
    # inherited whole-cell labels, not accepted from interface flags.
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
        "Facet materials or exterior status differ from actual reciprocal cell incidence.",
    )


def _collective_validity(
    mesh: CellMesh, geometry: CellGeometrySpec, policy: CellValidityPolicy, /
) -> tuple[CellValidityCertificate, bool]:
    """Canonical validity of this owner's actual target maps and the all-owner verdict.

    Every logical cell is owned by exactly one partition of the bound storage
    evidence; one partition per process makes the gathered owned counts and
    verdicts a complete all-owner decision.
    """
    storage = mesh.storage
    if storage is None:
        raise ValueError("Collective validity requires canonical mesh storage.")
    if (
        storage.partition_count != jax.process_count()
        or storage.partition_index != jax.process_index()
    ):
        raise ValueError(
            "Collective validity requires exactly one owner partition per process."
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
            "Owned target cells do not partition the logical cell inventory."
        )
    return validity, bool(np.all(gathered[:, 1] == 1))


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
) -> tuple[
    CellValidityCertificate, GlobalEmbeddingCertificate, DomainCoverageCertificate
]:
    """Create new target-bound validity, global embedding and domain coverage.

    Collective: every owner calls this with its owner-local target. The source
    is the original serial affine root mesh with its certified embedding and
    coverage; ``source_cell_regions`` are its coverage request's region indices.
    The embedding theorem composes the root embedding with the complete
    disjoint dyadic partition, exact root restriction charts, reciprocal facet
    incidence and source-boundary contact; validity is the canonical owner's
    certificate of the actual target maps on every owner. Coverage transports
    exact root integrals through the same partition. Bank failures refuse.
    """
    if (
        not isinstance(mesh, CellMesh)
        or mesh.storage is None
        or not isinstance(geometry, CellGeometrySpec)
    ):
        raise TypeError(
            "Collective premises require an owner-local target mesh and restored geometry."
        )
    if (
        not isinstance(domain, PiecewiseLinearDomain)
        or not isinstance(source_coverage, DomainCoverageCertificate)
        or not isinstance(source_embedding, GlobalEmbeddingCertificate)
    ):
        raise TypeError(
            "Collective premises require the original declared domain and actual source embedding and coverage."
        )
    if not isinstance(states, AdaptiveSimplexState) or not isinstance(
        initial_states, AdaptiveSimplexState
    ):
        raise TypeError(
            "Collective premises require actual neutral adaptive simplex states."
        )
    if source_mesh.storage is not None:
        raise ValueError(
            "Collective premises require the original serial source root mesh."
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
            "Collective premises require certified embedding and complete coverage of the actual original source domain."
        )
    storage = mesh.storage
    if (
        source_partition_id != storage.evidence_id
        or cell_geometry_id(geometry) != storage.logical_coordinate_geometry_id
    ):
        raise ValueError(
            "Collective premises must bind the actual target logical coordinate map, not its sampled carrier."
        )
    limits_ = MeshCertificateLimits() if limits is None else limits
    if not isinstance(limits_, MeshCertificateLimits):
        raise TypeError("limits must be MeshCertificateLimits or None.")
    policy = CellValidityPolicy() if validity_policy is None else validity_policy
    if not isinstance(policy, CellValidityPolicy):
        raise TypeError("validity_policy must be CellValidityPolicy or None.")
    arrays = dict(logical_arrays)
    owned = dict(storage.logical_arrays)
    if set(arrays) != set(owned) or logical_array_value_collection_digest(
        arrays
    ) != logical_array_value_collection_digest(owned):
        raise ValueError("Collective proof banks differ from actual owner-local storage.")
    _require(
        _raw_partition(states, initial_states),
        "The actual dyadic forest does not give a complete disjoint source partition.",
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
            "Original cell-region assignments must cover every authoritative source cell."
        )
    region_banks = tuple(name for name in _REGION_BANKS if name in arrays)
    if region_banks == _REGION_BANKS:
        _region_banks(
            domain, source_mesh, regions, target_sources, arrays, incidence, counts
        )
    elif region_banks:
        raise ValueError(
            "Collective proof banks publish an incomplete material organization."
        )
    validity, valid = _collective_validity(mesh, geometry, policy)
    embedding_state = _EmbeddingState(
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
        cell_count=count,
        boundary_facet_count=int(jax.device_get(jnp.sum(counts == 1))),
        shell_count=source_embedding.shell_count,
        candidate_pair_count=0,
        ray_test_count=0,
        subdivision_piece_count=0,
    )
    binding = MeshCertificateBinding(
        mesh,
        geometry,
        "mapped",
        limits_,
        source_id=domain.source_id,
        source_revision=domain.source_revision,
    )
    state = _EmbeddingState(
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
    # Exact affine source integrals are integrated independently over each
    # original root. Complete disjoint dyadic partitions transport these root
    # integrals to target regions; no old achieved report is copied.
    # The request's coordinate work and scratch limits own these root integrals.
    from ..discretization._coordinate_enclosure import (
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
    )

    budget = CoordinateEnclosureBudget(
        limits_.maximum_work_units, limits_.maximum_scratch_bytes
    )
    try:
        with budget.activate():
            measured, _ = _mapped_coverage_measures(
                state, source_mesh, source_geometry, regions
            )
        achieved = (
            *measured,
            *(Fraction(0) for _ in range(len(domain.region_ids) - len(measured))),
        )
    except CoordinateEnclosureResourceError:
        state.add("mapped_coverage_resource_budget", "unresolved", "mesh")
        achieved = tuple(None for _ in domain.region_ids)
    coverage = _coverage_result(
        binding,
        embedding,
        domain,
        state,
        _declared_coverage_measures(domain),
        achieved,
        source_coverage.source_facet_count,
        {},
        (source_coverage.certificate_id, source_partition_id),
        expression_work=(
            budget.work_units,
            budget.peak_bytes_upper,
            budget.required_work_units,
        ),
    )
    return validity, embedding, coverage
