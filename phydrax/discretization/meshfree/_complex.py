#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Geometry-authorized meshfree realization of every exterior degree.

An authoritative oriented simplicial ``CellMesh`` plus a declared domain
identity supplies incidence, orientation, measures and the de Rham moment map.
Generalized moving least squares reconstructs local polynomial k-forms from
k-cell moments; the reconstructions define sparse consistency-plus-stabilization
Hodges that native sparse Cholesky must admit. The native
``CochainDiscretization`` owns d, codifferentials, relative boundaries and
Hilbert complexes; ``phydrax.exterior`` owns integration, traces and cohomology.

The bounded radius-clique route is abstract research only. Matching Betti
numbers never turn it into a continuum-fidelity realization.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from itertools import combinations
from math import ceil, comb, factorial, isfinite, prod
from typing import final, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._polynomial._cubature import simplex_rule_data
from ..._strict import StrictModule
from ..._validation import canonical_identifier, nonnegative_integer, positive_integer
from ...exterior._complex import DiscreteForm
from ...exterior._de_rham import DeRhamBridge, simplicial_parameterizations
from ...linalg import compound_matrix
from ...metrix import CoordinateChart
from ...topology import (
    BettiDimensionResult,
    compute_betti_dimensions,
    ExactChainComplex,
    ExactIntegerCOO,
    RationalField,
    TopologyResourcePolicy,
)
from ...typing import Bool, Dim, Float64, Int32, parse
from .._boundary_complex import boundary_subcomplex
from .._cell_complex import simplicial_cell_complex, simplicial_cell_geometry
from .._cell_mesh import CellMesh
from .._cochain import CochainDiscretization
from .._cochain_hodge import SparseHodge
from .._cochain_orientation import reorient_cell_complex
from .._topology import CellComplexTopology
from ._neighbors import MeshfreeEdgeRelationPlan
from ._stencils import polynomial_design, weighted_svd_factors


class _VertexDim(Dim):
    """Authoritative complex vertices."""


class _AmbientDim(Dim):
    """Ambient embedding coordinates."""


class _TopCellDim(Dim):
    """Top-degree cells of the authoritative complex."""


class _PatchDim(Dim):
    """Padded local k-cell moment slots of one top-cell patch."""


class _NodePatchDim(Dim):
    """Padded meshfree sample nodes of one top-cell patch."""


class _FaceDim(Dim):
    """k-faces of one top simplex."""


class _FeatureDim(Dim):
    """Local polynomial k-form features."""


class _ScalarFeatureDim(Dim):
    """Local scalar polynomial features."""


class _CellDim(Dim):
    """k-cells of one degree."""


class _NodeDim(Dim):
    """Meshfree sample nodes."""


class _CellNodeDim(Dim):
    """Meshfree sample nodes owned by one top cell."""


class _FrameDim(Dim):
    """Intrinsic frame axes of one top cell."""


ComplexGeometryRole: TypeAlias = Literal["domain", "closed_surface", "open_surface"]
"""Declared embedding role: full-dimensional domain or codimension-one sheet."""

ComplexFidelity: TypeAlias = Literal["geometry-authorized", "abstract-research"]
"""Consumer-visible separation of authorized and abstract complex records."""


class MeshfreeComplexAdmissionError(ValueError):
    """A complex, its identity, or its meshfree realization fails admission."""


def _finite_float(value: float, name: str, /, *, minimum: float) -> float:
    result = float(value)
    if not isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}.")
    return result


@final
class ComplexDomainIdentity(StrictModule):
    """Declared scientific identity of the region a complex must represent.

    ``measure`` is the n-measure of the represented region and ``betti`` its
    absolute rational Betti numbers ``b_0..b_n``. Both are compared with the
    supplied complex before any realization is prepared; a triangulation that
    fills a hole or omits a handle is refused rather than relabelled.
    """

    domain_id: str = eqx.field(static=True)
    role: ComplexGeometryRole = eqx.field(static=True)
    measure: float = eqx.field(static=True)
    betti: tuple[int, ...] = eqx.field(static=True)
    measure_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        domain_id: str,
        role: ComplexGeometryRole,
        /,
        *,
        measure: float,
        betti: Sequence[int],
        measure_tolerance: float = 1e-2,
    ) -> None:
        identifier = canonical_identifier(domain_id, "domain_id")
        role_ = parse(role, ComplexGeometryRole, "role")
        measure_ = _finite_float(measure, "measure", minimum=0.0)
        if measure_ <= 0.0:
            raise ValueError("measure must be positive.")
        numbers = tuple(nonnegative_integer(value, "betti") for value in betti)
        if not numbers:
            raise ValueError("betti must list b_0 through b_n.")
        tolerance = _finite_float(measure_tolerance, "measure_tolerance", minimum=0.0)
        self.domain_id = identifier
        self.role = role_
        self.measure = measure_
        self.betti = numbers
        self.measure_tolerance = tolerance


def _lookup_rows(table: np.ndarray, queries: np.ndarray, /) -> np.ndarray:
    """Return the table row index of every query row; absent rows are refused."""
    _, groups = np.unique(np.concatenate((table, queries)), axis=0, return_inverse=True)
    groups = groups.reshape(-1)
    lookup = np.full((int(groups.max(initial=-1)) + 1,), -1, dtype=np.int32)
    lookup[groups[: table.shape[0]]] = np.arange(table.shape[0], dtype=np.int32)
    found = lookup[groups[table.shape[0] :]]
    if np.any(found < 0):
        raise MeshfreeComplexAdmissionError(
            "Every face of a top simplex must be a cell of the supplied complex."
        )
    return found


def _top_faces(rows: tuple[np.ndarray, ...], /) -> tuple[np.ndarray, ...]:
    """Index every k-face of every top simplex, in lexicographic local order."""
    top = rows[-1]
    dimension = top.shape[1] - 1
    faces: list[np.ndarray] = []
    for degree in range(dimension + 1):
        local = np.asarray(
            tuple(combinations(range(dimension + 1), degree + 1)), dtype=np.int32
        )
        queries = top[:, local].reshape((-1, degree + 1))
        faces.append(
            _lookup_rows(rows[degree], queries).reshape((top.shape[0], local.shape[0]))
        )
    return tuple(faces)


def _compound(matrix: Array, degree: int, /) -> Array:
    """Native lexicographic minors with the exact entry budget of this batch.

    The admitting resource bound is the patch workset capacity checked before
    any batched local geometry is formed.
    """
    rows, columns = matrix.shape[-2:]
    entries = (
        prod(matrix.shape[:-2])
        * comb(rows, degree)
        * comb(columns, degree)
        * max(1, degree * degree)
    )
    return compound_matrix(matrix, degree, maximum_minor_entries=max(1, entries))


def _simplex_jacobians(vertices: Array, cells: Array, /) -> tuple[Array, Array]:
    origin = vertices[cells[:, 0]]
    edges = vertices[cells[:, 1:]] - origin[:, None, :]
    return origin, jnp.swapaxes(edges, 1, 2)


@jax.jit
def _simplex_measures(vertices: Array, cells: Array, /) -> Array:
    """Return k-volumes sqrt(det(J^T J))/k! through the native compound owner."""
    degree = cells.shape[1] - 1
    _, jacobian = _simplex_jacobians(vertices, cells)
    determinant = _compound(jnp.swapaxes(jacobian, 1, 2) @ jacobian, degree)[:, 0, 0]
    return jnp.sqrt(jnp.maximum(determinant, 0.0)) / factorial(degree)


def _facet_cofaces(topology: CellComplexTopology, /) -> tuple[np.ndarray, np.ndarray]:
    """Return top-cell cofaces and incidence signs of every facet (-1 padded)."""
    incidence = topology.incidences[-1]
    valid = np.asarray(incidence.relation.valid, dtype=np.bool_)
    facets = np.asarray(incidence.relation.source_indices, dtype=np.int32)[valid]
    tops = np.asarray(incidence.relation.target_indices, dtype=np.int32)[valid]
    signs = np.asarray(incidence.signs, dtype=np.float64)[valid]
    count = topology.entities(topology.dimension - 1).count
    multiplicity = np.bincount(facets, minlength=count)
    if np.any(multiplicity > 2):
        raise MeshfreeComplexAdmissionError(
            "A facet bounds more than two top cells; the complex is not a manifold."
        )
    order = np.lexsort((tops, facets))
    cofaces = np.full((count, 2), -1, dtype=np.int32)
    coface_signs = np.zeros((count, 2), dtype=np.float64)
    first = np.searchsorted(facets[order], np.arange(count, dtype=np.int32))
    slot = np.arange(order.size, dtype=np.int32) - first[facets[order]]
    cofaces[facets[order], slot] = tops[order]
    coface_signs[facets[order], slot] = signs[order]
    return cofaces, coface_signs


def _validate_purity(rows: tuple[np.ndarray, ...], faces: tuple[np.ndarray, ...]) -> None:
    for degree, (cells, owned) in enumerate(zip(rows, faces, strict=True)):
        covered = np.zeros((cells.shape[0],), dtype=np.bool_)
        covered[owned.reshape(-1)] = True
        if not np.all(covered):
            raise MeshfreeComplexAdmissionError(
                f"Every degree-{degree} cell must be a face of a top cell; the complex is not pure."
            )


def _validate_embedding(
    role: ComplexGeometryRole, dimension: int, ambient: int, /
) -> None:
    match role:
        case "domain":
            admitted = ambient == dimension and 1 <= dimension <= 3
        case "closed_surface" | "open_surface":
            admitted = ambient == dimension + 1 and 1 <= dimension <= 2
        case _:
            raise ValueError("Invalid complex geometry role.")
    if not admitted:
        raise MeshfreeComplexAdmissionError(
            f"Role {role!r} is not supported for a {dimension}-complex in {ambient}-D space."
        )


def _validate_orientation(
    role: ComplexGeometryRole,
    cofaces: np.ndarray,
    coface_signs: np.ndarray,
    vertices: Array,
    top: np.ndarray,
    top_signs: np.ndarray,
    /,
) -> None:
    counts = np.sum(cofaces >= 0, axis=1)
    if role == "closed_surface" and np.any(counts != 2):
        raise MeshfreeComplexAdmissionError(
            "A closed surface requires exactly two top cells on every facet."
        )
    interior = counts == 2
    if np.any(coface_signs[interior, 0] * coface_signs[interior, 1] != -1.0):
        raise MeshfreeComplexAdmissionError(
            "Top-cell orientations are incoherent across an interior facet."
        )
    if role == "domain":
        _, jacobian = _simplex_jacobians(vertices, jnp.asarray(top))
        determinant = np.asarray(_compound(jacobian, jacobian.shape[1])[:, 0, 0])
        ambient_sign = np.sign(determinant) * top_signs
        if not (np.all(ambient_sign > 0) or np.all(ambient_sign < 0)):
            raise MeshfreeComplexAdmissionError(
                "Domain top cells are inverted or folded relative to the ambient orientation."
            )


def _validate_identity(
    identity: ComplexDomainIdentity,
    topology: CellComplexTopology,
    measures: tuple[Array, ...],
    resources: TopologyResourcePolicy | None,
    /,
) -> tuple[float, BettiDimensionResult]:
    measure = float(np.sum(np.asarray(measures[-1])))
    if abs(measure - identity.measure) > identity.measure_tolerance * identity.measure:
        raise MeshfreeComplexAdmissionError(
            f"Complex measure {measure:.6g} does not represent declared domain "
            f"{identity.domain_id!r} of measure {identity.measure:.6g}."
        )
    if len(identity.betti) != topology.dimension + 1:
        raise MeshfreeComplexAdmissionError(
            "Declared Betti numbers must cover degrees 0 through the complex dimension."
        )
    betti = compute_betti_dimensions(
        topology, coefficients=RationalField(), resources=resources
    )
    observed = tuple(betti.dimension(degree) for degree in range(topology.dimension + 1))
    if observed != identity.betti:
        raise MeshfreeComplexAdmissionError(
            f"Complex Betti numbers {observed} do not represent declared domain "
            f"{identity.domain_id!r} with Betti numbers {identity.betti}."
        )
    return measure, betti


@final
class ComplexGeometryAuthority(StrictModule):
    """Admitted oriented simplicial complex, embedding, and declared identity.

    Only a ``CellMesh`` carries vertex geometry and provenance; abstract
    topology or point-cloud complexes are refused. ``orientation`` optionally
    reorients positive-degree cells through the native cochain owner. Admission
    requires a pure simplicial complex, nondegenerate cells, at most two top
    cells per facet with coherent orientation, a uniformly oriented domain
    embedding, and agreement with the declared measure and Betti numbers.
    """

    __strict_contract__ = True
    topology: CellComplexTopology
    vertices: Float64[_VertexDim, _AmbientDim]
    identity: ComplexDomainIdentity
    measures: tuple[Array, ...]
    boundary_facets: Bool[_CellDim]
    betti: BettiDimensionResult
    measure: float = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    authority_id: str = eqx.field(static=True)

    def __init__(
        self,
        mesh: CellMesh,
        identity: ComplexDomainIdentity,
        /,
        *,
        orientation: Sequence[ArrayLike] | None = None,
        resources: TopologyResourcePolicy | None = None,
    ) -> None:
        if not isinstance(mesh, CellMesh):
            raise TypeError(
                "Geometry authority requires a CellMesh; abstract or point-cloud complexes carry no geometry."
            )
        if not isinstance(identity, ComplexDomainIdentity):
            raise TypeError("identity must be a ComplexDomainIdentity.")
        topology = (
            mesh.topology
            if orientation is None
            else reorient_cell_complex(mesh.topology, orientation)
        )
        dimension = topology.dimension
        _validate_embedding(identity.role, dimension, mesh.ambient_dimension)
        rows, signs = simplicial_cell_geometry(topology)
        faces = _top_faces(rows)
        _validate_purity(rows, faces)
        vertices = jnp.asarray(mesh.coordinates, dtype=jnp.float64)
        measures = tuple(
            _simplex_measures(vertices, jnp.asarray(cells)) for cells in rows
        )
        scale = float(np.max(np.asarray(measures[1])))
        for degree, measure in enumerate(measures[1:], start=1):
            if bool(np.any(np.asarray(measure) <= 1e-12 * scale**degree)):
                raise MeshfreeComplexAdmissionError(
                    f"The complex contains a degenerate degree-{degree} simplex."
                )
        cofaces, coface_signs = _facet_cofaces(topology)
        _validate_orientation(
            identity.role, cofaces, coface_signs, vertices, rows[-1], signs[-1]
        )
        total, betti = _validate_identity(identity, topology, measures, resources)
        authority_id = canonical_fingerprint(
            {
                "kind": "meshfree-complex-authority",
                "mesh": mesh.mesh_id,
                "topology": topology.topology_id,
                "domain": identity.domain_id,
                "role": identity.role,
                "betti": list(identity.betti),
            }
        )
        self.topology = topology
        self.vertices = vertices
        self.identity = identity
        self.measures = measures
        self.boundary_facets = jnp.asarray(np.sum(cofaces >= 0, axis=1) == 1)
        self.betti = betti
        self.measure = total
        self.source_id = mesh.mesh_id
        self.authority_id = authority_id


@final
class MeshfreeComplexPolicy(StrictModule):
    """Local reconstruction degree, support growth, capacities, and stabilization.

    ``polynomial_degree`` m selects full P_m Λ^k local reconstruction in every
    degree. Patches grow by facet adjacency until they hold ``oversampling``
    times the feature count and the native weighted SVD reports full rank with
    condition below ``maximum_condition``. ``stabilization`` scales the
    moment-residual term that makes the Hodge positive; zero keeps only the
    consistency term and therefore relies entirely on native admission.
    """

    polynomial_degree: int = eqx.field(static=True)
    oversampling: float = eqx.field(static=True)
    stabilization: float = eqx.field(static=True)
    quadrature_order: int = eqx.field(static=True)
    maximum_patch_cells: int = eqx.field(static=True)
    maximum_patch_entities: int = eqx.field(static=True)
    maximum_workset_entries: int = eqx.field(static=True)
    maximum_condition: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        polynomial_degree: int = 1,
        oversampling: float = 1.5,
        stabilization: float = 1.0,
        quadrature_order: int = 6,
        maximum_patch_cells: int = 64,
        maximum_patch_entities: int = 256,
        maximum_workset_entries: int = 1 << 24,
        maximum_condition: float = 1e10,
    ) -> None:
        degree = nonnegative_integer(polynomial_degree, "polynomial_degree")
        over = _finite_float(oversampling, "oversampling", minimum=1.0)
        stabilization_ = _finite_float(stabilization, "stabilization", minimum=0.0)
        order = nonnegative_integer(quadrature_order, "quadrature_order")
        cells = positive_integer(maximum_patch_cells, "maximum_patch_cells")
        entities = positive_integer(maximum_patch_entities, "maximum_patch_entities")
        workset = positive_integer(maximum_workset_entries, "maximum_workset_entries")
        condition = _finite_float(maximum_condition, "maximum_condition", minimum=1.0)
        self.polynomial_degree = degree
        self.oversampling = over
        self.stabilization = stabilization_
        self.quadrature_order = order
        self.maximum_patch_cells = cells
        self.maximum_patch_entities = entities
        self.maximum_workset_entries = workset
        self.maximum_condition = condition
        self.policy_id = canonical_fingerprint(
            {
                "kind": "meshfree-complex-policy",
                "polynomial_degree": degree,
                "oversampling": over,
                "stabilization": stabilization_,
                "quadrature_order": order,
                "maximum_patch_cells": cells,
                "maximum_patch_entities": entities,
                "maximum_workset_entries": workset,
                "maximum_condition": condition,
            }
        )


@final
class MeshfreeComplexDegreeEvidence(StrictModule):
    """Unisolvency, polynomial reproduction, and native Hodge admission per degree."""

    rank_deficient_patches: Array
    maximum_condition: Array
    minimum_singular_value: Array
    reproduction_residual: Array
    hodge_admitted: Array
    degree: int = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)
    patch_capacity: int = eqx.field(static=True)
    hodge_routes: int = eqx.field(static=True)


@final
class MeshfreeComplexEvidence(StrictModule):
    """Geometry-authorized realization record; never an abstract-clique record."""

    degrees: tuple[MeshfreeComplexDegreeEvidence, ...]
    sample_maximum_condition: Array
    measure: float = eqx.field(static=True)
    betti: tuple[int, ...] = eqx.field(static=True)
    boundary_facet_count: int = eqx.field(static=True)
    authority_id: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    role: ComplexGeometryRole = eqx.field(static=True)
    fidelity: ComplexFidelity = eqx.field(static=True)

    @property
    def admitted(self) -> Array:
        return jnp.all(
            jnp.stack(
                tuple(
                    value.hodge_admitted & (value.rank_deficient_patches == 0)
                    for value in self.degrees
                )
            )
        )


@final
class _TopFrames(StrictModule):
    """Orthonormal intrinsic frames, centers and diameters of top simplices."""

    __strict_contract__ = True
    centers: Float64[_TopCellDim, _AmbientDim]
    axes: Float64[_TopCellDim, _AmbientDim, _FrameDim]
    scales: Float64[_TopCellDim]

    def __init__(self, centers: Array, axes: Array, scales: Array, /) -> None:
        self.centers = centers
        self.axes = axes
        self.scales = scales

    def take(self, indices: Array, /) -> _TopFrames:
        return _TopFrames(self.centers[indices], self.axes[indices], self.scales[indices])

    def local(self, points: Array, /) -> Array:
        """Scaled frame coordinates of points carrying a leading cell axis."""
        offset = points - self.centers[:, None, :]
        return (offset @ self.axes) / self.scales[:, None, None]


@partial(jax.jit, static_argnames=("role",))
def _top_frames(vertices: Array, top: Array, *, role: ComplexGeometryRole) -> _TopFrames:
    origin, jacobian = _simplex_jacobians(vertices, top)
    edges = jnp.swapaxes(jacobian, 1, 2)
    corners = jnp.concatenate((jnp.zeros_like(origin)[:, None, :], edges), axis=1)
    centers = origin + jnp.mean(corners, axis=1)
    differences = corners[:, :, None, :] - corners[:, None, :, :]
    scales = jnp.sqrt(jnp.max(jnp.sum(differences * differences, axis=-1), axis=(1, 2)))
    count, ambient = origin.shape
    match role:
        case "domain":
            axes = jnp.broadcast_to(
                jnp.eye(ambient, dtype=jnp.float64), (count, ambient, ambient)
            )
        case "closed_surface" | "open_surface":
            first = edges[:, 0] / jnp.linalg.norm(edges[:, 0], axis=-1, keepdims=True)
            if ambient == 2:
                axes = first[:, :, None]
            else:
                normal = jnp.cross(edges[:, 0], edges[:, 1])
                normal = normal / jnp.linalg.norm(normal, axis=-1, keepdims=True)
                axes = jnp.stack((first, jnp.cross(normal, first)), axis=-1)
        case _:
            raise ValueError("Invalid complex geometry role.")
    return _TopFrames(centers, axes, scales)


def _reference_rule(degree: int, order: int, /) -> tuple[Array, Array]:
    if degree == 0:
        return jnp.zeros((1, 0), dtype=jnp.float64), jnp.ones((1,), dtype=jnp.float64)
    # Reference cubature is static host data, also when compiled kernels trace it.
    with jax.ensure_compile_time_eval():
        rule = simplex_rule_data(degree, order)
        return jnp.asarray(rule.points), jnp.asarray(rule.weights)


def _moment_rows(
    vertices: Array,
    cells: Array,
    signs: Array,
    frames: _TopFrames,
    polynomial_degree: int,
    /,
) -> Array:
    """Exact moments s∫_σ π_T^*(ŷ^α dŷ^I) of local polynomial forms on k-cells.

    ``frames`` carries one frame per row of ``cells``. Features are ordered
    monomial-major, increasing k-subset minor.
    """
    degree = cells.shape[1] - 1
    origin, jacobian = _simplex_jacobians(vertices, cells)
    local = (jnp.swapaxes(frames.axes, 1, 2) @ jacobian) / frames.scales[:, None, None]
    minors = _compound(local, degree)[:, :, 0]
    reference, weights = _reference_rule(degree, polynomial_degree)
    points = origin[:, None, :] + reference @ jnp.swapaxes(jacobian, 1, 2)
    monomials = polynomial_design(frames.local(points), polynomial_degree)
    average = jnp.sum(weights[None, :, None] * monomials, axis=1)
    rows = signs[:, None, None] * average[:, :, None] * minors[:, None, :]
    return rows.reshape((rows.shape[0], -1))


def _bfs_orders(
    cofaces: np.ndarray, top_count: int, limit: int, /
) -> tuple[np.ndarray, ...]:
    """Facet-adjacency breadth-first top-cell order around each top cell.

    Host topology preparation: each order is bounded by ``limit`` cells and
    ties within one layer are broken by cell index for determinism.
    """
    interior = cofaces[np.all(cofaces >= 0, axis=1)]
    pairs = np.concatenate((interior, interior[:, ::-1]))
    order = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs = pairs[order]
    offsets = np.searchsorted(pairs[:, 0], np.arange(top_count + 1, dtype=np.int32))
    neighbors = pairs[:, 1]
    orders: list[np.ndarray] = []
    for start in range(top_count):
        visited = {start}
        sequence = [start]
        frontier = [start]
        while frontier and len(sequence) < limit:
            layer = sorted(
                {
                    int(neighbor)
                    for cell in frontier
                    for neighbor in neighbors[offsets[cell] : offsets[cell + 1]]
                    if int(neighbor) not in visited
                }
            )
            visited.update(layer)
            sequence.extend(layer)
            frontier = layer
        orders.append(np.asarray(sequence[:limit], dtype=np.int32))
    return tuple(orders)


def _patch_entities(
    order: np.ndarray, faces: np.ndarray, target: int, /
) -> tuple[list[int], int]:
    """First-appearance k-cells over the shortest BFS prefix reaching ``target``."""
    entities: dict[int, None] = {}
    used = 0
    for cell in order:
        used += 1
        for face in faces[cell]:
            entities.setdefault(int(face), None)
        if len(entities) >= target:
            break
    return list(entities), used


def _padded(rows: Sequence[Sequence[int]], /) -> tuple[np.ndarray, np.ndarray]:
    capacity = max(len(row) for row in rows)
    indices = np.zeros((len(rows), capacity), dtype=np.int32)
    valid = np.zeros((len(rows), capacity), dtype=np.bool_)
    for position, row in enumerate(rows):
        indices[position, : len(row)] = row
        valid[position, : len(row)] = True
    return indices, valid


def _patch_weights(
    centroids: Array, centers: Array, scales: Array, valid: Array
) -> Array:
    """Inverse-square moving-least-squares weights on normalized centroid distance."""
    distance = jnp.linalg.norm(centroids - centers[:, None, :], axis=-1) / scales[:, None]
    distance = jnp.where(valid, distance, 0.0)
    reach = jnp.max(distance, axis=1, keepdims=True)
    ratio = distance / jnp.where(reach > 0.0, reach, 1.0)
    return jnp.where(valid, 1.0 / jnp.maximum(ratio, 0.25) ** 2, 0.0)


@final
class _PreparedDegree(StrictModule):
    """Static moment patches and dynamic local reconstruction factors of degree k."""

    __strict_contract__ = True
    patch: Int32[_TopCellDim, _PatchDim]
    valid: Bool[_TopCellDim, _PatchDim]
    factors: Float64[_TopCellDim, _FeatureDim, _PatchDim]
    owner: Int32[_CellDim]
    owner_rows: Float64[_CellDim, _FeatureDim]
    degree: int = eqx.field(static=True)
    feature_count: int = eqx.field(static=True)

    def __init__(
        self,
        patch: Array,
        valid: Array,
        factors: Array,
        owner: Array,
        owner_rows: Array,
        /,
        *,
        degree: int,
        feature_count: int,
    ) -> None:
        self.patch = patch
        self.valid = valid
        self.factors = factors
        self.owner = owner
        self.owner_rows = owner_rows
        self.degree = degree
        self.feature_count = feature_count


class _DegreeAssembly(NamedTuple):
    prepared: _PreparedDegree
    hodge: SparseHodge
    evidence: MeshfreeComplexDegreeEvidence


def _upper_routes(patch: np.ndarray, valid: np.ndarray, /) -> tuple[np.ndarray, ...]:
    """Coalesce local patch pairs into one upper-triangular metric pattern."""
    rows = np.broadcast_to(patch[:, :, None], patch.shape + patch.shape[1:])
    columns = np.broadcast_to(patch[:, None, :], rows.shape)
    keep = (valid[:, :, None] & valid[:, None, :] & (rows <= columns)).reshape(-1)
    selected = np.flatnonzero(keep).astype(np.int32)
    pairs = np.stack((rows.reshape(-1)[selected], columns.reshape(-1)[selected]), axis=1)
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(inverse, kind="stable").astype(np.int32)
    return unique[:, 0], unique[:, 1], selected[order], inverse[order]


@partial(jax.jit, static_argnames=("polynomial_degree",))
def _patch_fit(
    vertices: Array,
    cells: Array,
    signs: Array,
    centroids: Array,
    frames: _TopFrames,
    patch: Array,
    valid: Array,
    *,
    polynomial_degree: int,
) -> tuple[Array, Array, Array, Array, Array]:
    """Compiled patch moment design and native weighted-SVD GMLS factors."""
    top_count, capacity = patch.shape
    flat = patch.reshape(-1)
    frame_index = jnp.repeat(jnp.arange(top_count, dtype=jnp.int32), capacity)
    rows = _moment_rows(
        vertices, cells[flat], signs[flat], frames.take(frame_index), polynomial_degree
    ).reshape((top_count, capacity, -1))
    matrix = jnp.where(valid[:, :, None], rows, 0.0)
    weights = _patch_weights(centroids[patch], frames.centers, frames.scales, valid)
    factors, rank, condition, minimum = weighted_svd_factors(matrix, weights, valid)
    return matrix, factors, rank, condition, minimum


@partial(jax.jit, static_argnames=("polynomial_degree",))
def _owner_rows(
    vertices: Array,
    cells: Array,
    signs: Array,
    frames: _TopFrames,
    owner: Array,
    *,
    polynomial_degree: int,
) -> Array:
    return _moment_rows(vertices, cells, signs, frames.take(owner), polynomial_degree)


@partial(jax.jit, static_argnames=("degree", "segments"))
def _local_metric(
    matrix: Array,
    factors: Array,
    node_monomials: Array,
    node_weights: Array,
    scales: Array,
    face_measures: Array,
    top_measures: Array,
    stabilization: Array,
    selected: Array,
    routes: Array,
    *,
    degree: int,
    segments: int,
) -> tuple[Array, Array]:
    """Coalesced upper Hodge values and the polynomial-reproduction residual.

    Consistency integrates ⟨π^*p, π^*q⟩ in orthonormal frames, where dŷ^I has
    norm h^-k. Stabilization penalizes the moment residual of each top cell's
    own faces with |T|/|σ|^2, which vanishes on reproduced polynomial forms
    and makes the sum positive definite whenever stabilization is positive.
    """
    top_count, capacity, feature_count = matrix.shape
    scalar_features = node_monomials.shape[2]
    components = feature_count // scalar_features
    face_count = face_measures.shape[1]
    scalar = (
        jnp.swapaxes(node_monomials * node_weights[:, :, None], 1, 2) @ node_monomials
    )
    gram = (
        scalar[:, :, None, :, None]
        * jnp.eye(components, dtype=jnp.float64)[None, None, :, None, :]
    ).reshape((top_count, feature_count, feature_count)) / scales[:, None, None] ** (
        2 * degree
    )
    consistency = jnp.swapaxes(factors, 1, 2) @ gram @ factors
    face_matrix = matrix[:, :face_count, :]
    selection = jax.nn.one_hot(
        jnp.arange(face_count, dtype=jnp.int32), capacity, dtype=jnp.float64
    )[None]
    residual = selection - face_matrix @ factors
    penalty = stabilization * top_measures[:, None] / (face_count * face_measures**2)
    local = consistency + jnp.swapaxes(residual, 1, 2) @ (penalty[:, :, None] * residual)
    values = jax.ops.segment_sum(
        local.reshape(-1)[selected],
        routes,
        num_segments=segments,
        indices_are_sorted=True,
    )
    reproduction = jnp.max(
        jnp.abs(face_matrix - face_matrix @ (factors @ matrix)), initial=0.0
    )
    return values, reproduction


class _DesignContext(NamedTuple):
    """Geometry shared by every degree's local design and assembly."""

    vertices: Array
    rows: tuple[Array, ...]
    signs: tuple[Array, ...]
    frames: _TopFrames
    centroids: tuple[Array, ...]
    polynomial_degree: int


def _patch_rows(
    degree: int,
    targets: np.ndarray,
    orders: tuple[np.ndarray, ...],
    faces: np.ndarray,
    policy: MeshfreeComplexPolicy,
    /,
) -> list[list[int]]:
    entities: list[list[int]] = []
    for cell, order in enumerate(orders):
        row, used = _patch_entities(order, faces, int(targets[cell]))
        if len(row) > policy.maximum_patch_entities:
            raise MeshfreeComplexAdmissionError(
                f"Degree-{degree} patch exceeds maximum_patch_entities."
            )
        if used >= policy.maximum_patch_cells and len(row) < targets[cell]:
            raise MeshfreeComplexAdmissionError(
                f"Degree-{degree} patch exceeds maximum_patch_cells before unisolvency."
            )
        entities.append(row)
    return entities


def _admitted_patches(
    degree: int,
    feature_count: int,
    orders: tuple[np.ndarray, ...],
    faces: np.ndarray,
    design: _DesignContext,
    policy: MeshfreeComplexPolicy,
    /,
) -> tuple[np.ndarray, np.ndarray, tuple[Array, Array, Array, Array, Array]]:
    """Grow facet-adjacency patches until every local fit is unisolvent.

    Each growth round adds one feature count of k-cells to every refused patch.
    A patch that cannot grow within its capacities is refused explicitly.
    """
    top_count = faces.shape[0]
    targets = np.full(
        (top_count,),
        max(feature_count, ceil(policy.oversampling * feature_count)),
        dtype=np.int64,
    )
    previous = np.zeros((top_count,), dtype=np.int64)
    while True:
        entities = _patch_rows(degree, targets, orders, faces, policy)
        patch, valid = _padded(entities)
        if top_count * patch.shape[1] * max(patch.shape[1], feature_count) > (
            policy.maximum_workset_entries
        ):
            raise MeshfreeComplexAdmissionError(
                f"Degree-{degree} local workset exceeds maximum_workset_entries."
            )
        fit = _patch_fit(
            design.vertices,
            design.rows[degree],
            design.signs[degree],
            design.centroids[degree],
            design.frames,
            jnp.asarray(patch),
            jnp.asarray(valid),
            polynomial_degree=design.polynomial_degree,
        )
        _, _, rank, condition, _ = fit
        refused = np.asarray(
            (rank < feature_count) | ~(condition <= policy.maximum_condition)
        )
        if not np.any(refused):
            return patch, valid, fit
        counts = np.asarray([len(row) for row in entities], dtype=np.int64)
        stalled = refused & (counts <= previous)
        if np.any(stalled):
            raise MeshfreeComplexAdmissionError(
                f"Degree-{degree} local reconstruction is not unisolvent on "
                f"{int(np.count_nonzero(stalled))} exhausted patches."
            )
        previous = np.where(refused, counts, previous)
        targets = np.where(refused, counts + feature_count, targets)


def _assemble_degree(
    degree: int,
    orders: tuple[np.ndarray, ...],
    faces: np.ndarray,
    design: _DesignContext,
    node_monomials: Array,
    node_weights: Array,
    measures: tuple[Array, ...],
    policy: MeshfreeComplexPolicy,
    /,
) -> _DegreeAssembly:
    dimension = len(measures) - 1
    feature_count = node_monomials.shape[2] * comb(dimension, degree)
    patch, valid, fit = _admitted_patches(
        degree, feature_count, orders, faces, design, policy
    )
    matrix, factors, rank, condition, minimum = fit
    top_count, capacity = patch.shape
    rows_, columns_, selected, routes = _upper_routes(patch, valid)
    values, reproduction = _local_metric(
        matrix,
        factors,
        node_monomials,
        node_weights,
        design.frames.scales,
        measures[degree][jnp.asarray(faces)],
        measures[dimension],
        jnp.asarray(policy.stabilization, dtype=jnp.float64),
        jnp.asarray(selected),
        jnp.asarray(routes),
        degree=degree,
        segments=rows_.shape[0],
    )
    cell_count = design.rows[degree].shape[0]
    hodge = SparseHodge(rows_, columns_, values, cell_count)
    owner = np.full((cell_count,), top_count, dtype=np.int32)
    np.minimum.at(
        owner,
        faces.reshape(-1),
        np.repeat(np.arange(top_count, dtype=np.int32), faces.shape[1]),
    )
    owner_rows = _owner_rows(
        design.vertices,
        design.rows[degree],
        design.signs[degree],
        design.frames,
        jnp.asarray(owner),
        polynomial_degree=design.polynomial_degree,
    )
    evidence = MeshfreeComplexDegreeEvidence(
        rank_deficient_patches=jnp.sum(rank < feature_count),
        maximum_condition=jnp.max(condition),
        minimum_singular_value=jnp.min(minimum),
        reproduction_residual=reproduction,
        hodge_admitted=hodge.valid,
        degree=degree,
        feature_count=feature_count,
        patch_capacity=capacity,
        hodge_routes=rows_.shape[0],
    )
    prepared = _PreparedDegree(
        jnp.asarray(patch),
        jnp.asarray(valid),
        factors,
        jnp.asarray(owner),
        owner_rows,
        degree=degree,
        feature_count=feature_count,
    )
    return _DegreeAssembly(prepared, hodge, evidence)


def _boundary_masks(authority: ComplexGeometryAuthority, /) -> tuple[np.ndarray, ...]:
    topology = authority.topology
    masks = [
        np.zeros((entities.count,), dtype=np.bool_) for entities in topology.entity_sets
    ]
    facets = np.asarray(authority.boundary_facets, dtype=np.bool_)
    if np.any(facets):
        closure = boundary_subcomplex(topology, boundary_mask=facets)
        for degree, indices in enumerate(closure.parent_indices):
            masks[degree][np.asarray(indices)] = True
    return tuple(masks)


@final
class MeshfreeCellComplexPlan(StrictModule):
    """Prepare a geometry-authorized meshfree realization of every degree."""

    authority: ComplexGeometryAuthority
    chart: CoordinateChart
    policy: MeshfreeComplexPolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        authority: ComplexGeometryAuthority,
        /,
        *,
        chart: CoordinateChart,
        policy: MeshfreeComplexPolicy | None = None,
    ) -> None:
        if not isinstance(authority, ComplexGeometryAuthority):
            raise TypeError(
                "authority must be a ComplexGeometryAuthority; abstract complexes are research-only."
            )
        if not isinstance(chart, CoordinateChart):
            raise TypeError("chart must be a CoordinateChart.")
        if chart.dimension != authority.vertices.shape[1]:
            raise ValueError("chart dimension must equal the ambient vertex dimension.")
        resolved = MeshfreeComplexPolicy() if policy is None else policy
        if not isinstance(resolved, MeshfreeComplexPolicy):
            raise TypeError("policy must be a MeshfreeComplexPolicy or None.")
        self.authority = authority
        self.chart = chart
        self.policy = resolved
        self.plan_id = canonical_fingerprint(
            {
                "kind": "meshfree-cell-complex-plan",
                "authority": authority.authority_id,
                "chart": [chart.name, list(chart.coordinates)],
                "policy": resolved.policy_id,
            }
        )

    def prepare(self, /) -> PreparedMeshfreeCellComplex:
        return PreparedMeshfreeCellComplex(self)


@partial(jax.jit, static_argnames=("polynomial_degree",))
def _node_layout(
    vertices: Array, top: Array, frames: _TopFrames, *, polynomial_degree: int
) -> tuple[Array, Array, Array]:
    """Top-cell quadrature nodes exact for degree 2m, with frame monomials."""
    dimension = top.shape[1] - 1
    reference, weights = _reference_rule(dimension, 2 * polynomial_degree)
    origin, jacobian = _simplex_jacobians(vertices, top)
    points = origin[:, None, :] + reference @ jnp.swapaxes(jacobian, 1, 2)
    volume = jnp.sqrt(
        _compound(jnp.swapaxes(jacobian, 1, 2) @ jacobian, dimension)[:, 0, 0]
    )
    node_weights = volume[:, None] * weights[None, :]
    return (
        points,
        node_weights,
        polynomial_design(frames.local(points), polynomial_degree),
    )


@partial(jax.jit, static_argnames=("polynomial_degree",))
def _sample_fit(
    nodes: Array,
    node_weights: Array,
    frames: _TopFrames,
    patch: Array,
    valid: Array,
    *,
    polynomial_degree: int,
) -> tuple[Array, Array, Array]:
    """Compiled GMLS scalar fits on the node patch of every top cell."""
    flat = nodes.reshape((-1, nodes.shape[2]))[patch]
    coordinates = frames.local(flat)
    design = jnp.where(
        valid[:, :, None], polynomial_design(coordinates, polynomial_degree), 0.0
    )
    distance = jnp.linalg.norm(coordinates, axis=-1)
    reach = jnp.max(jnp.where(valid, distance, 0.0), axis=1, keepdims=True)
    kernel = 1.0 / jnp.maximum(distance / jnp.where(reach > 0, reach, 1.0), 0.25) ** 2
    weights = jnp.where(valid, kernel * node_weights.reshape(-1)[patch], 0.0)
    factors, rank, condition, _ = weighted_svd_factors(design, weights, valid)
    return factors, rank, jnp.max(condition)


def _sample_factors(
    nodes: Array,
    node_weights: Array,
    frames: _TopFrames,
    orders: tuple[np.ndarray, ...],
    polynomial_degree: int,
    policy: MeshfreeComplexPolicy,
    /,
) -> tuple[np.ndarray, Array, Array]:
    """GMLS scalar fits on the nodes of each top cell and its facet neighbors."""
    _, per_cell, _ = nodes.shape
    features = comb(frames.axes.shape[2] + polynomial_degree, polynomial_degree)
    cell_count = max(2, ceil(policy.oversampling * features / per_cell))
    rows = [
        [
            int(cell) * per_cell + node
            for cell in order[:cell_count]
            for node in range(per_cell)
        ]
        for order in orders
    ]
    patch, valid = _padded(rows)
    factors, rank, condition = _sample_fit(
        nodes,
        node_weights,
        frames,
        jnp.asarray(patch),
        jnp.asarray(valid),
        polynomial_degree=polynomial_degree,
    )
    if bool(np.any(np.asarray(rank < features))):
        raise MeshfreeComplexAdmissionError(
            "Meshfree sample patches are not unisolvent for the local polynomial degree."
        )
    return patch, factors, condition


@final
class PreparedMeshfreeCellComplex(StrictModule):
    """Native cochain realization with meshfree reconstruction and sampling.

    ``cochain`` is the canonical ``CochainDiscretization`` consumed by exterior
    traces/cohomology, Hodge–Laplace and Maxwell solvers. ``bridge`` is the
    exact de Rham integration map. ``sample`` and ``reconstruct`` are the
    meshfree GMLS maps between node samples and k-cell moments.
    """

    __strict_contract__ = True
    plan: MeshfreeCellComplexPlan
    cochain: CochainDiscretization
    bridge: DeRhamBridge
    nodes: Float64[_NodeDim, _AmbientDim]
    node_weights: Float64[_NodeDim]
    evidence: MeshfreeComplexEvidence
    _frames: _TopFrames
    _degrees: tuple[_PreparedDegree, ...]
    _node_monomials: Float64[_TopCellDim, _CellNodeDim, _ScalarFeatureDim]
    _sample_patch: Int32[_TopCellDim, _NodePatchDim]
    _sample_factors: Float64[_TopCellDim, _ScalarFeatureDim, _NodePatchDim]
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: MeshfreeCellComplexPlan, /) -> None:
        if not isinstance(plan, MeshfreeCellComplexPlan):
            raise TypeError("plan must be a MeshfreeCellComplexPlan.")
        authority, policy = plan.authority, plan.policy
        topology = authority.topology
        dimension = topology.dimension
        host_rows, host_signs = simplicial_cell_geometry(topology)
        faces = _top_faces(host_rows)
        rows = tuple(jnp.asarray(cells) for cells in host_rows)
        signs = tuple(jnp.asarray(values, dtype=jnp.float64) for values in host_signs)
        vertices = authority.vertices
        frames = _top_frames(vertices, rows[-1], role=authority.identity.role)
        cofaces, _ = _facet_cofaces(topology)
        orders = _bfs_orders(cofaces, rows[-1].shape[0], policy.maximum_patch_cells)
        points, weights, monomials = _node_layout(
            vertices, rows[-1], frames, polynomial_degree=policy.polynomial_degree
        )
        centroids = tuple(jnp.mean(vertices[cells], axis=1) for cells in rows)
        design = _DesignContext(
            vertices, rows, signs, frames, centroids, policy.polynomial_degree
        )
        assemblies = tuple(
            _assemble_degree(
                degree,
                orders,
                faces[degree],
                design,
                monomials,
                weights,
                authority.measures,
                policy,
            )
            for degree in range(dimension + 1)
        )
        for assembly in assemblies:
            if not bool(np.asarray(assembly.evidence.hodge_admitted)):
                raise MeshfreeComplexAdmissionError(
                    f"Degree-{assembly.evidence.degree} meshfree Hodge failed native "
                    "sparse Cholesky positive-definite admission."
                )
        sample_patch, sample_factors, sample_condition = _sample_factors(
            points, weights, frames, orders, policy.polynomial_degree, policy
        )
        cochain = CochainDiscretization(
            topology,
            tuple(assembly.hodge for assembly in assemblies),
            boundary_masks=_boundary_masks(authority),
            coordinates=centroids,
            primal_measures=authority.measures,
        )
        bridge = DeRhamBridge(
            cochain,
            plan.chart,
            simplicial_parameterizations(
                topology, vertices, order=policy.quadrature_order
            ),
        )
        evidence = MeshfreeComplexEvidence(
            degrees=tuple(assembly.evidence for assembly in assemblies),
            sample_maximum_condition=sample_condition,
            measure=authority.measure,
            betti=authority.identity.betti,
            boundary_facet_count=int(
                np.count_nonzero(np.asarray(authority.boundary_facets))
            ),
            authority_id=authority.authority_id,
            domain_id=authority.identity.domain_id,
            role=authority.identity.role,
            fidelity="geometry-authorized",
        )
        self.plan = plan
        self.cochain = cochain
        self.bridge = bridge
        self.nodes = points.reshape((-1, points.shape[2]))
        self.node_weights = weights.reshape(-1)
        self.evidence = evidence
        self._frames = frames
        self._degrees = tuple(assembly.prepared for assembly in assemblies)
        self._node_monomials = monomials
        self._sample_patch = jnp.asarray(sample_patch)
        self._sample_factors = sample_factors
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-meshfree-cell-complex",
                "plan": plan.plan_id,
                "realization": cochain.realization_id,
                "nodes": array_tree_fingerprint(np.asarray(points)),
            }
        )

    def _components(self, degree: int, /) -> int:
        if isinstance(degree, bool) or not isinstance(degree, int):
            raise TypeError("degree must be an integer.")
        if not 0 <= degree <= self.cochain.dimension:
            raise ValueError(f"degree must lie in [0, {self.cochain.dimension}].")
        return comb(self.nodes.shape[1], degree)

    def reconstruct(self, form: DiscreteForm, /) -> Array:
        """Evaluate the local GMLS k-form reconstruction at every node.

        Returns ambient components ``(nodes, binomial(D, k))`` of π_T^* p_T on
        each top cell T, with lexicographic increasing-index component order.
        """
        if not isinstance(form, DiscreteForm):
            raise TypeError("form must be a DiscreteForm.")
        degree = form.form_type.degree
        self._components(degree)
        if (
            form.realization_id != self.cochain.realization_id
            or form.form_type != self.cochain.form_type(degree)
        ):
            raise ValueError("form must be a primal form of this realization.")
        prepared = self._degrees[degree]
        values = jnp.where(prepared.valid, form.values[prepared.patch], 0.0)
        coefficients = (prepared.factors @ values[:, :, None])[:, :, 0]
        local = coefficients.reshape(
            (coefficients.shape[0], self._node_monomials.shape[2], -1)
        )
        tangential = self._node_monomials @ local
        pushforward = _compound(
            jnp.swapaxes(self._frames.axes, 1, 2) / self._frames.scales[:, None, None],
            degree,
        )
        return (tangential @ pushforward).reshape((self.nodes.shape[0], -1))

    def sample(self, degree: int, values: ArrayLike, /) -> DiscreteForm:
        """Map ambient k-form samples at the nodes to GMLS k-cell moments.

        Each k-cell takes the exact moment of the local polynomial fitted on the
        node patch of its owning top cell, in that cell's intrinsic frame.
        """
        components = self._components(degree)
        samples = jnp.asarray(values, dtype=jnp.float64)
        if samples.shape != (self.nodes.shape[0], components):
            raise ValueError(
                f"values must have shape ({self.nodes.shape[0]}, {components})."
            )
        prepared = self._degrees[degree]
        frames = self._frames.take(prepared.owner)
        pullback = _compound(frames.axes * frames.scales[:, None, None], degree)
        patch = self._sample_patch[prepared.owner]
        tangential = samples[patch] @ pullback
        fitted = self._sample_factors[prepared.owner] @ tangential
        moments = jnp.sum(
            prepared.owner_rows * fitted.reshape(fitted.shape[0], -1), axis=1
        )
        return DiscreteForm(
            self.cochain.realization_id, self.cochain.form_type(degree), moments
        )


@final
class RadiusCliquePolicy(StrictModule):
    """Simplex, work, pair, and topology capacities of the abstract clique route."""

    maximum_dimension: int = eqx.field(static=True)
    maximum_pairs: int = eqx.field(static=True)
    maximum_simplices: int = eqx.field(static=True)
    maximum_work: int = eqx.field(static=True)
    resources: TopologyResourcePolicy | None

    def __init__(
        self,
        *,
        maximum_pairs: int,
        maximum_dimension: int = 2,
        maximum_simplices: int = 100_000,
        maximum_work: int = 1_000_000,
        resources: TopologyResourcePolicy | None = None,
    ) -> None:
        if resources is not None and not isinstance(resources, TopologyResourcePolicy):
            raise TypeError("resources must be a TopologyResourcePolicy or None.")
        self.maximum_dimension = positive_integer(maximum_dimension, "maximum_dimension")
        self.maximum_pairs = positive_integer(maximum_pairs, "maximum_pairs")
        self.maximum_simplices = positive_integer(maximum_simplices, "maximum_simplices")
        self.maximum_work = positive_integer(maximum_work, "maximum_work")
        self.resources = resources


@final
class RadiusCliqueComplex(StrictModule):
    """Abstract research clique complex of a bounded radius graph.

    Orientation is the ascending-vertex simplicial orientation, d∘d = 0 is the
    native exact chain verification, and Betti numbers are exact rational
    topology. No measure, Hodge, interpolation, or continuum fidelity exists.
    """

    __strict_contract__ = True
    topology: CellComplexTopology
    points: Float64[_VertexDim, _AmbientDim]
    chain: ExactChainComplex
    betti: BettiDimensionResult
    radius: float = eqx.field(static=True)
    simplex_counts: tuple[int, ...] = eqx.field(static=True)
    work: int = eqx.field(static=True)
    fidelity: ComplexFidelity = eqx.field(static=True)
    complex_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: CellComplexTopology,
        points: Array,
        chain: ExactChainComplex,
        betti: BettiDimensionResult,
        /,
        *,
        radius: float,
        work: int,
    ) -> None:
        self.topology = topology
        self.points = points
        self.chain = chain
        self.betti = betti
        self.radius = radius
        self.simplex_counts = tuple(entities.count for entities in topology.entity_sets)
        self.work = work
        self.fidelity = "abstract-research"
        self.complex_id = topology.topology_id


def _upper_adjacency(pairs: np.ndarray, count: int, /) -> tuple[np.ndarray, ...]:
    lower = np.minimum(pairs[:, 0], pairs[:, 1])
    upper = np.maximum(pairs[:, 0], pairs[:, 1])
    order = np.lexsort((upper, lower))
    offsets = np.searchsorted(lower[order], np.arange(count + 1, dtype=np.int32))
    neighbors = upper[order]
    return tuple(neighbors[offsets[index] : offsets[index + 1]] for index in range(count))


def _clique_levels(
    adjacency: tuple[np.ndarray, ...], policy: RadiusCliquePolicy, /
) -> tuple[tuple[np.ndarray, ...], int]:
    """Enumerate ascending cliques level by level, refusing before any capacity."""
    count = len(adjacency)
    levels = [np.arange(count, dtype=np.int32)[:, None]]
    total = count
    work = 0
    for size in range(2, policy.maximum_dimension + 2):
        extended: list[tuple[int, ...]] = []
        for simplex in levels[-1]:
            cost = sum(adjacency[int(vertex)].size for vertex in simplex)
            work += cost
            if work > policy.maximum_work:
                raise MeshfreeComplexAdmissionError(
                    "Radius-clique enumeration exceeds maximum_work."
                )
            candidates = adjacency[int(simplex[0])]
            for vertex in simplex[1:]:
                candidates = np.intersect1d(
                    candidates, adjacency[int(vertex)], assume_unique=True
                )
            for vertex in candidates:
                total += 1
                if total > policy.maximum_simplices:
                    raise MeshfreeComplexAdmissionError(
                        "Radius-clique complex exceeds maximum_simplices."
                    )
                extended.append((*(int(value) for value in simplex), int(vertex)))
        if not extended:
            break
        levels.append(np.asarray(sorted(extended), dtype=np.int32).reshape((-1, size)))
    return tuple(levels), work


def _exact_chain(topology: CellComplexTopology, /) -> ExactChainComplex:
    counts = tuple(entities.count for entities in topology.entity_sets)
    boundaries = [
        ExactIntegerCOO(
            0,
            counts[0],
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            (),
            source_id=f"{topology.topology_id}:chains:0",
            target_id=f"{topology.topology_id}:chains:-1",
        )
    ]
    for incidence in topology.incidences:
        valid = np.asarray(incidence.relation.valid, dtype=np.bool_)
        boundaries.append(
            ExactIntegerCOO(
                counts[incidence.degree - 1],
                counts[incidence.degree],
                np.asarray(incidence.relation.source_indices, dtype=np.int32)[valid],
                np.asarray(incidence.relation.target_indices, dtype=np.int32)[valid],
                tuple(int(value) for value in np.asarray(incidence.signs)[valid]),
                source_id=f"{topology.topology_id}:chains:{incidence.degree}",
                target_id=f"{topology.topology_id}:chains:{incidence.degree - 1}",
            )
        )
    return ExactChainComplex(boundaries, complex_id=topology.topology_id)


def radius_clique_complex(
    points: ArrayLike, radius: float, /, *, policy: RadiusCliquePolicy
) -> RadiusCliqueComplex:
    """Build the bounded abstract clique complex of a meshfree radius graph.

    The native radius relation bounds pair capacity; enumeration refuses before
    exceeding simplex or work capacity. The result is research-only evidence:
    it is never admitted as geometry authority for a continuum realization.
    """
    if not isinstance(policy, RadiusCliquePolicy):
        raise TypeError("policy must be a RadiusCliquePolicy.")
    cloud = np.asarray(points, dtype=np.float64)
    radius_ = _finite_float(radius, "radius", minimum=0.0)
    if radius_ <= 0.0:
        raise ValueError("radius must be positive.")
    relation = MeshfreeEdgeRelationPlan(cloud, radius_, policy.maximum_pairs).prepare()
    valid = np.asarray(relation.relation.valid, dtype=np.bool_)
    pairs = np.stack(
        (
            np.asarray(relation.relation.source_indices)[valid],
            np.asarray(relation.relation.target_indices)[valid],
        ),
        axis=1,
    ).astype(np.int32)
    levels, work = _clique_levels(_upper_adjacency(pairs, cloud.shape[0]), policy)
    topology = simplicial_cell_complex(
        levels,
        topology_id=canonical_fingerprint(
            {
                "kind": "radius-clique-complex",
                "points": array_tree_fingerprint(cloud),
                "radius": radius_,
                "maximum_dimension": policy.maximum_dimension,
            }
        ),
    )
    chain = _exact_chain(topology)
    betti = compute_betti_dimensions(
        topology, coefficients=RationalField(), resources=policy.resources
    )
    return RadiusCliqueComplex(
        topology, jnp.asarray(cloud), chain, betti, radius=radius_, work=work
    )


__all__ = [
    "ComplexDomainIdentity",
    "ComplexFidelity",
    "ComplexGeometryAuthority",
    "ComplexGeometryRole",
    "MeshfreeCellComplexPlan",
    "MeshfreeComplexAdmissionError",
    "MeshfreeComplexDegreeEvidence",
    "MeshfreeComplexEvidence",
    "MeshfreeComplexPolicy",
    "PreparedMeshfreeCellComplex",
    "RadiusCliqueComplex",
    "RadiusCliquePolicy",
    "radius_clique_complex",
]
