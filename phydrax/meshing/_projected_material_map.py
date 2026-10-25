# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Local constrained finite-element transport through a cavity-local material chain.

Every chain edit re-solves only the DOFs of its created cells and the surviving
ring that shares a DOF-carrying entity with them; DOFs on entities shared with
cells outside that patch stay fixed. Collapses and flips have no interior
entities, so the ring carries their freedom. Patches are canonical owner
spaces: ``FiniteElementPlan`` on meshes whose vertex rows ascend by global ID,
so every shared-entity coefficient frame is the owner's canonical frame of its
global vertices and one DOF identity spans every patch. Mixed integrals use the
chain's exact material pieces, pulling simplices gathered from each edit's
shared arrangement vertices (each rounded once), with the owner overlap rule
for the actual source/target basis product degree and the owner mapped bases.

Material semantics:

* ``intensive``: scalar L2 projection of ``u o A^-1``;
* ``conservative-density``: scalar projection of ``rho o A^-1 / J``,
  constrained to the exact patch content of the old values;
* ``material-compatible``: covariant (``L^-T u``) or contravariant
  (``L u / J``) Piola pullback, constrained to the patch vector content and, in
  H(div), to the cellwise divergence moments of the pullback.

Every local KKT system has one rank-revealing factorization applied to all
inputs; an infeasible constraint refuses with a typed status.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._polynomial._cubature import cubature_rule_data
from ..discretization._cell_geometry import CellGeometrySpec
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_mesh import CellMesh
from ..discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateLiveStorage,
)
from ..discretization._transfer import TransferGeometryBinding, TransferSemantics
from ..discretization.fem._assembly import (
    FormDofLayout,
    lifted_dof_entities,
    lifted_finite_element_dof_layout,
    topology_vertex_sets,
)
from ..discretization.fem._generic import (
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPlan,
)
from ..discretization.fem._reference_types import FiniteElementSpec
from ..discretization.fem._topology_transfer import (
    _cell_integrals,
    _differential_tests,
    _overlap_basis,
    _overlap_quadrature,
    _polynomial_degree,
    _projection_field_elements,
    _require_affine_simplices,
    CLAIM_ULPS,
    coalesced_map,
    FiniteElementFieldTransfer,
    FiniteElementTopologyTransfer,
    FiniteElementTransferEvidence,
    SourceRealizationFieldSemantics,
)
from ..geometry._volume_material_map import (
    VolumeMaterialChain,
    VolumeMaterialMapError,
    VolumeMaterialStatus,
)
from ..linalg import (
    DenseLinearOperator,
    FactorizationKind,
    FactorizationPolicy,
    factorize,
    OperatorProperties,
    PreparedFactorization,
    RHSLayout,
    SmallLinearSolvePlan,
    solve_small_linear,
)
from ..sparse import EdgeRelation
from ..typing import parse


_EPSILON = float(np.finfo(np.float64).eps)
_MAXIMUM_LOCAL_CONDITION = 1e12

type _DofKey = tuple[int, tuple[int, ...], int]
type _Row = tuple[np.ndarray, np.ndarray]


def _refuse(status: VolumeMaterialStatus, message: str) -> VolumeMaterialMapError:
    return VolumeMaterialMapError(status, message)


@dataclass(frozen=True, slots=True)
class _Group:
    """Fields sharing one element, one source DOF map and one material meaning."""

    names: tuple[str, ...]
    spec: FiniteElementFieldSpec
    element: FiniteElementSpec
    meaning: SourceRealizationFieldSemantics
    ring_vertices: int | None


@dataclass(slots=True)
class _GroupState:
    """Composed rows of every current DOF over the group's source endpoint DOFs."""

    rows: dict[_DofKey, _Row]
    # Certified l1 forward-error bound of every composed row (``|computed -
    # exact|`` summed over source DOFs), propagated through every stage.
    errors: dict[_DofKey, float]
    # Live owner admitting ``rows``/``errors`` across every scope; its bound
    # is raised before rows are built and reconciled to their actual bytes.
    storage: CoordinateLiveStorage | None = None
    condition: float = 0.0
    constraint_defect: float = 0.0
    constraint_names: tuple[str, ...] = ()
    seen: set[str] = field(default_factory=set)


def _canonical_mesh(
    vertices: np.ndarray, points: np.ndarray, cell_ids: np.ndarray, block_name: str, /,
) -> CellMesh:
    """One-block tetrahedral mesh whose vertex rows ascend by global vertex ID."""
    identifiers, inverse = np.unique(vertices.reshape(-1), return_inverse=True)
    coordinates = np.zeros((identifiers.size, 3), dtype=np.float64)
    corners = points.reshape((-1, 3))
    coordinates[inverse] = corners
    if not np.array_equal(coordinates[inverse], corners):
        raise _refuse(
            VolumeMaterialStatus.STALE_IDENTITY,
            "volume_material_projection vertex IDs must name one point in every patch",
        )
    return CellMesh.from_tetrahedra(
        coordinates,
        inverse.reshape(vertices.shape).astype(np.int64),
        block_name=block_name,
        vertex_global_ids=identifiers,
        cell_global_ids=cell_ids,
    )


def _endpoint_canonical(mesh: CellMesh, /) -> CellMesh:
    if len(mesh.blocks) != 1 or mesh.blocks[0].cell_kind != "tetrahedron" or mesh.periodic_topology is not None:
        raise _refuse(
            VolumeMaterialStatus.UNSUPPORTED_SPACE,
            "volume_material_projection endpoints must be one nonperiodic tetrahedral block",
        )
    block = mesh.blocks[0]
    rows = np.asarray(block.vertices, dtype=np.int64)
    identifiers = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    return _canonical_mesh(
        identifiers[rows], np.asarray(mesh.coordinates, dtype=np.float64)[rows],
        np.asarray(block.global_ids, dtype=np.int64), block.name,
    )


def _space(
    mesh: CellMesh, spec: FiniteElementFieldSpec, budget: CoordinateEnclosureBudget, /,
) -> FiniteElementDiscretization:
    return FiniteElementPlan(
        mesh, (spec,), coordinate_spec=CellGeometrySpec.affine(mesh)
    ).prepare(budget=budget)


def _dof_keys(space: FiniteElementDiscretization, /) -> tuple[_DofKey, ...]:
    """Owner entity, global vertex IDs and entity position of every DOF.

    On canonical-row meshes the owner's coefficient frame of a shared entity
    depends only on its global vertices, so this key is one scientific DOF.
    """
    mesh = space.mesh
    spec = space.construction_plan.fields[0]
    layout = lifted_finite_element_dof_layout(mesh, space.elements[0], spec.component_shape)
    if layout.global_count != space.dof_maps[0].global_dof_count:
        raise _refuse(
            VolumeMaterialStatus.UNSUPPORTED_SPACE,
            "volume_material_projection DOFs must be the lifted owner layout",
        )
    degrees, entities, positions, _ = lifted_dof_entities(mesh, layout)
    vertex_sets = topology_vertex_sets(mesh)
    identifiers = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cells = identifiers[np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])]
    dimension = mesh.topological_dimension
    # Form moments are permutation covariant on every entity. Nodal and
    # cell-associated interior DOFs keep the reference order of their cell's
    # corners, so their identity retains that order and never matches a
    # reordered cell.
    ordered = not isinstance(layout, FormDofLayout)

    def vertices(degree: int, entity: int) -> tuple[int, ...]:
        if ordered and degree == dimension:
            return tuple(int(value) for value in cells[entity])
        return tuple(sorted(int(identifiers[row]) for row in vertex_sets[degree][entity]))

    return tuple(
        (int(degree), vertices(int(degree), int(entity)), int(position))
        for degree, entity, position in zip(degrees, entities, positions, strict=True)
    )


def _factor(matrix: np.ndarray, method: FactorizationKind, /) -> PreparedFactorization:
    return factorize(DenseLinearOperator(jnp.asarray(matrix)), FactorizationPolicy(method))


def _spectrum(factor: PreparedFactorization, size: int, /) -> tuple[float, float]:
    """``(||A||_2, kappa_2)`` from the prepared SVD factor's retained spectrum.

    Requires at least ``size`` retained singular values; computed once per
    prepared factor by its owner and reused by every solve against it.
    """
    rank = int(np.asarray(factor.rank()))
    retained = np.asarray(factor.singular_values(), dtype=np.float64)[:rank]
    if rank < size or rank == 0 or not np.all(np.isfinite(retained)) or not retained[-1] > 0.0:
        raise _refuse(
            VolumeMaterialStatus.INFEASIBLE_PROJECTION,
            f"volume_material_projection local system must determine its unknowns; got rank {rank} of {size}",
        )
    return float(retained[0]), float(retained[0] / retained[-1])


def _solve(
    factor: PreparedFactorization, spectrum: tuple[float, float], rhs: np.ndarray, /,
) -> tuple[np.ndarray, np.ndarray]:
    """Owner solve of every column with its certified forward-error bound.

    ``eta_j = ||A x_j - b_j|| / (||A|| ||x_j|| + ||b_j||)`` is the Rigal-Gaches
    backward error of column ``j``; a backward-stable solve of a consistent
    system meets the claim tolerance, while an inconsistent constrained
    system leaves a residual of the order of its incompatibility. With the
    factor's retained-spectrum condition ``kappa`` the 2-norm forward error of column
    ``j`` is at most ``kappa eta_j ||x_j|| / (1 - kappa eta_j)``.
    """
    solved = factor.solve(jnp.asarray(rhs), rhs_layout=RHSLayout((rhs.shape[1],)))
    value = np.asarray(solved.value, dtype=np.float64)
    diagnostics = solved.diagnostics
    residual = np.asarray(diagnostics.residual_norm, dtype=np.float64)
    norm, condition = spectrum
    scale = norm * np.linalg.norm(value, axis=0) + np.linalg.norm(rhs, axis=0)
    if (
        not bool(np.all(np.asarray(solved.successful)))
        or not bool(np.all(np.asarray(diagnostics.converged)))
        or not np.all(np.isfinite(value))
        or np.any(residual > CLAIM_ULPS * _EPSILON * scale)
    ):
        raise _refuse(
            VolumeMaterialStatus.INFEASIBLE_PROJECTION,
            "volume_material_projection local system must be solvable at binary64 backward error; "
            f"got backward errors up to {float(np.max(residual / np.maximum(scale, np.finfo(np.float64).tiny), initial=0.0)):.3e}",
        )
    backward = residual / np.maximum(scale, np.finfo(np.float64).tiny)
    amplification = condition * backward
    if not np.isfinite(condition) or np.any(amplification >= 0.5):
        raise _refuse(
            VolumeMaterialStatus.INFEASIBLE_PROJECTION,
            f"volume_material_projection local solve forward error must be resolved; got condition {condition:.3e}",
        )
    return value, amplification * np.linalg.norm(value, axis=0) / (1.0 - amplification)


class _TransformInverses:
    """Owner SVD inverses of the distinct cell transforms of one endpoint."""

    def __init__(self) -> None:
        self._factors: dict[bytes, tuple[PreparedFactorization, tuple[float, float]]] = {}
        self._solutions: dict[tuple[bytes, bytes], tuple[np.ndarray, float]] = {}

    def apply(self, transform: np.ndarray, rhs: np.ndarray, /) -> tuple[np.ndarray, float]:
        """``transform^-1 rhs`` and its row-l1 error bound; equal inputs reuse one solve."""
        key = np.ascontiguousarray(transform).tobytes() + bytes(str(transform.shape), "ascii")
        pair = (key, np.ascontiguousarray(rhs).tobytes() + bytes(str(rhs.shape), "ascii"))
        solution = self._solutions.get(pair)
        if solution is not None:
            return solution
        prepared = self._factors.get(key)
        if prepared is None:
            factor = _factor(transform, "svd")
            prepared = (factor, _spectrum(factor, transform.shape[0]))
            self._factors[key] = prepared
        value, forward = _solve(prepared[0], prepared[1], rhs)
        # Each entry errs by at most its column's 2-norm bound.
        solution = (value, float(np.sum(forward)))
        self._solutions[pair] = solution
        return solution


def _cell_maps(
    actual: FiniteElementDiscretization,
    field: int,
    canonical: FiniteElementDiscretization,
    inverses: _TransformInverses,
    to_canonical: bool,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per cell ``(actual routes, canonical routes, K)`` between local coefficients.

    Both maps realize one local moment vector ``T_a c_a = T_c c_c`` of the same
    cell and corner order, so ``K = T_c^-1 T_a`` (or its inverse direction).
    """
    actual_routes = np.asarray(actual.dof_maps[field].cell_dofs[0], dtype=np.int64)
    canonical_routes = np.asarray(canonical.dof_maps[0].cell_dofs[0], dtype=np.int64)
    actual_transforms = np.asarray(actual.dof_maps[field].cell_transforms[0], dtype=np.float64)
    canonical_transforms = np.asarray(canonical.dof_maps[0].cell_transforms[0], dtype=np.float64)
    maps = np.empty_like(actual_transforms)
    errors = np.zeros((actual_routes.shape[0],), dtype=np.float64)
    for cell in range(actual_routes.shape[0]):
        if to_canonical:
            maps[cell], errors[cell] = inverses.apply(canonical_transforms[cell], actual_transforms[cell])
        else:
            maps[cell], errors[cell] = inverses.apply(actual_transforms[cell], canonical_transforms[cell])
    return actual_routes, canonical_routes, maps, errors


def _combine(
    rows: Sequence[_Row],
    errors: Sequence[float],
    coefficients: np.ndarray,
    coefficient_error: float,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``coefficients @ rows`` over the union of row supports, with that union.

    Also returns each composed row's certified l1 forward error: inherited
    row errors through ``|coefficients|``, the coefficient matrix's own row
    l1 error ``coefficient_error`` times the largest input row, and the
    binary64 accumulation of every product.
    """
    columns, inverse = np.unique(np.concatenate([routes for routes, _ in rows]), return_inverse=True)
    dense = np.zeros((len(rows), columns.size), dtype=np.float64)
    owners = np.repeat(np.arange(len(rows)), [routes.size for routes, _ in rows])
    np.add.at(dense, (owners, inverse), np.concatenate([values for _, values in rows]))
    magnitude = np.abs(coefficients)
    norms = np.sum(np.abs(dense), axis=1)
    count = len(rows)
    gamma = count * _EPSILON / (1.0 - count * _EPSILON)
    composed_errors = (
        magnitude @ np.asarray(errors, dtype=np.float64)
        + coefficient_error * float(np.max(norms, initial=0.0))
        + gamma * (magnitude @ norms)
    )
    return np.asarray(coefficients @ dense), columns, composed_errors


def _sparse(row: np.ndarray, columns: np.ndarray, /) -> _Row:
    support = np.flatnonzero(row)
    return columns[support], row[support]


def _patch_cell_bytes(element: FiniteElementSpec, /) -> int:
    """Conservative owner preparation bytes of one patch cell.

    Per cell: coordinates and routes, orientations, the local transform
    (``L^2``), coordinate weights and DOF coordinates, and the plan's
    tabulation of every basis value and gradient at a rule exact to twice the
    element degree (at most ``(2p + 2)^3`` points), all binary64.
    """
    local = element.local_dof_count
    points = (2 * max(1, _polynomial_degree(element)) + 2) ** 3
    return 8 * (64 + 8 * local + local * local + 16 * points * local)


def _groups(
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    semantics: Mapping[str, SourceRealizationFieldSemantics],
    /,
) -> tuple[_Group, ...]:
    type GroupKey = tuple[str, str, SourceRealizationFieldSemantics]
    grouped: dict[GroupKey, list[str]] = {}
    specs: dict[GroupKey, tuple[FiniteElementFieldSpec, FiniteElementSpec]] = {}
    for name, meaning in semantics.items():
        si, ti = source.field_index(name), target.field_index(name)
        elements = _projection_field_elements(source, si, "source")
        targets = _projection_field_elements(target, ti, "target")
        element = elements[0]
        if (
            len({value.element_id for value in elements + targets}) != 1
            or source.dof_maps[si].component_shape != target.dof_maps[ti].component_shape
        ):
            raise _refuse(
                VolumeMaterialStatus.UNSUPPORTED_SPACE,
                f"volume_material_projection field {name!r} must keep one element on both endpoints",
            )
        meaning = parse(
            meaning, SourceRealizationFieldSemantics, "volume_material_projection.field_semantics"
        )
        piola = element.mapping in ("covariant_piola", "contravariant_piola")
        if piola != (meaning == "material-compatible"):
            raise _refuse(
                VolumeMaterialStatus.UNSUPPORTED_SPACE,
                f"volume_material_projection field {name!r} must pair scalar fields with intensive or "
                f"conservative-density and Piola fields with material-compatible; got {meaning!r} for {element.mapping!r}",
            )
        key: GroupKey = (element.element_id, source.dof_maps[si].dof_map_id, meaning)
        grouped.setdefault(key, []).append(name)
        specs[key] = (source.construction_plan.fields[si], element)
    result = []
    for key, names in grouped.items():
        spec, element = specs[key]
        degrees = [
            degree for degree in range(3)
            if any(len(dofs) for dofs in element.entity_dofs[degree])
        ]
        result.append(_Group(
            tuple(names), spec, element, key[2],
            None if not degrees else min(degrees) + 1,
        ))
    return tuple(result)


def _row_bytes(rows: Mapping[_DofKey, _Row], /) -> int:
    """Retained bytes of composed rows: routes, values, key and error per row."""
    return sum(routes.nbytes + values.nbytes + 256 for routes, values in rows.values())


def _initial_rows(
    space: FiniteElementDiscretization, field: int, spec: FiniteElementFieldSpec,
    storage: CoordinateLiveStorage, budget: CoordinateEnclosureBudget, /,
) -> _GroupState:
    """Canonical-frame DOF rows over the actual source DOFs, with their errors."""
    element = space.elements[field][0]
    local = element.local_dof_count
    cells = sum(block.cell_count for block in space.mesh.blocks)
    # Rows outlive the canonical copy's scope: at most one row per local DOF
    # of every cell, each with at most ``local`` source entries. The live
    # owner admits them before construction and keeps them admitted.
    storage.set_bound(storage.bound + cells * local * (16 * local + 256))
    with budget.temporary_scope():
        budget.reserve(0, cells * _patch_cell_bytes(element))
        state = _canonical_initial_rows(space, field, spec, budget)
    state.storage = storage
    storage.set_bound(_row_bytes(state.rows))
    return state


def _canonical_initial_rows(
    space: FiniteElementDiscretization, field: int, spec: FiniteElementFieldSpec,
    budget: CoordinateEnclosureBudget, /,
) -> _GroupState:
    canonical = _space(_endpoint_canonical(space.mesh), spec, budget)
    actual_routes, canonical_routes, maps, map_errors = _cell_maps(
        space, field, canonical, _TransformInverses(), True,
    )
    keys = _dof_keys(canonical)
    rows: dict[_DofKey, _Row] = {}
    errors: dict[_DofKey, float] = {}
    for routes, local, matrix, error in zip(actual_routes, canonical_routes, maps, map_errors, strict=True):
        for position, dof in enumerate(local.tolist()):
            if keys[dof] not in rows:
                rows[keys[dof]] = _sparse(matrix[position], routes)
                errors[keys[dof]] = float(error)
    return _GroupState(rows, errors)


def _local_projection(
    group: _Group,
    state: _GroupState,
    chain: VolumeMaterialChain,
    index: int,
    cells: dict[int, tuple[np.ndarray, np.ndarray]],
    incidence: dict[int, set[int]],
    block_name: str,
    budget: CoordinateEnclosureBudget,
    /,
) -> None:
    """Solve and compose one edit's local projection for one field group."""
    edit = chain.edits[index]
    removed = [int(value) for value in np.asarray(edit.source_cell_ids)]
    created = [int(value) for value in np.asarray(edit.target_cell_ids)]
    old_vertices = np.asarray(edit.source_vertex_ids, dtype=np.int64)
    new_vertices = np.asarray(edit.target_vertex_ids, dtype=np.int64)
    old_points = np.asarray(edit.source_points, dtype=np.float64)
    new_points = np.asarray(edit.target_points, dtype=np.float64)
    excluded = set(removed)
    ring: list[int] = []
    if group.ring_vertices is not None:
        candidates = {cell for vertex in new_vertices.ravel().tolist() for cell in incidence.get(vertex, ())}
        created_sets = [set(row.tolist()) for row in new_vertices]
        ring = sorted(
            cell for cell in candidates - excluded
            if any(
                len(created_set.intersection(cells[cell][0].tolist())) >= group.ring_vertices
                for created_set in created_sets
            )
        )
    ring_vertices = np.asarray([cells[cell][0] for cell in ring], dtype=np.int64).reshape((-1, 4))
    ring_points = np.asarray([cells[cell][1] for cell in ring], dtype=np.float64).reshape((-1, 4, 3))
    # Admit both patch meshes and owner spaces before building them.
    budget.reserve(0, (len(created) + len(removed) + 2 * len(ring)) * _patch_cell_bytes(group.element))
    target_mesh = _canonical_mesh(
        np.concatenate((new_vertices, ring_vertices)), np.concatenate((new_points, ring_points)),
        np.asarray(created + ring, dtype=np.int64), block_name,
    )
    source_mesh = _canonical_mesh(
        np.concatenate((old_vertices, ring_vertices)), np.concatenate((old_points, ring_points)),
        np.asarray(removed + ring, dtype=np.int64), block_name,
    )
    target_space = _space(target_mesh, group.spec, budget)
    source_space = _space(source_mesh, group.spec, budget)
    target_keys, source_keys = _dof_keys(target_space), _dof_keys(source_space)
    source_index = {key: position for position, key in enumerate(source_keys)}
    patch = set(created + ring)
    fixed = []
    for position, (degree, vertices, _) in enumerate(target_keys):
        if degree == 3:
            continue
        sharing = set.intersection(*(incidence.get(vertex, set()) for vertex in vertices))
        if sharing - excluded - patch:
            fixed.append(position)
    free = np.setdiff1d(np.arange(len(target_keys)), np.asarray(fixed, dtype=np.int64))
    fixed_index = np.asarray(fixed, dtype=np.int64)
    if any(target_keys[position] not in source_index for position in fixed):
        raise _refuse(
            VolumeMaterialStatus.UNSUPPORTED_SPACE,
            "volume_material_projection fixed DOFs must be old patch DOFs",
        )
    fixed_columns = np.asarray([source_index[target_keys[p]] for p in fixed], dtype=np.int64)
    sizes = (len(target_keys), len(source_keys))
    budget.reserve(0, 8 * (sizes[0] * sizes[0] + 4 * sizes[0] * sizes[1]))
    # Pieces: chain material pieces of the cavity, then the unchanged ring.
    piece_offsets = np.asarray(chain.piece_offsets)
    first, last = int(piece_offsets[index]), int(piece_offsets[index + 1])
    routes = np.asarray(chain.piece_routes)[first:last]
    simplex_offsets = np.asarray(chain.piece_simplex_offsets)[first:last + 1]
    vertex_offsets = np.asarray(chain.piece_vertex_offsets)[index:index + 2]
    # Admit every per-simplex bank before the first gather/repeat/concatenate:
    # the gathered simplices (12 doubles), three routes, the cone map
    # (9 + 3 + 1 doubles) and its mask, for the cavity pieces and the ring.
    total = int(simplex_offsets[-1] - simplex_offsets[0]) + len(ring)
    budget.reserve(0, total * 8 * (12 + 3 + 13 + 1))
    # The edit's shared rounded vertices, gathered by its edit-local rows.
    shared = np.asarray(chain.piece_vertices)[int(vertex_offsets[0]):int(vertex_offsets[1])]
    corners = np.asarray(chain.piece_simplex_vertices)[int(simplex_offsets[0]):int(simplex_offsets[-1])]
    simplices = [shared[corners]]
    counts = np.diff(simplex_offsets)
    target_cells = [np.repeat(routes[:, 0], counts)]
    source_cells = [np.repeat(routes[:, 1], counts)]
    charts = [np.repeat(routes[:, 2], counts)]
    target_order = {int(cell): row for row, cell in enumerate(np.asarray(target_mesh.blocks[0].global_ids).tolist())}
    source_order = {int(cell): row for row, cell in enumerate(np.asarray(source_mesh.blocks[0].global_ids).tolist())}
    target_cells[0] = np.asarray([target_order[created[int(t)]] for t in target_cells[0]], dtype=np.int64)
    source_cells[0] = np.asarray([source_order[removed[int(s)]] for s in source_cells[0]], dtype=np.int64)
    if ring:
        simplices.append(ring_points)
        target_cells.append(np.asarray([target_order[cell] for cell in ring], dtype=np.int64))
        source_cells.append(np.asarray([source_order[cell] for cell in ring], dtype=np.int64))
        charts.append(np.full((len(ring),), -1, dtype=np.int64))
    simplex = np.concatenate(simplices)
    target_cell = np.concatenate(target_cells)
    source_cell = np.concatenate(source_cells)
    chart = np.concatenate(charts)
    start = int(np.asarray(chain.chart_offsets)[index])
    linear = np.tile(np.eye(3), (simplex.shape[0], 1, 1))
    offset = np.zeros((simplex.shape[0], 3), dtype=np.float64)
    jacobian = np.ones((simplex.shape[0],), dtype=np.float64)
    curved = chart >= 0
    linear[curved] = np.asarray(chain.chart_linear)[start + chart[curved]]
    offset[curved] = np.asarray(chain.chart_translation)[start + chart[curved]]
    jacobian[curved] = np.asarray(chain.chart_jacobians)[start + chart[curved]]
    element = group.element
    degree = _polynomial_degree(element)
    # Owner polynomial degrees of the target and source bases; the affine cone
    # map and constant Jacobian weights keep their product degree.
    product = degree + _polynomial_degree(element)
    # The owner rule fixes Q before any bank exists: admit the quadrature
    # points, weights and mapped points, then every basis/pulled bank and
    # their contraction, for all piece simplices at once.
    rule = cubature_rule_data("tetrahedron", product)
    quadrature = int(np.asarray(rule.weights).shape[0])
    value_width = element.value_shape[0] if element.value_shape else 1
    budget.reserve(0, 8 * simplex.shape[0] * quadrature * (
        7 + 4 * element.local_dof_count * (value_width + 1)
    ))
    points, weights, _ = _overlap_quadrature(simplex, product)
    mapped = points @ np.swapaxes(linear, -1, -2) + offset[:, None, :]
    target_routes, target_valid, target_basis, _ = _overlap_basis(target_space, 0, target_cell, mapped)
    source_routes, source_valid, source_basis, source_differential = _overlap_basis(
        source_space, 0, source_cell, points
    )
    match element.mapping:
        case "covariant_piola":
            count, quadrature, width, _ = source_basis.shape
            transposed = np.swapaxes(linear, -1, -2)
            pulled = np.asarray(solve_small_linear(
                SmallLinearSolvePlan(3), transposed,
                np.moveaxis(source_basis, -1, 1).reshape((count, 3, quadrature * width)),
            ).value, dtype=np.float64)
            pulled = np.moveaxis(pulled.reshape((count, 3, quadrature, width)), 1, -1)
            factor = jacobian
        case "contravariant_piola":
            pulled = source_basis @ np.swapaxes(linear, -1, -2)[:, None]
            factor = np.ones_like(jacobian)
        case _:
            pulled = source_basis
            factor = jacobian if group.meaning == "intensive" else np.ones_like(jacobian)
    weighted = weights * factor[:, None]
    local = np.asarray(ein.contract("sq,sqiv,sqjv->sij", weighted, target_basis, pulled))
    mixed = np.zeros(sizes, dtype=np.float64)
    routed = target_valid[:, :, None] & source_valid[:, None, :]
    np.add.at(
        mixed,
        (np.broadcast_to(target_routes[:, :, None], local.shape)[routed],
         np.broadcast_to(source_routes[:, None, :], local.shape)[routed]),
        local[routed],
    )
    measures, triples = _cell_integrals(target_space, 0, mass=True)
    if triples is None:
        raise _refuse(VolumeMaterialStatus.UNSUPPORTED_SPACE, "volume_material_projection requires owner mass entries")
    mass = np.zeros((sizes[0], sizes[0]), dtype=np.float64)
    np.add.at(mass, (triples[0], triples[1]), triples[2])
    measures = np.asarray(measures, dtype=np.float64).reshape((sizes[0], -1))
    constraints: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    names: list[str] = []
    if group.meaning == "conservative-density":
        content = np.asarray(_cell_integrals(source_space, 0, mass=False)[0], dtype=np.float64)
        constraints.append(measures.T)
        targets.append(content.reshape((1, -1)))
        names.append("content")
    elif group.meaning == "material-compatible":
        # Vector content of each pulled source basis function over its pieces.
        accumulated = np.zeros((sizes[1], 3), dtype=np.float64)
        np.add.at(
            accumulated,
            source_routes[source_valid],
            np.asarray(ein.contract("sq,sqjv->sjv", weighted, pulled))[source_valid],
        )
        accumulated = accumulated.T
        constraints.append(measures.T)
        targets.append(accumulated)
        names.append("content")
        if element.mapping == "contravariant_piola" and degree >= 1:
            # Cellwise P_{p-1} divergence moments of the pullback. Admit the
            # cell rule, bases, tests and both moment banks before allocation.
            cell_count = target_mesh.blocks[0].cell_count
            cell_rule = int(np.asarray(cubature_rule_data("tetrahedron", 2 * (degree - 1)).weights).shape[0])
            tested = (degree + 1) * (degree + 2) * (degree + 3) // 6
            local = element.local_dof_count
            budget.reserve(0, 8 * (
                cell_count * cell_rule * (4 + 3 * local + tested)
                + cell_count * tested * (sizes[0] + sizes[1])
                + simplex.shape[0] * (quadrature * tested + tested * local)
            ))
            cell_points = np.asarray(target_mesh.coordinates)[np.asarray(target_mesh.blocks[0].vertices)]
            centers = cell_points.mean(axis=1)
            rule_points, rule_weights, _ = _overlap_quadrature(cell_points, 2 * (degree - 1))
            cell_routes, cell_valid, _, cell_divergence = _overlap_basis(
                target_space, 0, np.arange(cell_points.shape[0]), rule_points
            )
            if cell_divergence is None or source_differential is None:
                raise _refuse(VolumeMaterialStatus.UNSUPPORTED_SPACE, "volume_material_projection Piola fields require owner divergences")
            tests = _differential_tests(rule_points - centers[:, None, :], degree - 1)
            moments = np.asarray(
                ein.contract("cq,cqt,cqn->ctn", rule_weights, tests, cell_divergence[..., 0])
            )
            tested = moments.shape[1]
            divergence = np.zeros((cell_points.shape[0] * tested, sizes[0]), dtype=np.float64)
            for cell in range(cell_points.shape[0]):
                np.add.at(
                    divergence[cell * tested:(cell + 1) * tested].T,
                    cell_routes[cell][cell_valid[cell]],
                    moments[cell][:, cell_valid[cell]].T,
                )
            piece_tests = _differential_tests(mapped - centers[target_cell][:, None, :], degree - 1)
            pulled_moments = np.asarray(
                ein.contract("sq,sqt,sqn->stn", weights, piece_tests, source_differential[..., 0])
            )
            divergence_targets = np.zeros((cell_points.shape[0] * tested, sizes[1]), dtype=np.float64)
            for piece in range(simplex.shape[0]):
                block = slice(int(target_cell[piece]) * tested, (int(target_cell[piece]) + 1) * tested)
                np.add.at(
                    divergence_targets[block].T,
                    source_routes[piece][source_valid[piece]],
                    pulled_moments[piece][:, source_valid[piece]].T,
                )
            constraints.append(divergence)
            targets.append(divergence_targets)
            names.append("commuting")
    count = free.size
    constraint = np.concatenate(constraints) if constraints else np.zeros((0, sizes[0]))
    target_values = np.concatenate(targets) if targets else np.zeros((0, sizes[1]))
    rows = constraint.shape[0]
    # KKT system, the owner SVD factors (three more square banks) and the
    # right side, its solution and composed rows.
    budget.reserve(0, 8 * (4 * (count + rows) ** 2 + 3 * (count + rows) * sizes[1]))
    system = np.zeros((count + rows, count + rows), dtype=np.float64)
    system[:count, :count] = mass[np.ix_(free, free)]
    system[:count, count:] = constraint[:, free].T
    system[count:, :count] = constraint[:, free]
    rhs = np.zeros((count + rows, sizes[1]), dtype=np.float64)
    rhs[:count] = mixed[free]
    rhs[:count, fixed_columns] -= mass[np.ix_(free, fixed_index)]
    rhs[count:] = target_values
    rhs[count:, fixed_columns] -= constraint[:, fixed_index]
    factor = _factor(system, "svd")
    rank = int(np.asarray(factor.rank()))
    retained = np.asarray(factor.singular_values(), dtype=np.float64)[:rank]
    condition = float(retained[0] / retained[-1]) if rank else np.inf
    spectrum = (float(retained[0]) if rank else 0.0, condition)
    if rank < count or not np.isfinite(condition) or condition > _MAXIMUM_LOCAL_CONDITION:
        raise _refuse(
            VolumeMaterialStatus.INFEASIBLE_PROJECTION,
            f"volume_material_projection edit {index} local system must determine its free DOFs; "
            f"got rank {rank} of {count + rows}, condition {condition:.3e}",
        )
    solution, forward = _solve(factor, spectrum, rhs)
    operator = solution[:count]
    # Every operator entry errs by at most its column's 2-norm bound.
    operator_error = float(np.sum(forward))
    if rows:
        achieved = constraint[:, free] @ operator + constraint[:, fixed_index] @ np.eye(sizes[1])[fixed_columns]
        state.constraint_defect = max(
            state.constraint_defect,
            float(np.max(np.abs(achieved - target_values), initial=0.0))
            / max(float(np.max(np.abs(target_values), initial=0.0)), _EPSILON),
        )
        state.constraint_names = tuple(names)
    state.condition += condition
    support = sum(state.rows[key][0].size for key in source_keys)
    budget.reserve(0, 8 * (len(source_keys) + count + 2) * max(1, support))
    if state.storage is None:
        raise RuntimeError("volume_material_projection rows require their live storage owner")
    # The new rows outlive this edit's scope; admit them before they exist.
    state.storage.set_bound(state.storage.bound + count * (16 * max(1, support) + 256))
    composed, columns, composed_errors = _combine(
        [state.rows[key] for key in source_keys],
        [state.errors[key] for key in source_keys],
        operator, operator_error,
    )
    for position, row, error in zip(free.tolist(), composed, composed_errors, strict=True):
        state.rows[target_keys[position]] = _sparse(row, columns)
        state.errors[target_keys[position]] = float(error)


def prepare_volume_material_fe_transfers(
    chain: VolumeMaterialChain,
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    /,
    *,
    semantics: Mapping[str, SourceRealizationFieldSemantics],
    budget: CoordinateEnclosureBudget,
) -> tuple[FiniteElementFieldTransfer, ...]:
    """Compose the chain's local projections into one certified transfer per field.

    Transfers follow ``semantics`` order. Scalar point-value intensive fields
    claim constant preservation; conservative densities claim conservation of
    every source DOF measure; Piola fields claim no nodal invariant and certify
    their content (and H(div) commuting) constraints in their evidence.
    """
    chain.require_bound(source.mesh, source.coordinate_spec, target.mesh, target.coordinate_spec)
    _require_affine_simplices(source, "source", "volume material projection")
    _require_affine_simplices(target, "target", "volume material projection")
    groups = _groups(source, target, semantics)
    block_name = source.mesh.blocks[0].name
    if len(target.mesh.blocks) != 1 or target.mesh.blocks[0].name != block_name:
        raise _refuse(
            VolumeMaterialStatus.UNSUPPORTED_SPACE,
            "volume_material_projection endpoints must share one tetrahedral block",
        )
    with ExitStack() as stack:
        return _compose_transfers(
            chain, source, target, semantics, groups, block_name,
            [stack.enter_context(budget.live_storage()) for _ in groups], budget,
        )


def _compose_transfers(
    chain: VolumeMaterialChain,
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    semantics: Mapping[str, SourceRealizationFieldSemantics],
    groups: tuple[_Group, ...],
    block_name: str,
    storages: list[CoordinateLiveStorage],
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[FiniteElementFieldTransfer, ...]:
    states = [
        _initial_rows(source, source.field_index(group.names[0]), group.spec, storage, budget)
        for group, storage in zip(groups, storages, strict=True)
    ]
    mesh = chain.source_mesh
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cells: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    incidence: dict[int, set[int]] = {}
    for block in mesh.blocks:
        rows = np.asarray(block.vertices, dtype=np.int64)
        budget.reserve(4 * rows.shape[0], 160 * rows.shape[0])
        for identity, row in zip(np.asarray(block.global_ids).tolist(), rows, strict=True):
            cells[int(identity)] = (vertex_ids[row], coordinates[row])
            for vertex in vertex_ids[row].tolist():
                incidence.setdefault(vertex, set()).add(int(identity))
    for index, edit in enumerate(chain.edits):
        with budget.temporary_scope():
            for group, state in zip(groups, states, strict=True):
                _local_projection(group, state, chain, index, cells, incidence, block_name, budget)
        # Composed rows outlive the edit: admit their growth as retained (the
        # edit's temporary admission covered them while they were built).
        # Reconcile each live owner to its rows' actual bytes, after the
        # replaced rows were released and while the new ones stay admitted.
        for state in states:
            if state.storage is not None:
                state.storage.set_bound(_row_bytes(state.rows))
        removed = [int(value) for value in np.asarray(edit.source_cell_ids)]
        for vertex in np.unique(np.asarray(edit.source_vertex_ids)).tolist():
            incidence[vertex].difference_update(removed)
        for cell in removed:
            del cells[cell]
        created_vertices = np.asarray(edit.target_vertex_ids, dtype=np.int64)
        created_points = np.asarray(edit.target_points, dtype=np.float64)
        for position, cell in enumerate(np.asarray(edit.target_cell_ids).tolist()):
            vertices = np.asarray(created_vertices[position], dtype=np.int64)
            cells[int(cell)] = (vertices, np.asarray(created_points[position], dtype=np.float64))
            for vertex in vertices.tolist():
                incidence.setdefault(int(vertex), set()).add(int(cell))
    binding = TransferGeometryBinding(
        cell_geometry_id(source.coordinate_spec),
        cell_geometry_id(target.coordinate_spec),
        "topology-correspondence",
        source_topology_id=source.mesh.topology_id,
        target_topology_id=target.mesh.topology_id,
        coverage_defect=None,
    )
    transfers: dict[str, FiniteElementFieldTransfer] = {}
    for group, state in zip(groups, states, strict=True):
        for name in group.names:
            transfers[name] = _field_transfer(
                group, state, chain, source, target, name, binding, cells, budget,
            )
    # Each transfer's retained CSR and measures were admitted before it was built.
    return tuple(transfers[name] for name in semantics)


def _permuted_cell_rows(
    group: _Group,
    state: _GroupState,
    canonical: FiniteElementDiscretization,
    keys: tuple[_DofKey, ...],
    cells: Mapping[int, tuple[np.ndarray, np.ndarray]],
    budget: CoordinateEnclosureBudget,
    /,
) -> dict[int, tuple[list[_Row], np.ndarray]]:
    """Local rows (and their errors) of target cells whose corners the chain lists in another order.

    Order-dependent interior DOFs of such a cell are not chain DOFs, so the
    owner basis of the chain-ordered cell is interpolated at the target
    cell's owner DOF coordinates: point-value functionals of one polynomial
    space on one physical cell, exact for every corner permutation.
    """
    routes = np.asarray(canonical.dof_maps[0].cell_dofs[0], dtype=np.int64)
    identifiers = np.asarray(canonical.mesh.blocks[0].global_ids, dtype=np.int64)
    pending = [
        cell for cell in range(routes.shape[0])
        if any(keys[int(dof)] not in state.rows for dof in routes[cell].tolist())
    ]
    if not pending:
        return {}
    element = group.element
    if element.mapping != "identity" or element.representation != "point_value":
        raise _refuse(
            VolumeMaterialStatus.MISSING_ANCESTRY,
            "volume_material_projection must carry every target DOF",
        )
    local = group.element.local_dof_count
    # Pending-cell patch space, its basis at every target DOF point, and rows.
    budget.reserve(0, len(pending) * (_patch_cell_bytes(group.element) + 8 * local * local * 4))
    ordered = [cells[int(identifiers[cell])] for cell in pending]
    patch = _space(
        _canonical_mesh(
            np.stack([vertices for vertices, _ in ordered]),
            np.stack([points for _, points in ordered]),
            identifiers[pending], canonical.mesh.blocks[0].name,
        ),
        group.spec, budget,
    )
    patch_keys = _dof_keys(patch)
    coordinates = np.asarray(canonical.dof_maps[0].dof_coordinates, dtype=np.float64)[routes[pending]]
    patch_routes, valid, basis, _ = _overlap_basis(patch, 0, np.arange(len(pending)), coordinates)
    result: dict[int, tuple[list[_Row], np.ndarray]] = {}
    for row, cell in enumerate(pending):
        support = patch_routes[row][valid[row]]
        if any(patch_keys[dof] not in state.rows for dof in support.tolist()):
            raise _refuse(
                VolumeMaterialStatus.MISSING_ANCESTRY,
                "volume_material_projection must carry every target DOF",
            )
        values = basis[row][:, valid[row], 0]
        # Owner basis evaluation errs by a few ulps of each value's magnitude.
        composed, columns, errors = _combine(
            [state.rows[patch_keys[dof]] for dof in support.tolist()],
            [state.errors[patch_keys[dof]] for dof in support.tolist()],
            values, CLAIM_ULPS * _EPSILON * float(np.max(np.sum(np.abs(values), axis=1), initial=0.0)),
        )
        result[cell] = ([_sparse(entry, columns) for entry in composed], errors)
    return result


def _field_transfer(
    group: _Group,
    state: _GroupState,
    chain: VolumeMaterialChain,
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    name: str,
    binding: TransferGeometryBinding,
    cells: Mapping[int, tuple[np.ndarray, np.ndarray]],
    budget: CoordinateEnclosureBudget,
    /,
) -> FiniteElementFieldTransfer:
    si, ti = source.field_index(name), target.field_index(name)
    size = target.dof_maps[ti].global_dof_count
    local = group.element.local_dof_count
    widest = max((routes.size for routes, _ in state.rows.values()), default=1)
    fixed = 64 * (size + source.dof_maps[si].global_dof_count) + 4096
    # Target rows (each at most ``local`` chain rows wide) coexist with their
    # CSR routes, coefficients, coalescing copies, measures and evidence
    # until the transfer exists; admit all before any is built. On success the
    # owner shrinks to the retained transfer and stays open: an atomic
    # live-to-retained handoff with neither a lapse nor a double charge.
    output = CoordinateLiveStorage(budget)
    try:
        entries_upper = size * local * widest
        output.set_bound(entries_upper * (16 + 48) + 256 * size + fixed)
        with budget.temporary_scope():
            cells_count = sum(block.cell_count for block in target.mesh.blocks)
            budget.reserve(0, cells_count * _patch_cell_bytes(group.element))
            assigned, row_errors = _target_rows(group, state, target, ti, cells, budget)
        target_routes = np.concatenate([np.full(assigned[dof][0].shape, dof, dtype=np.int64) for dof in range(size)])
        source_routes = np.concatenate([assigned[dof][0] for dof in range(size)])
        values = np.concatenate([assigned[dof][1] for dof in range(size)])
        transfer = _assemble_transfer(
            group, state, chain, source, target, name, binding,
            assigned, row_errors, target_routes, source_routes, values,
        )
        retained = 48 * int(values.size) + fixed
        del assigned, row_errors, target_routes, source_routes, values
        output.set_bound(retained)
    except BaseException:
        output.close()
        raise
    return transfer


def _target_rows(
    group: _Group,
    state: _GroupState,
    target: FiniteElementDiscretization,
    ti: int,
    cells: Mapping[int, tuple[np.ndarray, np.ndarray]],
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[dict[int, _Row], np.ndarray]:
    """Rows of every actual target DOF over the source DOFs, with their errors."""
    canonical = _space(_endpoint_canonical(target.mesh), group.spec, budget)
    actual_routes, canonical_routes, maps, map_errors = _cell_maps(
        target, ti, canonical, _TransformInverses(), False,
    )
    keys = _dof_keys(canonical)
    permuted = _permuted_cell_rows(group, state, canonical, keys, cells, budget)
    size = target.dof_maps[ti].global_dof_count
    assigned: dict[int, _Row] = {}
    row_errors = np.zeros((size,), dtype=np.float64)
    for cell, (routes, local, matrix) in enumerate(zip(actual_routes, canonical_routes, maps, strict=True)):
        missing = [position for position, dof in enumerate(routes.tolist()) if dof not in assigned]
        if not missing:
            continue
        reordered = permuted.get(cell)
        if reordered is None:
            if any(keys[dof] not in state.rows for dof in local.tolist()):
                raise _refuse(
                    VolumeMaterialStatus.MISSING_ANCESTRY,
                    "volume_material_projection must carry every target DOF",
                )
            local_rows = [state.rows[keys[dof]] for dof in local.tolist()]
            local_errors = [state.errors[keys[dof]] for dof in local.tolist()]
        else:
            local_rows, local_errors = reordered[0], reordered[1].tolist()
        composed, columns, errors = _combine(local_rows, local_errors, matrix, float(map_errors[cell]))
        for position in missing:
            assigned[int(routes[position])] = _sparse(composed[position], columns)
            row_errors[int(routes[position])] = errors[position]
    if len(assigned) != size:
        raise _refuse(VolumeMaterialStatus.MISSING_ANCESTRY, "volume_material_projection must carry every target DOF")
    return assigned, row_errors


def _assemble_transfer(
    group: _Group,
    state: _GroupState,
    chain: VolumeMaterialChain,
    source: FiniteElementDiscretization,
    target: FiniteElementDiscretization,
    name: str,
    binding: TransferGeometryBinding,
    assigned: dict[int, _Row],
    row_errors: np.ndarray,
    target_routes: np.ndarray,
    source_routes: np.ndarray,
    values: np.ndarray,
    /,
) -> FiniteElementFieldTransfer:
    si, ti = source.field_index(name), target.field_index(name)
    size = target.dof_maps[ti].global_dof_count
    primal = coalesced_map(
        target_routes, source_routes, values,
        target_size=size,
        source_size=source.dof_maps[si].global_dof_count,
        properties=OperatorProperties(),
        operator_id=canonical_fingerprint({
            "kind": "volume-material-local-projection",
            "chain": chain.chain_id,
            "source": source.prepared_id,
            "target": target.prepared_id,
            "field": name,
            "semantics": group.meaning,
        }),
    )
    element = group.element
    scalar = element.mapping == "identity" and element.representation == "point_value"
    density = group.meaning == "conservative-density" and scalar
    constants = group.meaning == "intensive" and scalar
    source_measures = (
        np.asarray(_cell_integrals(source, si, mass=False)[0], dtype=np.float64) if density else None
    )
    target_measures = (
        np.asarray(_cell_integrals(target, ti, mass=False)[0], dtype=np.float64) if density else None
    )
    # The action condition is the propagated forward error itself: each row's
    # certified l1 error relative to the claim scale of its absolute row sum,
    # and, for conservation, the measure-weighted error of every column.
    unit = CLAIM_ULPS * _EPSILON
    row_scale = np.maximum(
        np.bincount(
            np.concatenate([np.full(assigned[dof][0].shape, dof) for dof in range(size)]),
            weights=np.abs(values), minlength=size,
        ),
        1.0,
    )
    condition = max(1.0, float(np.max(row_errors / (unit * row_scale), initial=0.0)))
    if source_measures is not None and target_measures is not None:
        total = float(np.sum(np.abs(source_measures)) + np.sum(np.abs(target_measures)))
        condition = max(condition, float(np.abs(target_measures) @ row_errors) / (unit * total))
    if condition > _MAXIMUM_LOCAL_CONDITION:
        # A claim tolerance this coarse would certify nothing.
        raise _refuse(
            VolumeMaterialStatus.INFEASIBLE_PROJECTION,
            f"volume_material_projection composed forward error must meet the conditioning bound; got {condition:.3e}",
        )
    tolerance = condition * unit
    meaning: TransferSemantics
    match element.mapping:
        case "covariant_piola":
            meaning = "covariant-piola"
        case "contravariant_piola":
            meaning = "contravariant-piola"
        case _:
            meaning = "conservative-remap" if density else "l2-projection"
    transfer = FiniteElementTopologyTransfer(
        primal,
        source.mesh.topology_id,
        target.mesh.topology_id,
        preserves_constants=constants,
        conservative=density,
        action_condition=condition,
        semantics=meaning,
        source_measures=source_measures,
        target_measures=target_measures,
    )
    relation = primal.relation
    if not isinstance(relation, EdgeRelation):
        raise TypeError(
            "volume_material_projection.primal must route coalesced target-source edges; "
            f"got {type(relation).__name__}"
        )
    coefficients = np.asarray(primal.coefficients, dtype=np.float64)
    rows = np.asarray(relation.target_indices, dtype=np.int64)
    columns = np.asarray(relation.source_indices, dtype=np.int64)
    defects: dict[str, float] = {
        constraint: state.constraint_defect for constraint in state.constraint_names
    }
    if source_measures is not None and target_measures is not None:
        pulled = np.bincount(columns, weights=coefficients * target_measures[rows], minlength=source_measures.size)
        total = float(np.sum(np.abs(source_measures)) + np.sum(np.abs(target_measures)))
        defects["content"] = float(np.max(np.abs(pulled - source_measures))) / total
    if constants:
        sums = np.bincount(rows, weights=coefficients, minlength=size)
        scale = np.bincount(rows, weights=np.abs(coefficients), minlength=size)
        defects["constants"] = float(np.max(np.abs(sums - 1.0) / np.maximum(scale, 1.0)))
    if not defects:
        defects["local_projection"] = 0.0
    evidence = FiniteElementTransferEvidence(
        defects, tolerance,
        estimates={"local_mass_condition_sum": state.condition},
        bounds={"row_forward_error": float(np.max(row_errors, initial=0.0))},
    )
    return FiniteElementFieldTransfer(
        transfer,
        source.field_spaces[si],
        target.field_spaces[ti],
        binding,
        evidence,
        source_measures=source_measures,
        target_measures=target_measures,
    )


__all__ = ["prepare_volume_material_fe_transfers"]
