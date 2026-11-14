#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded local coupled relocation proposals; native admission remains authoritative."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from fractions import Fraction
from itertools import product
from typing import NamedTuple, TypedDict

import numpy as np

from .._meshcore import TetMeshImplicitFacetRefusal
from ..discretization._periodic_exact import exact_periodic_element
from ..optim import (
    ConvexProgramResult,
    ConvexProgramStatus,
    ConvexSolvePolicy,
    DensePrimalDualQP,
    QuadraticProgram,
    solve_quadratic_program,
)
from . import _tetra_metric as tm
from ._periodic import (
    _PeriodicStencilTransport,
    _prepare_periodic_stencil_transport,
    _transport_periodic_stencil,
    _transport_periodic_vertex_ids,
)


class _CoupledBinding(NamedTuple):
    source_identity: tuple[str, str, str, str]
    generation: int
    edge: tuple[int, int]
    vertices: tuple[int, ...]
    source_cells: tuple[int, ...]


class _CoupledDemand(NamedTuple):
    variables: int
    equalities: int
    inequalities: int
    dimension: int
    input_entries: int
    kkt_entries: int
    workspace_bytes: int


class _CoupledSolve(NamedTuple):
    result: ConvexProgramResult
    policy: ConvexSolvePolicy
    demand: _CoupledDemand
    binding: _CoupledBinding
    primal: np.ndarray | None


class TetraMetricProposalError(RuntimeError):
    """A local model could not be certified; no global infeasibility is implied."""

    def __init__(
        self,
        message: str,
        *,
        result: ConvexProgramResult | None,
        policy: ConvexSolvePolicy,
        demand: _CoupledDemand,
        binding: _CoupledBinding,
    ) -> None:
        super().__init__(message)
        self.result = result
        self.policy = policy
        self.demand = demand
        self.binding = binding


def _admit(
    n: int,
    e: int,
    m: int,
    policy: ConvexSolvePolicy,
    binding: _CoupledBinding,
    *,
    result: ConvexProgramResult | None = None,
) -> _CoupledDemand:
    dimension = n + e + 2 * m
    # Include all canonical dense arrays, including empty bound/equality banks.
    entries = n * n + n + e * n + e + m * n + m + 2 * n
    kkt = dimension * dimension
    workspace = 8 * (kkt + 4 * dimension)
    demand = _CoupledDemand(n, e, m, dimension, entries, kkt, workspace)
    materialization, resources = policy.materialization, policy.resources
    if not isinstance(policy.method, DensePrimalDualQP):
        raise TypeError("Coupled QP requires the dense primal-dual method.")
    if (
        dimension > policy.method.max_kkt_dimension
        or max(entries, kkt) > materialization.max_entries
        or 8 * max(entries, kkt) > materialization.max_bytes
        or workspace > materialization.max_workspace_bytes
        or 8 * kkt > resources.factorization_bytes
        or workspace > resources.workspace_bytes
    ):
        raise TetraMetricProposalError(
            "Coupled QP exceeds its resource policy.",
            result=result,
            policy=policy,
            demand=demand,
            binding=binding,
        )
    return demand


def _solve(
    rows: list[np.ndarray],
    rhs: list[float],
    equalities: list[np.ndarray],
    policy: ConvexSolvePolicy,
    binding: _CoupledBinding,
) -> _CoupledSolve:
    n, e, m = 3 * len(binding.vertices), len(equalities), len(rows)
    demand = _admit(n, e, m, policy, binding)
    program = QuadraticProgram(
        np.eye(n, dtype=np.float64),
        np.zeros(n, dtype=np.float64),
        equality_matrix=np.asarray(equalities, dtype=np.float64).reshape((e, n)),
        equality_rhs=np.zeros(e, dtype=np.float64),
        inequality_matrix=np.asarray(rows, dtype=np.float64).reshape((m, n)),
        inequality_rhs=np.asarray(rhs, dtype=np.float64),
        convexity_evidence="Identity Hessian is positive definite.",
    )
    if program.linear.dtype != np.dtype(np.float64):
        raise TetraMetricProposalError(
            "Coupled QP requires binary64 solver inputs.",
            result=None,
            policy=policy,
            demand=demand,
            binding=binding,
        )
    result = solve_quadratic_program(program, policy=policy)
    status = int(result.status)
    if status == int(ConvexProgramStatus.PRIMAL_INFEASIBLE) and bool(
        result.certificate.dual_ray_valid
    ):
        return _CoupledSolve(result, policy, demand, binding, None)
    if status != int(ConvexProgramStatus.OPTIMAL) or not bool(result.successful):
        raise TetraMetricProposalError(
            "Coupled QP did not produce an audited optimum.",
            result=result,
            policy=policy,
            demand=demand,
            binding=binding,
        )
    primal = np.asarray(result.primal, dtype=np.float64)
    if primal.shape != (n,) or not np.all(np.isfinite(primal)):
        raise TetraMetricProposalError(
            "Coupled QP returned an invalid primal.",
            result=result,
            policy=policy,
            demand=demand,
            binding=binding,
        )
    return _CoupledSolve(result, policy, demand, binding, primal)


def _rung_limit(quanta: int, *, short: bool) -> float:
    """Round a proposed rung into the corresponding actual band quantum."""
    boundary = tm._LOWER if short else tm._UPPER
    direction = np.inf if short else -np.inf
    value = float(
        boundary * np.exp((-1 if short else 1) * quanta * tm._VIOLATION_QUANTUM)
    )
    while np.isfinite(value) and value > 0.0:
        band = tm._band_violations(np.asarray([value], dtype=np.float64))
        component = band[1 if short else 0]
        if not component or component[0] <= quanta:
            return value
        following = float(np.nextafter(value, direction))
        if following == value:
            break
        value = following
    raise FloatingPointError("Could not represent the requested metric violation rung.")


class _Proposal(TypedDict):
    vertex: int
    point: np.ndarray
    metric: np.ndarray
    stencil: dict[int, Fraction]
    current_stencil: dict[int, Fraction] | None
    star_cell_ids: tuple[int, ...]
    star_vertices: frozenset[int]
    cavity: frozenset[int]
    block: int
    region: int


class _VertexDetail(NamedTuple):
    vertex: int
    row: int
    simplices: tuple[tuple[int, ...], ...] | None
    star: tuple[int, ...]
    star_ids: tuple[int, ...]
    block: int
    region: int
    source_cells: tuple[tm._SourceTetRecord, ...]


class _Neighbor(NamedTuple):
    detail: _VertexDetail
    unit_direction: np.ndarray


class _ModelAttempt(NamedTuple):
    applied: int
    neighbors: tuple[_Neighbor, ...]


class _Search(NamedTuple):
    state: tm._NativeState
    prepared: tm._RelocationPreparation
    cells: np.ndarray
    edges: tuple[tuple[int, int], ...]
    measured: np.ndarray
    before: tuple[tuple[int, ...], tuple[int, ...]]
    orbits: dict[int, tuple[int, ...]]
    source_rows: dict[int, int]
    long_quanta: np.ndarray
    short_quanta: np.ndarray
    carriers: dict[tuple[int, int], tuple[int, ...]]
    source_transport: _PeriodicStencilTransport | None
    current_transport: _PeriodicStencilTransport | None
    native_by_target: dict[int, int]


def _active_plane(
    state: tm._NativeState,
    prepared: tm._RelocationPreparation,
    vertices: tuple[int, ...],
    proposals: list[_Proposal],
    refusal: TetMeshImplicitFacetRefusal,
) -> tuple[int, np.ndarray, tuple[int, ...]] | None:
    identity = refusal.identity
    if (
        not identity.identity_bound
        or int(identity.generation) != int(state.snapshot.generation)
        or (
            identity.source_id,
            identity.source_revision,
            identity.source_map_id,
            identity.frame_id,
        )
        != state.source_identity
    ):
        return None
    snapshot, top = state.snapshot, prepared.topology
    ids = np.asarray(snapshot.facet_ids, dtype=np.int64)
    found = np.flatnonzero(ids == int(refusal.facet_id))
    if len(found) != 1:
        return None
    row = int(found[0])
    source_group = state.source_face_group_rows.get(int(snapshot.face_sources[row]))
    if (
        source_group != int(refusal.facet_source)
        or source_group not in state.implicit_groups
    ):
        return None
    face = tuple(int(snapshot.vertex_ids[i]) for i in snapshot.faces[row])
    if tuple(int(v) for v in refusal.vertex_ids) != face or int(
        snapshot.facet_orientations[row]
    ) != int(refusal.orientation):
        return None
    proposed = {int(p["vertex"]): p["point"] for p in proposals}
    evaluated = np.asarray(
        [proposed.get(v, prepared.points[top.vertex_row[v]]) for v in face],
        dtype=np.float64,
    )
    if not np.array_equal(evaluated, refusal.points):
        return None
    owners = top.face_cells.get(tuple(sorted(face)), ())
    if len(owners) != 1:
        return None
    owner = top.cell_vertex_ids[owners[0]]
    opposite = next(int(v) for v in owner if int(v) not in face)
    a, b, c = (prepared.points[top.vertex_row[v]] for v in face)
    normal = np.cross(b - a, c - a)
    dot = float(normal @ (prepared.points[top.vertex_row[opposite]] - a))
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm == 0.0 or dot == 0.0:
        return None
    normal *= (-1.0 if dot > 0 else 1.0) / norm
    moved = tuple(v for v in vertices if v in proposed and v in face)
    return (int(refusal.facet_id), normal, moved) if moved else None


def _active_rows(
    binding: _CoupledBinding,
    plane: tuple[int, np.ndarray, tuple[int, ...]],
    refusal: TetMeshImplicitFacetRefusal,
) -> Iterator[tuple[tuple[int, tuple[int, ...]], np.ndarray]]:
    """Use a bound rejecting-piece sample only to guide the next proposal."""
    facet, normal, affected = plane
    n = 3 * len(binding.vertices)
    barycentric = refusal.distance_barycentric
    if refusal.distance_refusals > 0 and barycentric is not None:
        face = tuple(refusal.vertex_ids)
        vector = np.zeros(n, dtype=np.float64)
        for vertex in affected:
            position = face.index(vertex)
            k = binding.vertices.index(vertex)
            vector[3 * k : 3 * k + 3] = -barycentric[position] * normal
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            # Separate vertex halfspaces forbid compensating facet motion.
            # Native admission, not this sampled row, certifies the candidate.
            yield (facet, affected), vector / norm
        return
    for vertex in affected:
        vector = np.zeros(n, dtype=np.float64)
        k = binding.vertices.index(vertex)
        vector[3 * k : 3 * k + 3] = -normal
        yield (facet, (vertex,)), vector


def _prepare_search(state: tm._NativeState) -> _Search:
    prepared = tm._prepare_relocation_geometry(state)
    edges = tuple(prepared.topology.edge_cells)
    measured = np.asarray(
        [not prepared.protected[edge] for edge in edges], dtype=np.bool_
    )
    if not np.any(measured):
        measured[:] = True
    periodic = state.target_mesh.periodic_topology is not None
    long_quanta, short_quanta = tm._band_violation_rows(prepared.lengths[:, None])
    return _Search(
        state,
        prepared,
        np.asarray(state.snapshot.tetrahedra, dtype=np.int64),
        edges,
        measured,
        tm._band_violations(prepared.lengths[measured]),
        tm._periodic_vertex_orbit_members(state) if periodic else {},
        {int(v): i for i, v in enumerate(state.source_cell_ids)},
        long_quanta[:, 0],
        short_quanta[:, 0],
        {},
        _prepare_periodic_stencil_transport(state.source_mesh, state.source_exact_points)
        if periodic
        else None,
        _prepare_periodic_stencil_transport(state.target_mesh) if periodic else None,
        {target: native for native, target in state.vertex_target_ids.items()}
        if periodic
        else {},
    )


def _source_cells(
    search: _Search,
    vertex: int,
    simplices: tuple[tuple[int, ...], ...] | None,
    block: int,
    region: int,
) -> tuple[tm._SourceTetRecord, ...]:
    state = search.state
    image = tm._stencil_image(state, state.vertex_stencils[vertex])
    records: list[tm._SourceTetRecord] = []
    for record in state.source_records:
        row = search.source_rows[record[0]]
        if (
            state.source_cell_blocks[row] != block
            or state.source_cell_regions[row] != region
            or tm._locate_source_weights(image, (record,)) is None
        ):
            continue
        support = tm._source_cell_carrier(state, record, simplices)
        if support is not None:
            search.carriers[vertex, record[0]] = support
            records.append(record)
    return tuple(records)


def _vertex_details(
    search: _Search, vertices: tuple[int, ...]
) -> tuple[_VertexDetail, ...] | None:
    state, prepared = search.state, search.prepared
    snapshot, topology = state.snapshot, prepared.topology
    cell_ids = np.asarray(snapshot.cell_ids, dtype=np.int64)
    regions = np.asarray(snapshot.tetrahedron_regions, dtype=np.int64)
    dimensions = np.asarray(snapshot.vertex_dimensions, dtype=np.int8)
    protected = np.asarray(snapshot.protected_vertices, dtype=np.bool_)
    details: list[_VertexDetail] = []
    for vertex in vertices:
        row = topology.vertex_row[vertex]
        if vertex in state.fixed_vertices or protected[row]:
            return None
        implicit = tm._implicit_only_vertex(state, prepared.records, vertex)
        simplices = (
            None if implicit else tm._carrier_simplices(state, prepared.records, vertex)
        )
        if (
            simplices is None and not implicit and dimensions[row] != 3
        ) or simplices == ():
            return None
        star = tuple(topology.vertex_cells[vertex])
        star_ids = tuple(int(cell_ids[cell]) for cell in star)
        blocks = {state.cell_blocks[cell] for cell in star_ids}
        cell_regions = {int(regions[cell]) for cell in star}
        if not star or len(blocks) != 1 or len(cell_regions) != 1:
            return None
        candidates = _source_cells(
            search, vertex, simplices, next(iter(blocks)), next(iter(cell_regions))
        )
        if not candidates:
            return None
        details.append(
            _VertexDetail(
                vertex,
                row,
                simplices,
                star,
                star_ids,
                next(iter(blocks)),
                next(iter(cell_regions)),
                candidates,
            )
        )
    return tuple(details)


def _carrier_normals(
    state: tm._NativeState, detail: _VertexDetail
) -> tuple[np.ndarray, ...] | None:
    if detail.simplices is None:
        return ()
    corners = tuple(
        state.source_exact_points[state.source_vertex_rows[v]]
        for v in detail.simplices[0]
    )
    _, gradients = tm._source_carrier_frame(state, detail.simplices[0])
    if gradients is None:
        return None
    axes = tuple(
        tuple(corner[i] - corners[0][i] for i in range(3)) for corner in corners[1:]
    )
    if len(axes) == 2:
        a, b = axes
        exact = (
            (
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ),
        )
    else:
        axis = axes[0]
        pivot = max(range(3), key=lambda i: abs(axis[i]))
        exact = tuple(
            tuple(
                -axis[i] if j == pivot else axis[pivot] if j == i else Fraction(0)
                for j in range(3)
            )
            for i in range(3)
            if i != pivot
        )
    normals = []
    for row in exact:
        scale = max(abs(value) for value in row)
        vector = np.asarray([float(value / scale) for value in row], dtype=np.float64)
        normals.append(vector / np.linalg.norm(vector))
    return tuple(normals)


def _source_affine_rows(
    search: _Search, detail: _VertexDetail, record: tm._SourceTetRecord
) -> tuple[np.ndarray, list[Fraction]]:
    point = tuple(Fraction(float(value)) for value in search.prepared.points[detail.row])
    if detail.simplices is None:
        affine = np.asarray(
            [[float(Fraction(a, record[7])) for a in row[:3]] for row in record[6]],
            dtype=np.float64,
        )
        tail = [
            sum(
                (Fraction(a, record[7]) * z for a, z in zip(row[:3], point, strict=True)),
                Fraction(row[3], record[7]),
            )
            for row in record[6]
        ]
    else:
        support = search.carriers[detail.vertex, record[0]]
        origin, exact = tm._source_carrier_frame(search.state, support)
        if exact is None:
            raise RuntimeError("Selected source carrier lost its exact affine frame.")
        affine = np.asarray(
            [[float(value) for value in row] for row in exact], dtype=np.float64
        )
        offset = tuple(value - base for value, base in zip(point, origin, strict=True))
        tail = [
            sum(
                (value * delta for value, delta in zip(row, offset, strict=True)),
                Fraction(0),
            )
            for row in exact
        ]
    return np.vstack((-np.sum(affine, axis=0), affine)), [
        1 - sum(tail, Fraction(0)),
        *tail,
    ]


def _source_rows(
    n: int,
    search: _Search,
    details: tuple[_VertexDetail, ...],
    selected: tuple[tm._SourceTetRecord, ...],
    normals: tuple[tuple[np.ndarray, ...], ...],
) -> tuple[list[np.ndarray], list[float], list[np.ndarray]]:
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    equality: list[np.ndarray] = []
    for k, (detail, record) in enumerate(zip(details, selected, strict=True)):
        gradients, weights = _source_affine_rows(search, detail, record)
        for grad, weight in zip(gradients, weights, strict=True):
            vector = np.zeros(n, dtype=np.float64)
            vector[3 * k : 3 * k + 3] = -grad
            rows.append(vector)
            rhs.append(float(weight))
        for normal in normals[k]:
            vector = np.zeros(n, dtype=np.float64)
            vector[3 * k : 3 * k + 3] = normal
            equality.append(vector)
    return rows, rhs, equality


def _orbit_rows(search: _Search, vertices: tuple[int, ...], n: int) -> list[np.ndarray]:
    state = search.state
    topology = state.target_mesh.periodic_topology
    if topology is None:
        return []
    rows = []
    for k, vertex in enumerate(vertices):
        first = vertices.index(search.orbits.get(vertex, (vertex,))[0])
        if first == k:
            continue
        shifts = (
            state.periodic_vertex_windings[vertex]
            - state.periodic_vertex_windings[vertices[first]]
        )
        transform = exact_periodic_element(topology.cell, shifts)
        for axis in range(3):
            vector = np.zeros(n, dtype=np.float64)
            vector[3 * k + axis] = 1
            vector[3 * first : 3 * first + 3] = [
                -float(value) for value in transform[axis][:3]
            ]
            rows.append(vector)
    return rows


def _edge_limits(search: _Search, index: int, selected: int) -> tuple[float, float]:
    # Lower the selected edge's rung without worsening higher-priority rungs.
    # A global-only bound can already admit zero on a less-bad selected edge.
    target_long = int(search.long_quanta[selected])
    target_short = int(search.short_quanta[selected])
    if target_long:
        long = (
            target_long - 1
            if index == selected
            else max(int(search.long_quanta[index]), target_long - 1)
        )
        short = search.before[1][0] if search.before[1] else 0
    else:
        long = int(search.long_quanta[index])
        short = (
            target_short - 1
            if index == selected
            else max(int(search.short_quanta[index]), target_short - 1)
        )
    return _rung_limit(short, short=True), _rung_limit(long, short=False)


def _edge_rows(
    search: _Search,
    vertices: tuple[int, ...],
    incident: list[int],
    selected_edge: tuple[int, int],
    rows: list[np.ndarray],
    rhs: list[float],
) -> None:
    prepared = search.prepared
    points, metrics, topology = prepared.points, prepared.metrics, prepared.topology
    selected = next(index for index in incident if search.edges[index] == selected_edge)
    for index in incident:
        short_limit, long_limit = _edge_limits(search, index, selected)
        first, second = search.edges[index]
        a, b = topology.vertex_row[first], topology.vertex_row[second]
        delta = points[a] - points[b]
        matrix = 0.5 * (metrics[a] + metrics[b])
        gradient = matrix @ delta / float(np.sqrt(delta @ matrix @ delta))
        vector = np.zeros(3 * len(vertices), dtype=np.float64)
        for k, vertex in enumerate(vertices):
            if vertex == first:
                vector[3 * k : 3 * k + 3] += gradient
            if vertex == second:
                vector[3 * k : 3 * k + 3] -= gradient
        rows.extend((-vector, vector))
        rhs.extend(
            (
                float(prepared.lengths[index] - short_limit),
                float(long_limit - prepared.lengths[index]),
            )
        )


def _model_rows(
    search: _Search,
    details: tuple[_VertexDetail, ...],
    selected: tuple[tm._SourceTetRecord, ...],
    incident: list[int],
    binding: _CoupledBinding,
    policy: ConvexSolvePolicy,
) -> tuple[list[np.ndarray], list[float], list[np.ndarray]] | None:
    normals = tuple(
        _carrier_normals(search.state, detail)
        if search.orbits.get(detail.vertex, (detail.vertex,))[0] == detail.vertex
        else ()
        for detail in details
    )
    if any(group is None for group in normals):
        return None
    present = tuple(group for group in normals if group is not None)
    vertices = binding.vertices
    orbit_count = sum(
        3
        for k, vertex in enumerate(vertices)
        if search.orbits and vertices.index(search.orbits.get(vertex, (vertex,))[0]) != k
    )
    n = 3 * len(vertices)
    _admit(
        n,
        sum(len(group) for group in present) + orbit_count,
        sum(
            len(search.carriers[detail.vertex, record[0]])
            for detail, record in zip(details, selected, strict=True)
        )
        + 2 * len(incident),
        policy,
        binding,
    )
    rows, rhs, equality = _source_rows(n, search, details, selected, present)
    equality.extend(_orbit_rows(search, vertices, n))
    _edge_rows(search, vertices, incident, binding.edge, rows, rhs)
    return rows, rhs, equality


def _periodic_stencil(
    search: _Search, detail: _VertexDetail, carrier: np.ndarray
) -> dict[int, Fraction] | None:
    top, points = search.prepared.topology, search.prepared.points
    neighbors = sorted(
        {
            int(other)
            for cell in detail.star
            for other in top.cell_vertex_ids[cell]
            if int(other) != detail.vertex
        }
    )
    identifiers = (detail.vertex, *neighbors)
    local_rows = [top.vertex_row[vertex] for vertex in identifiers]
    row_map = {row: index for index, row in enumerate(local_rows)}
    local_cells = np.asarray(
        [[row_map[int(x)] for x in search.cells[cell]] for cell in detail.star],
        dtype=np.int64,
    )
    return tm._current_star_stencil(
        points[local_rows],
        local_cells,
        np.asarray(identifiers, dtype=np.int64),
        np.asarray(detail.star_ids, dtype=np.int64),
        {vertex: index for index, vertex in enumerate(identifiers)},
        carrier,
    )


def _stationary_carrier(
    state: tm._NativeState,
    detail: _VertexDetail,
    goal: np.ndarray,
    current: np.ndarray,
) -> bool:
    if np.array_equal(goal, current):
        return True
    if detail.simplices is None:
        return False
    simplex = detail.simplices[0]
    origin, gradients = tm._source_carrier_frame(state, simplex)
    if gradients is None:
        return False
    query = (Fraction(float(goal[0])), Fraction(float(goal[1])), Fraction(float(goal[2])))
    offset = tuple(value - base for value, base in zip(query, origin, strict=True))
    tail = tuple(
        sum(
            (value * delta for value, delta in zip(row, offset, strict=True)), Fraction(0)
        )
        for row in gradients
    )
    corners = tuple(
        state.source_exact_points[state.source_vertex_rows[v]] for v in simplex[1:]
    )
    # The carrier group has one affine hull. Checking its projection once also
    # handles a point in a different triangle of the same declared plane.
    return all(
        float(
            origin[axis]
            + sum(
                (
                    weight * (corner[axis] - origin[axis])
                    for weight, corner in zip(tail, corners, strict=True)
                ),
                Fraction(0),
            )
        )
        == current[axis]
        for axis in range(3)
    )


def _proposal(
    search: _Search,
    detail: _VertexDetail,
    record: tm._SourceTetRecord,
    stencil: dict[int, Fraction],
    carrier: np.ndarray,
    current: dict[int, Fraction] | None,
) -> _Proposal | None:
    state, prepared = search.state, search.prepared
    if (
        np.array_equal(carrier, prepared.points[detail.row])
        or tm._locate_source_weights(tm._stencil_image(state, stencil), (record,)) is None
    ):
        return None
    return _Proposal(
        vertex=detail.vertex,
        point=carrier,
        metric=tm._source_metric_from_stencil(
            state.source_metric,
            state.source_vertex_rows,
            stencil,
            source_metric_log=state.source_metric_log,
        ),
        stencil=stencil,
        current_stencil=current,
        star_cell_ids=detail.star_ids,
        star_vertices=frozenset(
            int(vertex)
            for cell in detail.star
            for vertex in prepared.topology.cell_vertex_ids[cell]
        ),
        cavity=frozenset(detail.star_ids),
        block=detail.block,
        region=detail.region,
    )


def _lift_orbit(
    search: _Search,
    representative: int,
    details: dict[int, _VertexDetail],
    selected: dict[int, tm._SourceTetRecord],
    stencil: dict[int, Fraction],
    carrier: np.ndarray,
) -> list[_Proposal] | None:
    source_transport, current_transport = (
        search.source_transport,
        search.current_transport,
    )
    if source_transport is None or current_transport is None:
        raise RuntimeError("Periodic carrier lifting lost its bound source transports.")
    state = search.state
    current = _periodic_stencil(search, details[representative], carrier)
    if current is None:
        return None
    target_stencil = {
        state.vertex_target_ids[vertex]: weight for vertex, weight in current.items()
    }
    proposals: list[_Proposal] = []
    for vertex in search.orbits[representative]:
        shifts = (
            state.periodic_vertex_windings[vertex]
            - state.periodic_vertex_windings[representative]
        )
        source_image = _transport_periodic_stencil(
            source_transport, stencil, carrier, shifts
        )
        current_image = _transport_periodic_stencil(
            current_transport, target_stencil, carrier, shifts
        )
        if (
            source_image is None
            or current_image is None
            or not np.array_equal(source_image[1], current_image[1])
        ):
            return None
        current_stencil = dict(
            sorted(
                (search.native_by_target[target], weight)
                for target, weight in current_image[0].items()
            )
        )
        proposal = _proposal(
            search,
            details[vertex],
            selected[vertex],
            source_image[0],
            source_image[1],
            current_stencil,
        )
        # A rounded stationary member cannot be published as a moved lift.
        if proposal is None:
            return None
        proposals.append(proposal)
    return proposals


def _quantize_orbit(
    search: _Search,
    representative: int,
    details: dict[int, _VertexDetail],
    selected: dict[int, tm._SourceTetRecord],
    goal: np.ndarray,
) -> list[_Proposal] | None:
    state = search.state
    detail, record = details[representative], selected[representative]
    local_state = replace(state, source_records=(record,))
    simplices = (
        None
        if detail.simplices is None
        else (search.carriers[representative, record[0]],)
    )
    for stencil, carrier in tm._quantized_relocation_candidates(
        local_state,
        simplices,
        goal,
        search.prepared.points[detail.row],
        frames=state.source_supports,
        fine_first=True,
    ):
        if search.orbits:
            proposals = _lift_orbit(
                search, representative, details, selected, stencil, carrier
            )
            if proposals is not None:
                return proposals
        else:
            proposal = _proposal(search, detail, record, stencil, carrier, None)
            if proposal is not None:
                return [proposal]
    return None


def _quantize(
    search: _Search,
    details: tuple[_VertexDetail, ...],
    selected: tuple[tm._SourceTetRecord, ...],
    primal: np.ndarray,
) -> tuple[list[_Proposal], np.ndarray, np.ndarray] | None:
    state, prepared = search.state, search.prepared
    by_vertex = {detail.vertex: detail for detail in details}
    records = {
        detail.vertex: record for detail, record in zip(details, selected, strict=True)
    }
    new_points, new_metrics = prepared.points.copy(), prepared.metrics.copy()
    proposals: list[_Proposal] = []
    for index, detail in enumerate(details):
        if search.orbits.get(detail.vertex, (detail.vertex,))[0] != detail.vertex:
            continue
        goal = prepared.points[detail.row] + primal[3 * index : 3 * index + 3]
        if _stationary_carrier(state, detail, goal, prepared.points[detail.row]):
            continue
        orbit = _quantize_orbit(search, detail.vertex, by_vertex, records, goal)
        if orbit is None:
            return None
        for proposal in orbit:
            row = by_vertex[proposal["vertex"]].row
            new_points[row] = proposal["point"]
            new_metrics[row] = proposal["metric"]
        proposals.extend(orbit)
    return (proposals, new_points, new_metrics) if proposals else None


def _candidate_admissible(
    search: _Search,
    details: tuple[_VertexDetail, ...],
    points: np.ndarray,
    metrics: np.ndarray,
) -> bool:
    changed = sorted({cell for detail in details for cell in detail.star})
    if not tm._floor_oriented(points[search.cells[changed]]):
        return False
    lengths = tm._metric_lengths(points, metrics, search.prepared.edge_rows)
    if not tm._band_violations(lengths[search.measured]) < search.before:
        return False
    quality = tm._tetra_quality(points, metrics, search.cells[changed])
    return bool(np.all(np.isfinite(quality)) and np.all(quality > 0))


def _dual_neighbor_scores(
    search: _Search,
    incident: list[int],
    binding: _CoupledBinding,
    base_rows: list[np.ndarray],
    solved: _CoupledSolve,
) -> tuple[dict[int, list[tuple[float, int]]], dict[int, np.ndarray]]:
    # Source barycentric rows precede lower/upper edge pairs. Native-refusal
    # cuts follow base_rows and must never masquerade as fixed-neighbor evidence.
    offset = len(base_rows) - 2 * len(incident)
    ray = np.asarray(solved.result.certificate.inequality_dual_ray, dtype=np.float64)
    if ray.shape != (solved.demand.inequalities,) or not np.all(np.isfinite(ray)):
        raise TetraMetricProposalError(
            "Coupled QP returned an invalid dual ray.",
            result=solved.result,
            policy=solved.policy,
            demand=solved.demand,
            binding=binding,
        )
    metric_ray = ray[offset : len(base_rows)]
    scale = float(np.max(np.abs(metric_ray), initial=0.0))
    scores: dict[int, list[tuple[float, int]]] = {
        vertex: [] for vertex in binding.vertices
    }
    directions: dict[int, np.ndarray] = {}
    if scale == 0.0:
        return scores, directions
    positions = {vertex: k for k, vertex in enumerate(binding.vertices)}
    for k, index in enumerate(incident):
        first, second = search.edges[index]
        if (first in positions) == (second in positions):
            continue
        seed, neighbor = (first, second) if first in positions else (second, first)
        lower, upper = float(metric_ray[2 * k]), float(metric_ray[2 * k + 1])
        score = max(lower, upper)
        if score <= 0.0:
            continue
        scores[seed].append((score, neighbor))
        start = 3 * positions[seed]
        # The omitted endpoint has the opposite edge coefficient. Moving
        # against its dual-weighted coefficient releases the recorded blocker.
        direction = (upper / scale - lower / scale) * base_rows[offset + 2 * k + 1][
            start : start + 3
        ]
        if neighbor in directions:
            directions[neighbor] += direction
        else:
            directions[neighbor] = direction
    return scores, directions


def _blocking_neighbors(
    search: _Search,
    incident: list[int],
    binding: _CoupledBinding,
    base_rows: list[np.ndarray],
    solved: _CoupledSolve,
) -> tuple[_Neighbor, ...]:
    if (
        len(binding.vertices) != 2
        or search.state.target_mesh.periodic_topology is not None
    ):
        return ()
    scores, directions = _dual_neighbor_scores(
        search, incident, binding, base_rows, solved
    )
    chosen: dict[int, _Neighbor] = {}
    target_ids = search.state.vertex_target_ids
    for seed in binding.vertices:
        for _, vertex in sorted(
            scores[seed], key=lambda item: (-item[0], target_ids[item[1]])
        ):
            if vertex in chosen:
                break
            direction = directions[vertex]
            scale = float(np.max(np.abs(direction)))
            if scale == 0.0:
                continue
            details = _vertex_details(search, (vertex,))
            if details is None:
                continue
            scaled = direction / scale
            chosen[vertex] = _Neighbor(details[0], scaled / np.linalg.norm(scaled))
            break
    return tuple(chosen[vertex] for vertex in sorted(chosen, key=target_ids.__getitem__))


def _search_model(
    search: _Search,
    details: tuple[_VertexDetail, ...],
    selected: tuple[tm._SourceTetRecord, ...],
    incident: list[int],
    binding: _CoupledBinding,
    policy: ConvexSolvePolicy,
    counters: dict[str, int],
) -> _ModelAttempt:
    model = _model_rows(search, details, selected, incident, binding, policy)
    if model is None:
        return _ModelAttempt(0, ())
    base_rows, rhs, equality = model
    active: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}
    n = 3 * len(binding.vertices)
    while True:
        solved = _solve(
            base_rows + list(active.values()),
            rhs + [0.0] * len(active),
            equality,
            policy,
            binding,
        )
        if solved.primal is None:
            return _ModelAttempt(
                0, _blocking_neighbors(search, incident, binding, base_rows, solved)
            )
        candidate = _quantize(search, details, selected, solved.primal)
        if candidate is None or not _candidate_admissible(
            search, details, candidate[1], candidate[2]
        ):
            return _ModelAttempt(0, ())
        proposals = candidate[0]
        status = tm._apply_relocation_batch(search.state, proposals)
        if status is tm.TetMeshEditStatus.APPLIED:
            counters["relocations"] += len(proposals)
            return _ModelAttempt(len(proposals), ())
        tm._raise_fatal_edit_status(status, "coupled relocation")
        tm._count_rejection(status, counters, len(proposals))
        refusal = search.state.native.implicit_facet_refusal
        if (
            refusal is None
            or refusal.status != status
            or refusal.transversality_refusals + refusal.distance_refusals <= 0
        ):
            return _ModelAttempt(0, ())
        plane = _active_plane(
            search.state, search.prepared, binding.vertices, proposals, refusal
        )
        if plane is None:
            return _ModelAttempt(0, ())
        fresh = False
        for key, vector in _active_rows(binding, plane, refusal):
            if key not in active:
                _admit(
                    n,
                    len(equality),
                    len(base_rows) + len(active) + 1,
                    policy,
                    binding,
                    result=solved.result,
                )
                active[key] = vector
                fresh = True
        if not fresh:
            return _ModelAttempt(0, ())


def _search_neighbor_patch(
    search: _Search,
    details: tuple[_VertexDetail, ...],
    selected: tuple[tm._SourceTetRecord, ...],
    binding: _CoupledBinding,
    policy: ConvexSolvePolicy,
    counters: dict[str, int],
    neighbors: tuple[_Neighbor, ...],
    index: int,
) -> int:
    by_vertex = {detail.vertex: detail for detail in details}
    by_source = dict(zip(binding.vertices, selected, strict=True))
    for neighbor in neighbors:
        detail = neighbor.detail
        by_vertex[detail.vertex] = detail
        # Added variables have one validated preferred carrier, not an
        # additional Cartesian source-choice search. Seed choices stay exact.
        by_source[detail.vertex] = _ordered_source_cells(
            search, detail, binding.edge, index, unit_direction=neighbor.unit_direction
        )[0]
    vertices = tuple(sorted(by_vertex, key=search.state.vertex_target_ids.__getitem__))
    patch_incident = [
        i
        for i, edge in enumerate(search.edges)
        if search.measured[i] and any(vertex in edge for vertex in vertices)
    ]
    patch_selected = tuple(by_source[vertex] for vertex in vertices)
    patch_binding = _CoupledBinding(
        search.state.source_identity,
        int(search.state.snapshot.generation),
        binding.edge,
        vertices,
        tuple(record[0] for record in patch_selected),
    )
    return _search_model(
        search,
        tuple(by_vertex[vertex] for vertex in vertices),
        patch_selected,
        patch_incident,
        patch_binding,
        policy,
        counters,
    ).applied


def _ordered_source_cells(
    search: _Search,
    detail: _VertexDetail,
    edge: tuple[int, int],
    index: int,
    *,
    unit_direction: np.ndarray | None = None,
) -> tuple[tm._SourceTetRecord, ...]:
    """Prefer a valid carrier along the seed pull or a dual-witness direction."""
    prepared = search.prepared
    first, second = (prepared.topology.vertex_row[vertex] for vertex in edge)
    lower, upper = _edge_limits(search, index, index)
    target = lower if search.short_quanta[index] else upper
    pull = (
        0.5
        * (target / prepared.lengths[index] - 1.0)
        * (prepared.points[first] - prepared.points[second])
    )
    goal = prepared.points[detail.row]
    if unit_direction is not None:
        goal = goal + np.linalg.norm(pull) * unit_direction
    elif detail.vertex == edge[0]:
        goal = goal + pull
    elif detail.vertex == edge[1]:
        goal = goal - pull
    query = (Fraction(float(goal[0])), Fraction(float(goal[1])), Fraction(float(goal[2])))
    return tuple(
        sorted(
            detail.source_cells,
            key=lambda record: (
                tm._locate_source_weights(query, (record,)) is None,
                record[0],
            ),
        )
    )


def _orbit_source_choices(
    search: _Search,
    representative: int,
    choices: dict[int, tuple[tm._SourceTetRecord, ...]],
) -> tuple[dict[int, tm._SourceTetRecord], ...]:
    transport = search.source_transport
    if transport is None:
        raise RuntimeError("Periodic source selection has no bound stencil transport.")
    members = search.orbits[representative]
    available = {
        vertex: {
            tuple(sorted(search.carriers[vertex, record[0]])): record
            for record in reversed(choices[vertex])
        }
        for vertex in members
    }
    alternatives: list[dict[int, tm._SourceTetRecord]] = []
    seen: set[tuple[int, ...]] = set()
    for record in choices[representative]:
        support = search.carriers[representative, record[0]]
        if support in seen:
            continue
        seen.add(support)
        selected = {representative: record}
        for vertex in members:
            if vertex == representative:
                continue
            shifts = (
                search.state.periodic_vertex_windings[vertex]
                - search.state.periodic_vertex_windings[representative]
            )
            mapped = _transport_periodic_vertex_ids(transport, support, shifts)
            member = (
                None if mapped is None else available[vertex].get(tuple(sorted(mapped)))
            )
            if member is None:
                break
            selected[vertex] = member
        if len(selected) == len(members):
            alternatives.append(selected)
    return tuple(alternatives)


def _source_choices(
    search: _Search,
    details: tuple[_VertexDetail, ...],
    choices: tuple[tuple[tm._SourceTetRecord, ...], ...],
) -> Iterator[tuple[tm._SourceTetRecord, ...]]:
    if not search.orbits:
        distinct: list[tuple[tm._SourceTetRecord, ...]] = []
        for detail, records in zip(details, choices, strict=True):
            by_support: dict[tuple[int, ...], tm._SourceTetRecord] = {}
            for record in records:
                support = tuple(sorted(search.carriers[detail.vertex, record[0]]))
                by_support.setdefault(support, record)
            distinct.append(tuple(by_support.values()))
        yield from product(*distinct)
        return
    vertices = tuple(detail.vertex for detail in details)
    representatives = tuple(
        dict.fromkeys(search.orbits[vertex][0] for vertex in vertices)
    )
    by_vertex = dict(zip(vertices, choices, strict=True))
    alternatives = tuple(
        _orbit_source_choices(search, representative, by_vertex)
        for representative in representatives
    )
    # Two edge endpoints define at most two independent source choices,
    # regardless of the number of lifted members in either periodic orbit.
    for combination in product(*alternatives):
        selected = {
            vertex: record for group in combination for vertex, record in group.items()
        }
        yield tuple(selected[vertex] for vertex in vertices)


def _coupled_relocation_phase(state: tm._NativeState, counters: dict[str, int], /) -> int:
    """Search selected incident source cells around violating edges once per state."""
    search = _prepare_search(state)
    if not search.before[0] and not search.before[1]:
        return 0
    policy = ConvexSolvePolicy()
    pending: list[
        tuple[
            int,
            tuple[_VertexDetail, ...],
            tuple[tm._SourceTetRecord, ...],
            _CoupledBinding,
            tuple[_Neighbor, ...],
        ]
    ] = []
    for index, (edge, length) in enumerate(
        zip(search.edges, search.prepared.lengths, strict=True)
    ):
        if search.prepared.protected[edge] or tm._LOWER <= length <= tm._UPPER:
            continue
        vertices = tuple(
            sorted(
                set().union(*(search.orbits.get(v, (v,)) for v in edge)),
                key=state.vertex_target_ids.__getitem__,
            )
        )
        details = _vertex_details(search, vertices)
        if details is None:
            continue
        incident = [
            i
            for i, other in enumerate(search.edges)
            if search.measured[i] and any(v in other for v in vertices)
        ]
        choices = tuple(
            _ordered_source_cells(search, detail, edge, index) for detail in details
        )
        patch_saved = False
        for selected in _source_choices(search, details, choices):
            binding = _CoupledBinding(
                state.source_identity,
                int(state.snapshot.generation),
                edge,
                vertices,
                tuple(record[0] for record in selected),
            )
            attempt = _search_model(
                search, details, selected, incident, binding, policy, counters
            )
            if attempt.applied:
                return attempt.applied
            if not patch_saved and attempt.neighbors:
                pending.append((index, details, selected, binding, attempt.neighbors))
                patch_saved = True
    # Preserve the complete pair-first phase. Retain at most one bounded
    # patch per violating edge, never a Cartesian set of enriched carriers.
    for index, details, selected, binding, neighbors in pending:
        count = _search_neighbor_patch(
            search, details, selected, binding, policy, counters, neighbors, index
        )
        if count:
            return count
    return 0
