from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from math import floor, pi, sqrt
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx
from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.discretization import CellBlock, CellMesh
from phydrax.discretization._exact_algebraic_geometry import ExactScalar
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization._periodic_exact import exact_periodic_element
from phydrax.discretization._periodic_topology import (
    _entity_vertex_rows,
    PeriodicMeshTopology,
)
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    SizeCompliancePolicy,
    SurfaceMeshingSpec,
)
from phydrax.meshing._controls import (
    FeatureKind,
    PeriodicConstraint,
    ProtectedFeature,
    RegionControl,
)
from phydrax.meshing._lineage import EntityLineageKind
from phydrax.meshing._organization import RegionRole
from phydrax.meshing._periodic import (
    _cell_edge_rows,
    _entity_layout,
    _periodic_cell_layout,
    periodic_delaunay_mesh,
    periodic_entity_orbit_map,
    periodic_orbit_closure,
    PeriodicPointOrbits,
    PeriodicRefinementEdgeLock,
    prepare_periodic_refinement_edge_lock,
)
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._sizing import UniformSizeControl
from phydrax.meshing.providers import NativeMeshingOptions, NativeMeshingProvider
from phydrax.meshing.providers._native_periodic import NativePeriodicSource

from .._runtime import accepted_result_record, require_meshcore, runtime_identity


_MIGRATION = Path(__file__).resolve().parents[1] / "native_case_migration.json"
_PROFILE_IDS = (
    "native-periodic-lifecycle:skew-lattice-exact-algebraic",
    "native-periodic-lifecycle:skew-lattice-exact-algebraic:larger",
    "native-periodic-lifecycle:skew-material-feature-exact-algebraic",
    "native-periodic-lifecycle:skew-material-feature-exact-algebraic:larger",
)
_SOURCE_IDS = {
    "skew-lattice-exact-algebraic": "qualification-skew-torus-exact-algebraic",
    "skew-material-feature-exact-algebraic": "qualification-skew-material-torus-exact-algebraic",
}
_VECTORS = ((Fraction(1), Fraction(0)), (Fraction(1, 4), Fraction(1)))
_VECTOR_ARRAY = np.asarray([[1.0, 0.0], [0.25, 1.0]], dtype=np.float64)
_SOURCE_COORDINATE_TOLERANCE = 1.0e-14
_MEASURE_TOLERANCE = 1.0e-10
_TRANSFER_TOLERANCE = 1.0e-10


@dataclass(frozen=True, slots=True)
class _Profile:
    profile_id: str
    case: str
    count: int
    material_feature: bool


_PROFILES = {
    profile_id: _Profile(
        profile_id,
        case,
        count,
        material_feature,
    )
    for profile_id, case, count, material_feature in (
        (_PROFILE_IDS[0], "skew-lattice-exact-algebraic", 4, False),
        (_PROFILE_IDS[1], "skew-lattice-exact-algebraic", 8, False),
        (_PROFILE_IDS[2], "skew-material-feature-exact-algebraic", 4, True),
        (_PROFILE_IDS[3], "skew-material-feature-exact-algebraic", 8, True),
    )
}


def _payload(profile: _Profile, /) -> dict[str, Any]:
    count = profile.count
    numerators = np.stack(
        np.meshgrid(
            np.arange(count, dtype=np.int64),
            np.arange(count, dtype=np.int64),
            indexing="ij",
        ),
        axis=-1,
    ).reshape((-1, 2))
    payload: dict[str, Any] = {
        "case": "skew-material-feature" if profile.material_feature else "skew-lattice",
        "vectors": [[1.0, 0.0], [0.25, 1.0]],
        "fractional_seed_numerators": numerators.tolist(),
        "denominator": count,
    }
    if profile.material_feature:
        corners: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
        regions: list[int] = []
        for i in range(count):
            for j in range(count):
                corners.extend(
                    (
                        ((i, j), (i + 1, j), (i + 1, j + 1)),
                        ((i, j), (i + 1, j + 1), (i, j + 1)),
                    )
                )
                regions.extend((0 if i < count // 2 else 1,) * 2)
        integer_corners = np.asarray(corners, dtype=np.int64)
        payload.update(
            simplices=(
                (integer_corners[..., 0] % count) * count
                + integer_corners[..., 1] % count
            ).tolist(),
            shifts=(integer_corners // count).tolist(),
            cell_regions=regions,
        )
    return payload


def _point_ids(numerators: np.ndarray, count: int, /) -> np.ndarray:
    """Stable authored IDs keyed by the same rational site at either resolution."""
    scale = 8 // count
    scaled = np.asarray(numerators, dtype=np.int64) * scale
    return 100 + 16 * scaled[:, 0] + scaled[:, 1]


def _source_revision(payload: dict[str, Any], /) -> str:
    return canonical_fingerprint(
        {"authored_source": payload, "carrier_representation": "exact-algebraic"}
    )


def _fractional_rows(coordinates: np.ndarray, /) -> tuple[tuple[Fraction, Fraction], ...]:
    (a, b), (c, d) = _VECTORS
    determinant = a * d - b * c
    result = []
    for raw_x, raw_y in np.asarray(coordinates, dtype=np.float64):
        x, y = Fraction.from_float(float(raw_x)), Fraction.from_float(float(raw_y))
        result.append(((d * x - c * y) / determinant, (a * y - b * x) / determinant))
    return tuple(result)


def _fractional_coordinates(
    mesh: CellMesh, /
) -> tuple[tuple[ExactScalar, ExactScalar], ...]:
    topology = mesh.periodic_topology
    witness = None if topology is None else topology.exact_coordinate_source
    if witness is None:
        return _fractional_rows(np.asarray(mesh.coordinates, dtype=np.float64))
    (a, b), (c, d) = _VECTORS
    determinant = a * d - b * c
    return tuple(
        ((d * x - c * y) / determinant, (a * y - b * x) / determinant)
        for x, y in witness.source_coordinates()
    )


def _exact_physical_rows(mesh: CellMesh, /) -> tuple[tuple[ExactScalar, ...], ...]:
    """Exact lifted vertex coordinates: the witness bank, else exact binary64."""
    topology = mesh.periodic_topology
    witness = None if topology is None else topology.exact_coordinate_source
    if witness is None:
        return tuple(
            tuple(Fraction.from_float(float(value)) for value in row)
            for row in np.asarray(mesh.coordinates, dtype=np.float64)
        )
    return witness.source_coordinates()


def _maximum_matching(adjacency: list[list[int]], match: list[int], /) -> list[int]:
    """Augment ``match`` to a maximum-cardinality matching (Edmonds blossoms).

    Matched vertices stay matched; neighbors are visited in their given order, so
    the result is deterministic.
    """
    count = len(adjacency)

    def augmenting_path(root: int) -> tuple[int, list[int]]:
        used = [False] * count
        parent = [-1] * count
        base = list(range(count))
        used[root] = True
        queue = [root]

        def common_base(first: int, second: int) -> int:
            seen = [False] * count
            while True:
                first = base[first]
                seen[first] = True
                if match[first] == -1:
                    break
                first = parent[match[first]]
            while True:
                second = base[second]
                if seen[second]:
                    return second
                second = parent[match[second]]

        def mark(vertex: int, ancestor: int, child: int, blossom: list[bool]) -> None:
            while base[vertex] != ancestor:
                blossom[base[vertex]] = blossom[base[match[vertex]]] = True
                parent[vertex] = child
                child = match[vertex]
                vertex = parent[match[vertex]]

        head = 0
        while head < len(queue):
            vertex = queue[head]
            head += 1
            for neighbor in adjacency[vertex]:
                if base[vertex] == base[neighbor] or match[vertex] == neighbor:
                    continue
                if neighbor == root or (
                    match[neighbor] != -1 and parent[match[neighbor]] != -1
                ):
                    ancestor = common_base(vertex, neighbor)
                    blossom = [False] * count
                    mark(vertex, ancestor, neighbor, blossom)
                    mark(neighbor, ancestor, vertex, blossom)
                    for row in range(count):
                        if blossom[base[row]]:
                            base[row] = ancestor
                            if not used[row]:
                                used[row] = True
                                queue.append(row)
                elif parent[neighbor] == -1:
                    parent[neighbor] = vertex
                    if match[neighbor] == -1:
                        return neighbor, parent
                    used[match[neighbor]] = True
                    queue.append(match[neighbor])
        return -1, parent

    for root in range(count):
        if match[root] != -1:
            continue
        end, parent = augmenting_path(root)
        while end != -1:
            previous = parent[end]
            following = match[previous]
            match[end], match[previous] = previous, end
            end = following
    return match


def _matched_bisection_lock(
    mesh: CellMesh, /
) -> tuple[PeriodicRefinementEdgeLock, dict[str, int]]:
    """Mark every cell orbit with a Maubach-compatible exact longest-edge plan.

    Locked periodic bisection requires matched refinement edges: both quotient
    cells beside a selected quotient edge select it. The plan is a perfect
    matching of the quotient dual graph, seeded greedily by exact squared length
    in the coefficient field (edges longest in both cells first, ties by quotient
    edge ID) and completed by Edmonds augmentation, which keeps seeded cells
    matched. Every lifted image of a cell selects the image of its matched edge.
    """
    dimension = mesh.topological_dimension
    topology = mesh.periodic_topology
    cell_layout = _periodic_cell_layout(mesh)
    if dimension != 2 or topology is None or cell_layout is None:
        raise RuntimeError(
            "Matched periodic bisection requires a periodic triangle mesh."
        )
    cell_ids = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    edge_orbits = np.asarray(_entity_layout(mesh, 1).orbit_rows, dtype=np.int64)
    cell_orbits = np.asarray(cell_layout.orbit_rows, dtype=np.int64)
    cell_edges = _cell_edge_rows(mesh)
    exact = _exact_physical_rows(mesh)
    squared: dict[int, ExactScalar] = {}
    for row, (first, second) in enumerate(_entity_vertex_rows(mesh)[1]):
        squared[int(edge_orbits[row])] = sum(
            (
                (exact[int(second)][axis] - exact[int(first)][axis]) ** 2
                for axis in range(mesh.ambient_dimension)
            ),
            Fraction(0),
        )
    quotient_edge_ids = np.asarray(
        topology.quotient.entity_sets[1].entity_ids, dtype=np.int64
    )
    cell_count = int(np.max(cell_orbits)) + 1
    edges_of_cell: list[set[int]] = [set() for _ in range(cell_count)]
    cells_of_edge: dict[int, set[int]] = {}
    for row, edges in enumerate(cell_edges):
        for edge in edges:
            edges_of_cell[int(cell_orbits[row])].add(int(edge_orbits[edge]))
            cells_of_edge.setdefault(int(edge_orbits[edge]), set()).add(
                int(cell_orbits[row])
            )
    longest = [
        max(squared[edge] for edge in edges_of_cell[orbit]) for orbit in range(cell_count)
    ]
    pairs: dict[tuple[int, int], tuple[int, ExactScalar, int, int]] = {}
    for edge, cells in cells_of_edge.items():
        if len(cells) != 2:
            continue
        first, second = sorted(cells)
        tier = int(squared[edge] == longest[first]) + int(
            squared[edge] == longest[second]
        )
        key = (-tier, -squared[edge], int(quotient_edge_ids[edge]), edge)
        if (first, second) not in pairs or key < pairs[(first, second)]:
            pairs[(first, second)] = key
    match = [-1] * cell_count
    for (first, second), key in sorted(pairs.items(), key=lambda item: item[1]):
        if key[0] == -2 and match[first] == -1 and match[second] == -1:
            match[first], match[second] = second, first
    adjacency: list[list[int]] = [[] for _ in range(cell_count)]
    for (first, second), _ in sorted(pairs.items(), key=lambda item: item[1]):
        adjacency[first].append(second)
        adjacency[second].append(first)
    match = _maximum_matching(adjacency, match)
    if any(value == -1 for value in match):
        raise RuntimeError("The quotient dual graph has no perfect bisection matching.")
    selected_orbit = [
        pairs[(min(orbit, partner), max(orbit, partner))][3]
        for orbit, partner in enumerate(match)
    ]
    selected_rows: list[int] = []
    for row, edges in enumerate(cell_edges):
        members = [
            int(edge)
            for edge in edges
            if edge_orbits[edge] == selected_orbit[int(cell_orbits[row])]
        ]
        if len(members) != 1:
            raise RuntimeError("A marked cell meets its matched quotient edge twice.")
        selected_rows.append(members[0])
    statistics = {
        "cell_orbits": cell_count,
        "longest_edge_cells": sum(
            squared[selected_orbit[orbit]] == longest[orbit]
            for orbit in range(cell_count)
        ),
    }
    return (
        prepare_periodic_refinement_edge_lock(
            mesh, cell_ids, edge_ids[np.asarray(selected_rows, dtype=np.int64)]
        ),
        statistics,
    )


_TARGET_L2_ERROR = 0.05


class _SkewTorusForcing(StrictModule, NonTrainableState):
    """Manufactured skew-torus Poisson forcing; lattice data stay dynamic leaves."""

    dual: Array
    eigenvalues: Array

    def __init__(self, dual: Array, /) -> None:
        self.dual = dual
        self.eigenvalues = 4.0 * jnp.pi**2 * jnp.sum(dual * dual, axis=0)

    def __call__(self, points: Array, context: object) -> Array:
        del context
        fractional = points @ self.dual
        return self.eigenvalues[0] * jnp.cos(
            2.0 * jnp.pi * fractional[..., 0]
        ) + 0.5 * self.eigenvalues[1] * jnp.sin(2.0 * jnp.pi * fractional[..., 1])


def _skew_torus_solution(points: np.ndarray, /) -> np.ndarray:
    fractional = np.asarray(points, dtype=np.float64) @ np.linalg.inv(_VECTOR_ARRAY)
    return np.cos(2.0 * pi * fractional[..., 0]) + 0.5 * np.sin(
        2.0 * pi * fractional[..., 1]
    )


def _vertex_quotient_rows(mesh: CellMesh, /) -> np.ndarray:
    """Quotient vertex row of every lifted vertex row (vertex orbits are unsigned)."""
    orbit_map = periodic_entity_orbit_map(mesh, 0)
    lifted_ids = np.asarray(orbit_map.lifted_entity_ids, dtype=np.int64)
    if not np.all(np.asarray(orbit_map.orientation_signs) == 1):
        raise RuntimeError("Periodic vertex orbits must carry positive orientation.")
    row_by_id = {int(identifier): row for row, identifier in enumerate(lifted_ids)}
    orbits = np.asarray(orbit_map.lifted_to_quotient, dtype=np.int64)
    return np.asarray(
        [orbits[row_by_id[int(identifier)]] for identifier in mesh.vertex_global_ids],
        dtype=np.int64,
    )


def _periodic_poisson(result: Any, /) -> dict[str, Any]:
    """Solve the original mean-zero skew-torus Poisson problem on the quotient.

    Lifted P1 coefficients are prolonged from quotient vertex orbits, and the
    constant nullspace is removed by a zero-mean pivot, in one exact affine DOF
    constraint. Physical L2 error and per-cell indicators use an independent
    order-8 Gauss-Legendre triangle rule on the accepted geometry.
    """
    from phydrax.linalg import ArraySpace
    from phydrax.sparse import EdgeRelation, SparseCoordinateOperator

    mesh = result.mesh
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("triangle", 1)
    )
    space = phx.discretization.FiniteElementPlan(
        mesh, field, coordinate_spec=result.geometry
    ).prepare()
    vertex_count = int(np.asarray(mesh.vertex_global_ids).size)
    dof_count = int(space.dof_maps[0].global_dof_count)
    dof_of_vertex = np.full((vertex_count,), -1, dtype=np.int64)
    for index, block in enumerate(mesh.blocks):
        routes = np.asarray(space.dof_maps[0].cell_dofs[index], dtype=np.int64)
        vertices = np.asarray(block.vertices, dtype=np.int64)
        if routes.shape != vertices.shape:
            raise RuntimeError("P1 cell DOFs do not align with triangle corners.")
        for vertex, dof in zip(vertices.ravel(), routes.ravel(), strict=True):
            if dof_of_vertex[vertex] not in (-1, dof):
                raise RuntimeError("One lifted vertex has two P1 DOFs.")
            dof_of_vertex[vertex] = dof
    if dof_count != vertex_count or np.unique(dof_of_vertex).size != vertex_count:
        raise RuntimeError("Lifted P1 DOFs are not one per lifted vertex.")
    lifted_weights = np.zeros((dof_count,), dtype=np.float64)
    for index, geometry in enumerate(
        space.evaluate_geometry("u", space.default_runtime.coordinates)
    ):
        local = np.einsum(
            "ql,cq->cl",
            np.asarray(geometry.basis_values),
            np.asarray(geometry.physical_weights),
        )
        routes = np.asarray(space.dof_maps[0].cell_dofs[index])
        np.add.at(lifted_weights, routes.reshape(-1), local.reshape(-1))
    quotient_of_vertex = _vertex_quotient_rows(mesh)
    quotient_count = int(np.max(quotient_of_vertex)) + 1
    quotient_of_dof = np.empty((dof_count,), dtype=np.int64)
    quotient_of_dof[dof_of_vertex] = quotient_of_vertex
    quotient_weights = np.zeros((quotient_count,), dtype=np.float64)
    np.add.at(quotient_weights, quotient_of_dof, lifted_weights)
    pivot = int(np.argmax(quotient_weights))
    if not quotient_weights[pivot] > 0.0:
        raise RuntimeError("The zero-mean pivot needs positive quotient mass.")
    kept = np.flatnonzero(np.arange(quotient_count) != pivot)
    reduced_column = np.full((quotient_count,), -1, dtype=np.int64)
    reduced_column[kept] = np.arange(kept.size)
    columns, rows, data = [], [], []
    for dof in range(dof_count):
        orbit = int(quotient_of_dof[dof])
        if orbit != pivot:
            columns.append(int(reduced_column[orbit]))
            rows.append(dof)
            data.append(1.0)
        else:
            columns.extend(range(kept.size))
            rows.extend([dof] * kept.size)
            data.extend((-quotient_weights[kept] / quotient_weights[pivot]).tolist())
    relation = EdgeRelation(
        np.asarray(columns, dtype=np.int32),
        np.asarray(rows, dtype=np.int64),
        source_size=kept.size,
        target_size=dof_count,
    )
    identity = canonical_fingerprint(
        {
            "kind": "skew-torus-periodic-zero-mean-p1",
            "mesh": mesh.mesh_id,
            "periodic_topology": mesh.periodic_topology.periodic_topology_id,
            "weights": array_tree_fingerprint(lifted_weights),
        }
    )
    prolongation = SparseCoordinateOperator(
        relation,
        np.asarray(data, dtype=np.float64),
        source=ArraySpace((kept.size,), dtype=jnp.float64),
        target=space.field_spaces[0].vector_space,
        operator_id=identity,
    )
    constraint = phx.discretization.affine_dof_constraint(
        space, "u", prolongation, constraint_id=identity
    )
    dual = jnp.asarray(np.linalg.inv(_VECTOR_ARRAY))
    form = phx.equations.FiniteElementForm(
        "qualification-skew-periodic-poisson",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(
                    _SkewTorusForcing(dual),
                    coefficient_id=canonical_fingerprint(
                        {
                            "kind": "skew-torus-manufactured-forcing",
                            "vectors": _VECTOR_ARRAY.tolist(),
                        }
                    ),
                ),
            ),
        ),
    )
    problem = phx.equations.compile_finite_element_problem(
        form, space, constraint=constraint
    )
    operator, rhs = problem.linear_system()
    solved = phx.linalg.solve(operator, rhs)
    if not bool(jnp.all(solved.successful)):
        raise RuntimeError("The periodic skew-torus Poisson solve did not converge.")
    values = np.asarray(problem.expand(solved.value), dtype=np.float64)
    quotient_values = np.zeros((quotient_count,), dtype=np.float64)
    quotient_values[quotient_of_dof] = values
    if not np.array_equal(values, quotient_values[quotient_of_dof]):
        raise RuntimeError("The periodic P1 solution differs between orbit images.")
    mean = float(lifted_weights @ values / np.sum(lifted_weights))
    rule = phx.integration.reference_rule_data(
        phx.integration.ReferenceTriangleRule(phx.integration.GaussLegendreRule(8))
    )
    squared_errors, measures = [], []
    for index in range(len(mesh.blocks)):
        geometry = space.evaluate_block_geometry(
            "u", index, space.default_runtime.coordinates, rule.points, rule.weights
        )
        routes = np.asarray(space.dof_maps[0].cell_dofs[index])
        approximate = np.einsum(
            "ql,cl->cq", np.asarray(geometry.basis_values), values[routes]
        )
        exact = _skew_torus_solution(np.asarray(geometry.physical_points))
        weights = np.asarray(geometry.physical_weights)
        squared_errors.append(np.sum(weights * (approximate - exact) ** 2, axis=1))
        measures.append(np.sum(weights, axis=1))
    squared = np.concatenate(squared_errors)
    error = float(np.sqrt(np.sum(squared)))
    if abs(mean) > 1.0e-10 or not np.isfinite(error):
        raise RuntimeError(
            f"Periodic Poisson state failed mean/error acceptance: mean={mean}, error={error}."
        )
    return {
        "values": values,
        "vertex_values": values[dof_of_vertex],
        "cell_ids": np.concatenate(
            tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
        ),
        "cell_squared_errors": squared,
        "cell_measures": np.concatenate(measures),
        "record": {
            "l2_error": error,
            "mean": mean,
            "quotient_dofs": quotient_count,
            "lifted_dofs": dof_count,
            "solution_fingerprint": array_tree_fingerprint(values),
            "constraint_id": identity,
        },
    }


def _material_mesh(
    profile: _Profile,
    cell: PeriodicCell,
    point_ids: np.ndarray,
    /,
) -> tuple[CellMesh, np.ndarray]:
    count = profile.count
    vertices = np.arange((count + 1) ** 2, dtype=np.int64).reshape((count + 1, count + 1))
    grid = np.stack(
        np.meshgrid(
            np.arange(count + 1, dtype=np.int64),
            np.arange(count + 1, dtype=np.int64),
            indexing="ij",
        ),
        axis=-1,
    ).reshape((-1, 2))
    coordinates = (grid.astype(np.float64) / count) @ _VECTOR_ARRAY
    representatives = vertices[grid[:, 0] % count, grid[:, 1] % count].astype(np.int64)
    shifts = (grid // count).astype(np.int64)
    lifted_ids = np.empty((grid.shape[0],), dtype=np.int64)
    for row, (i, j) in enumerate(grid.tolist()):
        if i < count and j < count:
            lifted_ids[row] = point_ids[i * count + j]
        else:
            lifted_ids[row] = 1_000_000 + row

    triangles: list[tuple[int, int, int]] = []
    cell_regions: list[int] = []
    cell_ids: list[int] = []
    for i in range(count):
        for j in range(count):
            triangles.extend(
                (
                    (
                        int(vertices[i, j]),
                        int(vertices[i + 1, j]),
                        int(vertices[i + 1, j + 1]),
                    ),
                    (
                        int(vertices[i, j]),
                        int(vertices[i + 1, j + 1]),
                        int(vertices[i, j + 1]),
                    ),
                )
            )
            region = 0 if i < count // 2 else 1
            cell_regions.extend((region, region))
            authored_cell_base = 10_000 + i * 128 + j * 2
            cell_ids.extend((authored_cell_base, authored_cell_base + 1))

    block = CellBlock(
        "authored-skew-periodic",
        "triangle",
        np.asarray(triangles, dtype=np.int32),
        global_ids=np.asarray(cell_ids, dtype=np.int64),
    )
    lifted = CellMesh(coordinates, (block,), vertex_global_ids=lifted_ids)
    periodic = PeriodicMeshTopology(lifted, cell, representatives, shifts)
    mesh = CellMesh(
        coordinates,
        (block,),
        vertex_global_ids=lifted_ids,
        periodic_topology=periodic,
    )
    return mesh, np.asarray(cell_regions, dtype=np.int32)


def _mesh_source(
    profile: _Profile,
    payload: dict[str, Any],
    cell: PeriodicCell,
    points: np.ndarray,
    point_ids: np.ndarray,
    /,
) -> tuple[NativePeriodicSource, CellMesh, np.ndarray | None]:
    source_id = _SOURCE_IDS[profile.case]
    revision = _source_revision(payload)
    if profile.material_feature:
        domain, regions = _material_mesh(profile, cell, point_ids)
        source = NativePeriodicSource(
            domain,
            source_id,
            revision,
            cell_regions=regions,
        )
        return source, domain, regions
    orbits = PeriodicPointOrbits(cell, points, point_ids)
    source = NativePeriodicSource(orbits, source_id, revision)
    return source, periodic_delaunay_mesh(orbits).mesh, None


def _periodic_constraint(
    mesh: CellMesh,
    source_id: str,
    revision: str,
    /,
) -> PeriodicConstraint:
    orbit_map = periodic_entity_orbit_map(mesh, 0)
    periodic_topology = mesh.periodic_topology
    if periodic_topology is None:
        raise RuntimeError("Authored periodic source omitted its periodic topology.")
    ids = np.asarray(orbit_map.lifted_entity_ids, dtype=np.int64)
    quotient_rows = np.asarray(orbit_map.lifted_to_quotient, dtype=np.int64)
    windings = np.asarray(orbit_map.winding_shifts, dtype=np.int64)
    for quotient_row in range(orbit_map.quotient_entity_ids.size):
        members = np.flatnonzero(quotient_rows == quotient_row)
        for source_row in members:
            for target_row in members:
                exponent = windings[target_row] - windings[source_row]
                if not np.any(exponent):
                    continue
                entity_set_id = mesh.entity_set(0).entity_set_id
                source_scope = MeshingScope(
                    source_id,
                    revision,
                    MeshingEntityKind.MESH,
                    0,
                    entity_set_id,
                    ids[[source_row]],
                )
                target_scope = MeshingScope(
                    source_id,
                    revision,
                    MeshingEntityKind.MESH,
                    0,
                    entity_set_id,
                    ids[[target_row]],
                )
                transform = np.asarray(
                    exact_periodic_element(periodic_topology.cell, exponent),
                    dtype=np.float64,
                )
                return PeriodicConstraint(source_scope, target_scope, transform)
    raise RuntimeError("Authored periodic source has no nontrivial vertex winding pair.")


def _edge_vertex_rows(mesh: CellMesh, /) -> list[list[int]]:
    """Return each edge-row's two vertex rows from the mesh's vertex/edge incidence."""
    if mesh.topological_dimension != 2:
        raise RuntimeError("Periodic material features require a two-dimensional mesh.")
    relation = mesh.topology.incidences[0].relation
    valid = np.asarray(relation.valid, dtype=np.bool_)
    vertex_rows = np.asarray(relation.source_indices, dtype=np.int64)[valid]
    edge_rows = np.asarray(relation.target_indices, dtype=np.int64)[valid]
    edge_count = int(np.asarray(mesh.entity_set(1).entity_ids).size)
    endpoints: list[list[int]] = [[] for _ in range(edge_count)]
    for vertex_row, edge_row in zip(
        vertex_rows.tolist(), edge_rows.tolist(), strict=True
    ):
        endpoints[edge_row].append(vertex_row)
    if any(len(rows) != 2 for rows in endpoints):
        raise RuntimeError(
            "Mesh edge incidence does not bind exactly two vertices per edge."
        )
    return endpoints


def _source_feature_ids(mesh: CellMesh, /) -> np.ndarray:
    edge_ids: list[int] = np.asarray(
        mesh.entity_set(1).entity_ids, dtype=np.int64
    ).tolist()
    edge_rows = _edge_vertex_rows(mesh)
    fractional = _fractional_coordinates(mesh)
    selected: list[int] = []
    for edge_id, endpoints in zip(edge_ids, edge_rows, strict=True):
        x0 = fractional[endpoints[0]][0]
        x1 = fractional[endpoints[1]][0]
        x0 -= floor(x0)
        x1 -= floor(x1)
        if x0 == x1 and x0 in (Fraction(0), Fraction(1, 2)):
            selected.append(edge_id)
    if not selected:
        raise RuntimeError(
            "Exact material source has no authored periodic interface edges."
        )
    closure = periodic_orbit_closure(mesh, 1, np.asarray(selected, dtype=np.int64))
    closed = closure.require_source(mesh, 1)
    if not np.array_equal(np.sort(closed), np.sort(np.asarray(selected, dtype=np.int64))):
        raise RuntimeError(
            "Authored material-interface edge IDs are not a complete periodic orbit set."
        )
    return closed


def _make_specification(
    source: NativePeriodicSource,
    mesh: CellMesh,
    profile: _Profile,
    cell_regions: np.ndarray | None,
    /,
) -> SurfaceMeshingSpec:
    mesh_source = isinstance(source.domain, CellMesh)
    entity_kind = MeshingEntityKind.MESH if mesh_source else MeshingEntityKind.GEOMETRY
    domain_scope = MeshingScope(
        source.source_id,
        source.source_revision,
        entity_kind,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    if mesh_source:
        periodic_topology = mesh.periodic_topology
        if periodic_topology is None:
            raise ValueError(
                "Mesh-backed periodic profile omitted its quotient topology."
            )
        quotient_vertices = periodic_topology.quotient.entity_sets[0]
        size_set_id = quotient_vertices.entity_set_id
        size_ids = np.asarray(quotient_vertices.entity_ids, dtype=np.int64)
    else:
        size_set_id = source.scope_entity_set_id(0)
        size_ids = source.scope_entity_ids(0)
    size_scope = MeshingScope(
        source.source_id,
        source.source_revision,
        entity_kind,
        0,
        size_set_id,
        size_ids,
    )
    controls: tuple[RegionControl, ...] = ()
    features: tuple[ProtectedFeature, ...] = ()
    constraints: tuple[PeriodicConstraint, ...] = ()
    if mesh_source:
        cell_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
        if cell_regions is None:
            raise RuntimeError("Material source omitted its authored cell-region table.")
        by_cell_id = {
            int(identifier): int(code)
            for identifier, code in zip(
                np.concatenate(
                    tuple(
                        np.asarray(block.global_ids, dtype=np.int64)
                        for block in mesh.blocks
                    )
                ),
                cell_regions,
                strict=True,
            )
        }
        controls = tuple(
            RegionControl(
                MeshingScope(
                    source.source_id,
                    source.source_revision,
                    MeshingEntityKind.MESH,
                    2,
                    mesh.entity_set(2).entity_set_id,
                    cell_ids[
                        np.asarray(
                            [
                                by_cell_id[int(identifier)] == code
                                for identifier in cell_ids
                            ],
                            dtype=np.bool_,
                        )
                    ],
                ),
                f"material-{code}",
                f"material-{code}",
                RegionRole.USER,
            )
            for code in (0, 1)
        )
        feature_ids = _source_feature_ids(mesh)
        features = (
            ProtectedFeature(
                MeshingScope(
                    source.source_id,
                    source.source_revision,
                    MeshingEntityKind.MESH,
                    1,
                    mesh.entity_set(1).entity_set_id,
                    feature_ids,
                ),
                FeatureKind.MATERIAL_INTERFACE,
                maximum_deviation=0.0,
                hard=True,
            ),
        )
        constraints = (
            _periodic_constraint(mesh, source.source_id, source.source_revision),
        )
    return SurfaceMeshingSpec(
        CellMeshingTarget(2, 2, CellFamilyPolicy(required=("triangle",))),
        domain_scope,
        size_controls=(
            UniformSizeControl(
                size_scope,
                1.0 / profile.count,
                maximum_size=1.5 / profile.count,
            ),
        ),
        protected_features=features,
        region_controls=controls,
        periodic_constraints=constraints,
        size_compliance=SizeCompliancePolicy(),
    )


def _cells(mesh: CellMesh, /) -> tuple[tuple[int, tuple[int, ...]], ...]:
    records: list[tuple[int, tuple[int, ...]]] = []
    for block in mesh.blocks:
        if block.cell_kind != "triangle":
            raise RuntimeError(
                f"Periodic 2D oracle requires triangle blocks, found {block.cell_kind!r}."
            )
        cell_ids: list[int] = np.asarray(block.global_ids, dtype=np.int64).tolist()
        cell_vertices: list[list[int]] = np.asarray(
            block.vertices, dtype=np.int64
        ).tolist()
        records.extend(
            (cell_id, tuple(vertices))
            for cell_id, vertices in zip(cell_ids, cell_vertices, strict=True)
        )
    return tuple(records)


def _fractional_area(
    triangle: tuple[int, ...],
    fractional: tuple[tuple[ExactScalar, ExactScalar], ...],
    /,
) -> ExactScalar:
    a, b, c = (fractional[index] for index in triangle)
    determinant = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return determinant / 2


def _measure_oracles(
    mesh: CellMesh,
    cell_regions: dict[int, int] | None = None,
    /,
) -> tuple[ExactScalar, dict[int, ExactScalar], int]:
    fractional = _fractional_coordinates(mesh)
    area: ExactScalar = Fraction(0)
    material_measures: dict[int, ExactScalar] = {}
    for cell_id, vertices in _cells(mesh):
        cell_area = _fractional_area(vertices, fractional)
        if cell_area <= 0:
            raise RuntimeError(
                f"Periodic cell {cell_id} has nonpositive exact signed area {cell_area}."
            )
        area += cell_area
        if cell_regions is not None:
            code = cell_regions[cell_id]
            material_measures[code] = material_measures.get(code, Fraction(0)) + cell_area
    return area, material_measures, len(_cells(mesh))


def _assert_quotient(mesh: CellMesh, input_point_ids: np.ndarray, /) -> dict[str, Any]:
    periodic = mesh.periodic_topology
    if periodic is None:
        raise RuntimeError("Periodic producer returned a mesh without quotient topology.")
    periodic.require_lifted(mesh.topology, mesh.lifted_topology_id, mesh.blocks)
    periodic.require_coordinates(mesh.coordinates)
    vertex_map = periodic_entity_orbit_map(mesh, 0)
    edge_map = periodic_entity_orbit_map(mesh, 1)
    cell_map = periodic_entity_orbit_map(mesh, 2)
    representatives = np.asarray(periodic.vertex_representatives, dtype=np.int64)
    quotient_vertices = np.asarray(vertex_map.quotient_entity_ids, dtype=np.int64)
    if np.unique(representatives).size != quotient_vertices.size:
        raise RuntimeError(
            "Periodic vertex representatives do not map one-to-one to quotient sites."
        )
    if representatives.size <= quotient_vertices.size:
        raise RuntimeError("Periodic mesh has no shared lifted vertex representatives.")
    if not set(map(int, input_point_ids)) <= set(map(int, quotient_vertices)):
        raise RuntimeError("Periodic output lost an authored stable source point ID.")
    quotient_entity_counts = tuple(
        int(periodic.quotient.entity_sets[degree].entity_ids.size) for degree in range(3)
    )
    quotient_euler = (
        quotient_entity_counts[0] - quotient_entity_counts[1] + quotient_entity_counts[2]
    )
    boundary_edges = int(
        np.count_nonzero(
            np.asarray(periodic.quotient.entity_sets[1].subset("boundary").mask)
        )
    )
    if quotient_euler != 0 or boundary_edges:
        raise RuntimeError(
            f"Periodic quotient is not a closed torus: euler={quotient_euler}, boundary_edges={boundary_edges}."
        )
    winding_summary: dict[str, dict[str, Any]] = {}
    for name, orbit_map in (
        ("vertices", vertex_map),
        ("edges", edge_map),
        ("cells", cell_map),
    ):
        shifts = np.asarray(orbit_map.winding_shifts, dtype=np.int64)
        winding_rows = np.flatnonzero(np.any(shifts != 0, axis=1))
        if name in ("vertices", "edges") and winding_rows.size < 2:
            raise RuntimeError(f"Periodic {name} lack multiple nonzero winding entities.")
        winding_summary[name] = {
            "lifted_entities": int(shifts.shape[0]),
            "quotient_entities": int(orbit_map.quotient_entity_ids.size),
            "nonzero_winding_entities": int(winding_rows.size),
            "winding_values": sorted(
                {tuple(int(value) for value in row) for row in shifts[winding_rows]}
            ),
        }
    periodic_cell = periodic.cell
    if not isinstance(periodic_cell, PeriodicCell):
        raise RuntimeError(
            "Periodic output does not retain the authored lattice-cell representation."
        )
    if tuple(np.asarray(periodic_cell.vectors, dtype=np.float64).shape) != (2, 2):
        raise RuntimeError(
            "Periodic output changed the full-rank two-dimensional lattice shape."
        )
    if not np.array_equal(
        np.asarray(periodic_cell.vectors, dtype=np.float64), _VECTOR_ARRAY
    ):
        raise RuntimeError("Periodic output changed the authored skew lattice vectors.")
    return {
        "periodic_topology_id": periodic.periodic_topology_id,
        "quotient_entity_counts": list(quotient_entity_counts),
        "euler_characteristic": quotient_euler,
        "quotient_boundary_edges": boundary_edges,
        "shared_representative_vertices": int(
            representatives.size - quotient_vertices.size
        ),
        "source_point_ids_retained": True,
        "winding_entities": winding_summary,
    }


def _material_values(result: Any, /) -> dict[int, int]:
    attributes = [
        attribute
        for attribute in result.attributes
        if attribute.name == "periodic-material-id"
    ]
    if len(attributes) != 1:
        raise RuntimeError(
            "Periodic material producer did not publish exactly one material-ID field."
        )
    attribute = attributes[0]
    identifiers = np.asarray(attribute.scope.entity_ids, dtype=np.int64)
    values = np.asarray(attribute.values, dtype=np.int64)
    if identifiers.shape != values.shape:
        raise RuntimeError("Periodic material attribute IDs and values are not aligned.")
    return {
        int(identifier): int(value)
        for identifier, value in zip(identifiers, values, strict=True)
    }


def _check_material_mapping(
    result: Any,
    source_mesh: CellMesh,
    source_regions: np.ndarray,
    source_cell_ids: np.ndarray,
    /,
) -> dict[str, Any]:
    mesh = result.mesh
    cells = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    values_by_id = _material_values(result)
    if set(values_by_id) != set(map(int, cells)):
        raise RuntimeError(
            "Periodic material-ID output does not cover the exact target cell entity set."
        )
    values = np.asarray(
        [values_by_id[int(identifier)] for identifier in cells], dtype=np.int64
    )
    if set(map(int, np.unique(values))) != {0, 1}:
        raise RuntimeError(
            "Periodic adaptation lost one of the two authored material identities."
        )
    zones = {
        zone.material_id: zone for zone in result.zones if zone.material_id is not None
    }
    if set(zones) != {"material-0", "material-1"}:
        raise RuntimeError("Periodic adaptation lost one or more named material zones.")
    for code in (0, 1):
        zone_ids = np.asarray(zones[f"material-{code}"].scope.entity_ids, dtype=np.int64)
        expected_ids = cells[values == code]
        if not np.array_equal(np.sort(zone_ids), np.sort(expected_ids)):
            raise RuntimeError(
                f"Periodic material-{code} zone does not match its material-ID field."
            )

    cell_map = periodic_entity_orbit_map(mesh, 2)
    lifted = np.asarray(cell_map.lifted_to_quotient, dtype=np.int64)
    for quotient_row in range(cell_map.quotient_entity_ids.size):
        members = np.flatnonzero(lifted == quotient_row)
        if np.unique(values[members]).size != 1:
            raise RuntimeError(
                "Periodic material IDs are not synchronized over every cell orbit."
            )

    exact_regions = {
        int(identifier): int(code)
        for identifier, code in zip(source_cell_ids, source_regions, strict=True)
    }
    source_area, source_material, _ = _measure_oracles(source_mesh, exact_regions)
    target_area, target_material, _ = _measure_oracles(mesh, values_by_id)
    expected_region_area = {0: Fraction(1, 2), 1: Fraction(1, 2)}
    if source_area != Fraction(1) or source_material != expected_region_area:
        raise RuntimeError(
            f"Authored exact material source differs from its independent half-domain oracle: area={source_area}, materials={source_material}."
        )
    material_measures = {
        str(code): float(measure) for code, measure in target_material.items()
    }
    if abs(float(target_area) - 1.0) > _MEASURE_TOLERANCE or any(
        abs(material_measures.get(str(code), float("inf")) - 0.5) > _MEASURE_TOLERANCE
        for code in (0, 1)
    ):
        raise RuntimeError(
            f"Adapted exact quotient/material measures violate their algebraic oracle: total={float(target_area)}, materials={material_measures}."
        )

    interface = _material_interface_oracle(mesh, values_by_id)
    expected_interface = sqrt(17.0) / 2.0
    if abs(interface["length"] - expected_interface) > _MEASURE_TOLERANCE:
        raise RuntimeError(
            f"Adapted material interface length {interface['length']} differs from exact sqrt(17)/2 oracle {expected_interface}."
        )
    return {
        "source_region_measure_exact": {
            str(code): str(value) for code, value in source_material.items()
        },
        "target_region_measure": material_measures,
        "target_total_measure": float(target_area),
        "interface_orbit_count": interface["interface_orbit_count"],
        "interface_lifted_edge_count": interface["interface_lifted_edge_count"],
        "interface_length": interface["length"],
        "interface_length_exact_oracle": "sqrt(17)/2",
        "material_orbit_mapping": "consistent",
    }


def _material_interface_oracle(
    mesh: CellMesh, values_by_id: dict[int, int], /
) -> dict[str, Any]:
    edge_rows = _edge_vertex_rows(mesh)
    edge_map = periodic_entity_orbit_map(mesh, 1)
    cell_ids = np.asarray(mesh.entity_set(2).entity_ids, dtype=np.int64)
    cell_code = np.asarray(
        [values_by_id[int(identifier)] for identifier in cell_ids], dtype=np.int64
    )
    relation = mesh.topology.incidences[1].relation
    valid = np.asarray(relation.valid, dtype=np.bool_)
    edge_rows_by_incidence = np.asarray(relation.source_indices, dtype=np.int64)[valid]
    cell_rows = np.asarray(relation.target_indices, dtype=np.int64)[valid]
    fractional = _fractional_coordinates(mesh)
    orbit_rows = np.asarray(edge_map.lifted_to_quotient, dtype=np.int64)
    representative_rows = np.asarray(edge_map.representative_lifted_rows, dtype=np.int64)
    by_orbit: dict[int, set[int]] = {}
    for edge_row, cell_row in zip(edge_rows_by_incidence, cell_rows, strict=True):
        by_orbit.setdefault(int(orbit_rows[edge_row]), set()).add(
            int(cell_code[cell_row])
        )
    interface_orbits = [
        orbit for orbit, adjacent_codes in by_orbit.items() if adjacent_codes == {0, 1}
    ]
    length = 0.0
    for orbit in interface_orbits:
        edge_row = int(representative_rows[orbit])
        endpoints = edge_rows[edge_row]
        x0, y0 = fractional[endpoints[0]]
        x1, y1 = fractional[endpoints[1]]
        if x0 != x1:
            raise RuntimeError(
                "Material interface includes an edge not parallel to the exact skew period vector."
            )
        x = x0 - floor(x0)
        if x not in (Fraction(0), Fraction(1, 2)):
            raise RuntimeError(
                "Material interface is not on an authored exact periodic feature orbit."
            )
        dx = float((x1 - x0) + Fraction(1, 4) * (y1 - y0))
        dy = float(y1 - y0)
        length += float(np.hypot(dx, dy))
    if not interface_orbits:
        raise RuntimeError(
            "Adapted material field has no periodic region-interface edge orbits."
        )
    return {
        "interface_orbit_count": len(interface_orbits),
        "interface_lifted_edge_count": int(
            np.count_nonzero(np.isin(orbit_rows, interface_orbits))
        ),
        "interface_quotient_ids": sorted(
            int(edge_map.quotient_entity_ids[orbit]) for orbit in interface_orbits
        ),
        "length": length,
    }


def _periodic_scalar(point: tuple[ExactScalar, ExactScalar], /) -> ExactScalar:
    x, y = point
    x -= floor(x)
    y -= floor(y)
    x_term, y_term = x * (1 - x), y * (1 - y)
    return 1 + x_term + 2 * y_term + x_term * y_term


def _fractional_point(
    point: tuple[ExactScalar, ...], /
) -> tuple[ExactScalar, ExactScalar]:
    (a, b), (c, d) = _VECTORS
    determinant = a * d - b * c
    x, y = point
    return ((d * x - c * y) / determinant, (a * y - b * x) / determinant)


def _exact_generation_oracle(
    result: Any, specification: SurfaceMeshingSpec, /
) -> dict[str, Any]:
    """Independently integrate the published exact rows against the hard controls.

    Squared quotient edge lengths, positive orientation and the fundamental
    measure are evaluated in the coefficient field; p50/p95 must equal the
    authored target exactly and the maximum may not exceed the authored bound.
    """
    mesh = result.mesh
    witness = mesh.periodic_topology.exact_coordinate_source
    if witness is None:
        raise RuntimeError("The algebraic positive requires an exact coordinate witness.")
    exact = _exact_physical_rows(mesh)
    endpoints = _entity_vertex_rows(mesh)[1]
    edge_map = periodic_entity_orbit_map(mesh, 1)
    squares = sorted(
        sum(
            (
                (
                    exact[int(endpoints[row][1])][axis]
                    - exact[int(endpoints[row][0])][axis]
                )
                ** 2
                for axis in range(2)
            ),
            Fraction(0),
        )
        for row in np.asarray(edge_map.representative_lifted_rows, dtype=np.int64)
    )
    control = specification.size_controls[0]
    if not isinstance(control, UniformSizeControl) or control.maximum_size is None:
        raise RuntimeError("The original periodic request is one bounded uniform size.")
    target = Fraction.from_float(float(control.target_size)) ** 2
    maximum = Fraction.from_float(float(control.maximum_size)) ** 2
    quantiles = {}
    for quantile in (0.5, 0.95):
        rank = float(np.float64(quantile) * np.float64(len(squares) - 1))
        lower = int(rank)
        upper = min(lower + int(rank != lower), len(squares) - 1)
        quantiles[str(quantile)] = squares[lower] == target and squares[upper] == target
    area, _, cells = _measure_oracles(mesh)
    checks = {
        "exact_fundamental_measure": area == 1,
        "exact_p50_and_p95": all(quantiles.values()),
        "exact_maximum_size": squares[-1] <= maximum,
        "positive_exact_cell_orientation": True,
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"Exact periodic generation violates its original hard controls: {checks}."
        )
    return {
        **checks,
        "exact_witness": type(witness).__name__,
        "quotient_edges": len(squares),
        "cells": cells,
        "target_squared": str(target),
        "maximum_squared": str(maximum),
        "witness_source_map_id": witness.source_map_id,
    }


def _epoch_acceptance(
    result: Any, point_ids: np.ndarray, material: bool, /
) -> dict[str, Any]:
    """Exact quotient, orbit, measure and material acceptance of one epoch."""
    if not result.audit.passed or not result.compliance.passed:
        raise RuntimeError("A periodic epoch failed its canonical audit or compliance.")
    quotient = _assert_quotient(result.mesh, point_ids)
    values_by_id = _material_values(result) if material else None
    area, materials, cells = _measure_oracles(result.mesh, values_by_id)
    if area != 1:
        raise RuntimeError(f"Periodic epoch measure {area} differs from exact 1.")
    witness = result.mesh.periodic_topology.exact_coordinate_source
    record: dict[str, Any] = {
        "result_id": result.result_id,
        "mesh_id": result.mesh.mesh_id,
        "periodic_topology_id": quotient["periodic_topology_id"],
        "quotient": quotient,
        "cells": cells,
        "exact_measure": "1",
        "witness": None if witness is None else type(witness).__name__,
        "certified": result.certification is not None and result.certification.passed,
    }
    if values_by_id is not None:
        if materials != {0: Fraction(1, 2), 1: Fraction(1, 2)}:
            raise RuntimeError(
                f"Periodic material measures changed exactly: {materials}."
            )
        interface = _material_interface_oracle(result.mesh, values_by_id)
        if abs(interface["length"] - sqrt(17.0) / 2.0) > _MEASURE_TOLERANCE:
            raise RuntimeError(
                f"Periodic material interface length {interface['length']} differs from sqrt(17)/2."
            )
        record["material"] = {
            "exact_measures": {"0": "1/2", "1": "1/2"},
            "interface_orbits": interface["interface_orbit_count"],
            "interface_length": interface["length"],
        }
    return record


def _cell_areas(mesh: CellMesh, /) -> dict[int, ExactScalar]:
    fractional = _fractional_coordinates(mesh)
    return {
        cell_id: _fractional_area(vertices, fractional)
        for cell_id, vertices in _cells(mesh)
    }


def _exact_p1_integral(mesh: CellMesh, values: dict[int, ExactScalar], /) -> ExactScalar:
    fractional = _fractional_coordinates(mesh)
    ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    total: ExactScalar = Fraction(0)
    for _, vertices in _cells(mesh):
        total += (
            _fractional_area(vertices, fractional)
            * sum((values[int(ids[vertex])] for vertex in vertices), Fraction(0))
            / 3
        )
    return total


def _nested_transition_oracle(adaptation: Any, /, *, refine: bool) -> dict[str, Any]:
    """Prove one orbit transition's vertex stencil, P1 transfer and exact measures.

    Every canonical stencil row must reproduce its exact target vertex modulo an
    integer lattice translation. The exact skew-periodic field is interpolated
    in the coefficient field: target values must agree on every vertex orbit,
    a refinement must conserve the exact P1 integral, and parent/child cell
    measures must partition exactly. Coarsening injects retained vertex values;
    its exact P1 integral change is reported, not claimed conservative.
    """
    transition = adaptation.transition
    if (
        transition is None
        or transition.vertex_stencil is None
        or transition.lineage is None
    ):
        raise RuntimeError("A periodic transition lost its lineage or vertex stencil.")
    stencil = transition.vertex_stencil
    source, target = adaptation.source.mesh, adaptation.target.mesh
    source_exact = _exact_physical_rows(source)
    target_exact = _exact_physical_rows(target)
    source_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    target_ids = np.asarray(target.vertex_global_ids, dtype=np.int64)
    source_row = {int(identifier): row for row, identifier in enumerate(source_ids)}
    target_row = {int(identifier): row for row, identifier in enumerate(target_ids)}
    source_values = [_periodic_scalar(_fractional_point(point)) for point in source_exact]
    target_values: dict[int, ExactScalar] = {}
    stencil_sources = np.asarray(stencil.source_global_ids, dtype=np.int64)
    stencil_valid = np.asarray(stencil.valid, dtype=np.bool_)
    stencil_targets = np.asarray(stencil.target_global_ids, dtype=np.int64)
    for row, identifier in enumerate(stencil_targets.tolist()):
        point: list[ExactScalar] = [Fraction(0), Fraction(0)]
        value: ExactScalar = Fraction(0)
        for source_id, weight, active in zip(
            stencil_sources[row],
            stencil.exact_weights[row],
            stencil_valid[row],
            strict=True,
        ):
            if not active:
                continue
            source_point = source_exact[source_row[int(source_id)]]
            point = [point[axis] + weight * source_point[axis] for axis in range(2)]
            value += weight * source_values[source_row[int(source_id)]]
        offset = _fractional_point(
            tuple(
                target_exact[target_row[int(identifier)]][axis] - point[axis]
                for axis in range(2)
            )
        )
        if any(component != floor(component) for component in offset):
            raise RuntimeError(
                "A canonical periodic stencil row does not reproduce its exact target vertex."
            )
        target_values[int(identifier)] = value
    if set(target_values) != set(target_row):
        raise RuntimeError(
            "The periodic transition stencil does not cover every target vertex."
        )
    quotient_rows = _vertex_quotient_rows(target)
    for quotient_row in np.unique(quotient_rows).tolist():
        members = np.flatnonzero(quotient_rows == quotient_row)
        if len({target_values[int(target_ids[row])] for row in members}) != 1:
            raise RuntimeError(
                "Exact transferred values do not commute with periodic vertex identification."
            )
    source_integral = _exact_p1_integral(
        source,
        {
            int(identifier): source_values[row]
            for row, identifier in enumerate(source_ids)
        },
    )
    target_integral = _exact_p1_integral(target, target_values)
    defect = target_integral - source_integral
    if refine and defect != 0:
        raise RuntimeError(
            f"Nested periodic refinement changed the exact P1 integral by {defect}."
        )
    actual = np.asarray(
        stencil.apply(source_ids, np.asarray([float(value) for value in source_values])),
        dtype=np.float64,
    )
    expected = np.asarray(
        [float(target_values[int(identifier)]) for identifier in stencil_targets],
        dtype=np.float64,
    )
    residual = float(np.max(np.abs(actual - expected), initial=0.0))
    if residual > _TRANSFER_TOLERANCE:
        raise RuntimeError(
            f"Binary64 periodic P1 transfer residual {residual} is too large."
        )
    cells = transition.lineage.entity_lineage(2)
    source_areas, target_areas = _cell_areas(source), _cell_areas(target)
    groups: dict[tuple[str, int], tuple[set[int], set[int]]] = {}
    for source_id, target_id, kind in zip(
        np.asarray(cells.source_global_ids).tolist(),
        np.asarray(cells.target_global_ids).tolist(),
        np.asarray(cells.relation_kinds).tolist(),
        strict=True,
    ):
        key = (
            ("source", source_id)
            if kind == int(EntityLineageKind.REFINED_FROM)
            else ("target", target_id)
        )
        members = groups.setdefault(key, (set(), set()))
        members[0].add(source_id)
        members[1].add(target_id)
    for source_cells, target_cells in groups.values():
        if sum((source_areas[cell] for cell in source_cells), Fraction(0)) != sum(
            (target_areas[cell] for cell in target_cells), Fraction(0)
        ):
            raise RuntimeError(
                "Periodic parent and child cells do not partition exact measure."
            )
    covered_source = set().union(*(value[0] for value in groups.values()))
    covered_target = set().union(*(value[1] for value in groups.values()))
    if covered_source != set(source_areas) or covered_target != set(target_areas):
        raise RuntimeError("Periodic cell lineage does not cover both epochs.")
    return {
        "stencil_id": stencil.stencil_id,
        "stencil_weight_types": sorted(
            {type(value).__name__ for row in stencil.exact_weights for value in row}
        ),
        "exact_vertex_reproduction": True,
        "exact_orbit_commuting": True,
        "exact_integral_source": str(source_integral),
        "exact_integral_target": str(target_integral),
        "exact_integral_defect": str(defect),
        "conservative": defect == 0,
        "binary64_action_residual": residual,
        "exact_measure_partitions": len(groups),
        "tolerance": _TRANSFER_TOLERANCE,
    }


def _compatible_transition(adaptation: Any, /) -> dict[str, Any]:
    """Certify the full lowest-order de Rham transfer on the periodic quotients.

    The product admits the quotient transfer only when every lifted form degree
    commutes with the exterior derivative within its certified moment intervals
    and descends to the source and target quotient complexes.
    """
    complexes = tuple(
        phx.discretization.FiniteElementDeRhamComplex(
            endpoint.mesh,
            family="trimmed",
            order=1,
            coordinate_spec=endpoint.geometry,
        )
        for endpoint in (adaptation.source, adaptation.target)
    )
    transfer = complexes[0].transfer_periodic_quotient(
        complexes[1], geometry_transition=adaptation.transition.geometry_transition
    )
    defects = np.asarray(transfer.commuting_defects, dtype=np.float64)
    bounds = np.asarray(transfer.commuting_error_bounds, dtype=np.float64)
    if defects.shape != bounds.shape or np.any(defects > bounds):
        raise RuntimeError(
            "Periodic quotient de Rham transfer exceeds its commuting enclosure."
        )
    return {
        "transfer_id": transfer.transfer_id,
        "family": "trimmed",
        "order": 1,
        "commuting_defects": defects.tolist(),
        "commuting_error_bounds": bounds.tolist(),
    }


def _refinement_parents(adaptation: Any, /) -> dict[int, int]:
    cells = adaptation.transition.lineage.entity_lineage(2)
    return {
        int(target_id): int(source_id)
        for source_id, target_id, kind in zip(
            np.asarray(cells.source_global_ids).tolist(),
            np.asarray(cells.target_global_ids).tolist(),
            np.asarray(cells.relation_kinds).tolist(),
            strict=True,
        )
        if kind == int(EntityLineageKind.REFINED_FROM)
    }


def _coarsening_marks(
    coarse: dict[str, Any], refinement: Any, target_error: float, /
) -> np.ndarray:
    """Mark children whose retained coarse error permits the target density.

    The retained coarse Galerkin error is an a-posteriori marking indicator,
    not an acceptance certificate; the final solve enforces the unchanged
    physical L2 target independently. A created bisection vertex is removable
    only with its whole periodic star, so marks are closed by dropping every
    vertex orbit whose star contains a child of an inadmissible parent; marks
    are never added beyond the accuracy indicator.
    """
    squared = np.asarray(coarse["cell_squared_errors"], dtype=np.float64)
    measures = np.asarray(coarse["cell_measures"], dtype=np.float64)
    eligible = (
        np.ones(squared.size, dtype=np.bool_)
        if np.sum(squared) <= target_error**2
        else squared <= target_error**2 * measures / np.sum(measures)
    )
    admitted = {
        int(identifier)
        for identifier, allowed in zip(coarse["cell_ids"], eligible, strict=True)
        if allowed
    }
    parents = _refinement_parents(refinement)
    fine = refinement.target.mesh
    created = set(map(int, np.asarray(fine.vertex_global_ids))) - set(
        map(int, np.asarray(refinement.source.mesh.vertex_global_ids))
    )
    quotient_rows = _vertex_quotient_rows(fine)
    fine_ids = np.asarray(fine.vertex_global_ids, dtype=np.int64)
    stars: dict[int, set[int]] = {}
    for cell_id, vertices in _cells(fine):
        for vertex in vertices:
            if int(fine_ids[vertex]) in created:
                stars.setdefault(int(quotient_rows[vertex]), set()).add(cell_id)
    marks: set[int] = set()
    for star in stars.values():
        if all(cell in parents and parents[cell] in admitted for cell in star):
            marks |= star
    return np.asarray(sorted(marks), dtype=np.int64)


def _accepted_bisection(
    adaptation: Any, operation: str, /
) -> tuple[phx.meshing.BisectionHierarchy, str]:
    """Require a complete bisection transaction with lineage and orbit history."""
    hierarchy = adaptation.hierarchy
    transition = adaptation.transition
    if (
        adaptation.status is not phx.meshing.MeshAdaptationStatus.COMPLETE
        or not isinstance(hierarchy, phx.meshing.BisectionHierarchy)
        or transition is None
        or transition.lineage is None
    ):
        raise RuntimeError(
            f"A periodic orbit {operation} was not completely accepted with its history."
        )
    return hierarchy, transition.lineage.lineage_id


def _marked_bisection_lifecycle(
    result: Any, point_ids: np.ndarray, material: bool, /
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Run the original accuracy-controlled marked-bisection orbit lifecycle.

    Refine every cell orbit with a matched exact longest-edge plan until the
    unchanged physical L2 target is met, then coarsen the sibling families whose
    retained coarse indicator admits the target error density. Every accepted
    epoch is solved, exactly accepted and connected by a proved transition.
    """
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
    )
    # The explicit attribute remap: the material ID follows exact cell ancestry.
    markers = tuple(attribute.name for attribute in result.attributes)
    current = result
    solution = _periodic_poisson(current)
    epochs = [
        {
            "epoch": 0,
            **solution["record"],
            "acceptance": _epoch_acceptance(current, point_ids, material),
        }
    ]
    transitions: list[dict[str, Any]] = []
    hierarchy: phx.meshing.BisectionHierarchy | None = None
    coarse = solution
    last_refinement = None
    # The donor cycle always refines once, then continues to the unchanged target.
    while last_refinement is None or solution["record"]["l2_error"] > _TARGET_L2_ERROR:
        lock, statistics = _matched_bisection_lock(current.mesh)
        adaptation = phx.meshing.execute_mesh_adaptation(
            phx.meshing.prepare_mesh_adaptation(
                current,
                phx.meshing.MarkedMeshAdaptation(
                    np.asarray(lock.cell_global_ids, dtype=np.int64),
                    hierarchy=hierarchy,
                    periodic_edge_lock=lock,
                    inherited_cell_markers=markers,
                ),
                policy=policy,
            )
        )
        coarse, last_refinement = solution, adaptation
        hierarchy, lineage_id = _accepted_bisection(adaptation, "refinement")
        current = adaptation.target
        transfer = _nested_transition_oracle(adaptation, refine=True)
        solution = _periodic_poisson(current)
        transitions.append(
            {
                "operation": "refine",
                "marked_cells": int(np.asarray(lock.cell_global_ids).size),
                "edge_lock_id": lock.lock_id,
                "matched_plan": statistics,
                "hierarchy_id": hierarchy.hierarchy_id,
                "lineage_id": lineage_id,
                "transfer": transfer,
                "compatible_transfer": _compatible_transition(adaptation),
            }
        )
        epochs.append(
            {
                "epoch": len(epochs),
                **solution["record"],
                "acceptance": _epoch_acceptance(current, point_ids, material),
            }
        )
    if last_refinement is None:
        raise RuntimeError("The original lifecycle requires at least one refinement.")
    marks = _coarsening_marks(coarse, last_refinement, _TARGET_L2_ERROR)
    if marks.size == 0:
        raise RuntimeError("No accuracy-admissible periodic sibling family can coarsen.")
    # The indicator certifies only the last refinement's sibling families, so the
    # coarse epoch's vertices stay; restored parents cannot cascade further.
    fine = current.mesh
    retained = np.intersect1d(
        np.asarray(fine.vertex_global_ids, dtype=np.int64),
        np.asarray(last_refinement.source.mesh.vertex_global_ids, dtype=np.int64),
    )
    vertices = fine.entity_set(0)
    adaptation = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            current,
            phx.meshing.MarkedMeshAdaptation(
                (), marks, hierarchy=hierarchy, inherited_cell_markers=markers
            ),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION,
                protected_scopes=(
                    MeshingScope(
                        fine.mesh_id,
                        fine.numeric_version,
                        MeshingEntityKind.MESH,
                        0,
                        vertices.entity_set_id,
                        retained,
                    ),
                ),
            ),
        )
    )
    coarse_hierarchy, coarse_lineage_id = _accepted_bisection(adaptation, "coarsening")
    before = sum(block.cell_count for block in current.mesh.blocks)
    after = sum(block.cell_count for block in adaptation.target.mesh.blocks)
    if after >= before:
        raise RuntimeError("Accuracy-controlled periodic coarsening removed no cells.")
    transfer = _nested_transition_oracle(adaptation, refine=False)
    current = adaptation.target
    solution = _periodic_poisson(current)
    transitions.append(
        {
            "operation": "coarsen",
            "marked_cells": int(marks.size),
            "removed_cells": before - after,
            "coarse_indicator_l2": float(np.sqrt(np.sum(coarse["cell_squared_errors"]))),
            "hierarchy_id": coarse_hierarchy.hierarchy_id,
            "lineage_id": coarse_lineage_id,
            "transfer": transfer,
            "compatible_transfer": _compatible_transition(adaptation),
        }
    )
    epochs.append(
        {
            "epoch": len(epochs),
            **solution["record"],
            "acceptance": _epoch_acceptance(current, point_ids, material),
        }
    )
    if solution["record"]["l2_error"] > _TARGET_L2_ERROR:
        raise RuntimeError(
            f"Final periodic epoch error {solution['record']['l2_error']} exceeds {_TARGET_L2_ERROR}."
        )
    return (
        {
            "target_l2_error": _TARGET_L2_ERROR,
            "indicator": "retained-coarse-Galerkin-cell-physical-L2",
            "epochs": epochs,
            "transitions": transitions,
        },
        current,
        solution,
    )


def _cold_state(result: Any, values: np.ndarray, /) -> Any:
    from phydrax.solver._finite_element_schedule import (
        FiniteElementAcceptedFieldLayout,
        FiniteElementAcceptedState,
        FiniteElementFieldTransferSemantics,
    )

    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("triangle", 1)
    )
    semantics = FiniteElementFieldTransferSemantics.NONCONSERVATIVE_INTERPOLATION
    return FiniteElementAcceptedState(
        (np.asarray(values, dtype=np.float64),),
        np.asarray(0.0, dtype=np.float64),
        0,
        result.mesh.topology_id,
        result.result_id,
        "qualification:skew-torus-periodic-poisson",
        field_layouts=(
            FiniteElementAcceptedFieldLayout(field, transfer_semantics=semantics),
        ),
        state_version=1,
    )


def _cold_identity(result: Any, /) -> dict[str, Any]:
    witness = result.mesh.periodic_topology.exact_coordinate_source
    return {
        "result_id": result.result_id,
        "mesh_id": result.mesh.mesh_id,
        "periodic_topology_id": result.mesh.periodic_topology.periodic_topology_id,
        "witness": None if witness is None else type(witness).__name__,
        "witness_source_map_id": None if witness is None else witness.source_map_id,
    }


def _read_cold(directory: Path, expected: dict[str, Any], /) -> dict[str, Any]:
    """Fresh-process reconstruction: identities, exact acceptance, continued solve."""
    import os

    from phydrax.meshing._model_codec import (
        read_model_artifact,
        restore_model_archive_value,
    )
    from phydrax.meshing._result import CellMeshingResult
    from phydrax.solver._finite_element_schedule import FiniteElementRestartManifest

    result = read_model_artifact(
        directory / "periodic-result.phx",
        expected_type=CellMeshingResult,
        restore=restore_model_archive_value,
        limits=None,
    )
    manifest = read_model_artifact(
        directory / "periodic-state.phx",
        expected_type=FiniteElementRestartManifest,
        restore=FiniteElementRestartManifest.from_stored_fields,
        limits=None,
    )
    identity = _cold_identity(result)
    if identity != expected["identity"]:
        raise RuntimeError(
            "Cold periodic reconstruction changed its accepted identities."
        )
    state = manifest.state
    if (
        state.accepted_id != expected["accepted_id"]
        or state.topology_id != result.mesh.topology_id
        or state.prepared_id != result.result_id
    ):
        raise RuntimeError("Cold periodic state lost its result/topology binding.")
    acceptance = _epoch_acceptance(
        result, np.asarray(expected["point_ids"], dtype=np.int64), expected["material"]
    )
    solution = _periodic_poisson(result)
    stored = np.asarray(state.fields[0], dtype=np.float64)
    if stored.shape != solution["values"].shape:
        raise RuntimeError("Cold periodic state changed its lifted DOF layout.")
    difference = float(np.max(np.abs(stored - solution["values"]), initial=0.0))
    if difference > _TRANSFER_TOLERANCE:
        raise RuntimeError(
            "Cold periodic continued solve differs from the accepted state."
        )
    if solution["record"]["l2_error"] > _TARGET_L2_ERROR:
        raise RuntimeError("Cold periodic continued solve misses the original L2 target.")
    return {
        "pid": os.getpid(),
        "identity": identity,
        "accepted_id": state.accepted_id,
        "acceptance": acceptance,
        "continued_solve": solution["record"],
        "stored_state_max_difference": difference,
        "stored_state_bitwise": bool(np.array_equal(stored, solution["values"])),
    }


def _cold_reconstruction(
    result: Any, solution: dict[str, Any], point_ids: np.ndarray, material: bool, /
) -> dict[str, Any]:
    """Write the final accepted epoch, then reconstruct it in a fresh process."""
    import os
    import subprocess
    import sys
    import tempfile

    from phydrax.meshing._model_codec import write_model_artifact
    from phydrax.solver._finite_element_schedule import FiniteElementRestartManifest

    state = _cold_state(result, solution["values"])
    expected = {
        "identity": _cold_identity(result),
        "accepted_id": state.accepted_id,
        "point_ids": [int(value) for value in point_ids],
        "material": material,
    }
    with tempfile.TemporaryDirectory(prefix="phydrax-periodic-cold-") as directory:
        write_model_artifact(Path(directory) / "periodic-result.phx", result, limits=None)
        write_model_artifact(
            Path(directory) / "periodic-state.phx",
            FiniteElementRestartManifest(state),
            limits=None,
        )
        reader = subprocess.run(
            (sys.executable, "-m", __name__, "read", directory, json.dumps(expected)),
            capture_output=True,
            text=True,
        )
    if reader.returncode:
        raise RuntimeError(
            f"Periodic cold reader failed:\n{reader.stdout}\n{reader.stderr}"
        )
    restored = json.loads(reader.stdout)
    if restored["pid"] == os.getpid():
        raise RuntimeError("Periodic cold writer and reader must be separate processes.")
    return {"producer_pid": os.getpid(), "expected": expected, "consumer": restored}


def _permuted_material_mesh(
    mesh: CellMesh,
    regions: np.ndarray,
    /,
) -> tuple[CellMesh, np.ndarray]:
    order = np.arange(mesh.coordinates.shape[0] - 1, -1, -1, dtype=np.int64)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size, dtype=np.int64)
    cell_order = np.arange(
        sum(block.cell_count for block in mesh.blocks) - 1, -1, -1, dtype=np.int64
    )
    topology = mesh.periodic_topology
    if topology is None:
        raise RuntimeError("Material source permutation requires periodic topology.")
    new_blocks = tuple(
        CellBlock(
            block.name,
            block.cell_kind,
            inverse[np.asarray(block.vertices, dtype=np.int64)[cell_order]],
            global_ids=np.asarray(block.global_ids, dtype=np.int64)[cell_order],
        )
        for block in mesh.blocks
    )
    lifted = CellMesh(
        np.asarray(mesh.coordinates)[order],
        new_blocks,
        vertex_global_ids=np.asarray(mesh.vertex_global_ids, dtype=np.int64)[order],
    )
    old_representatives = np.asarray(topology.vertex_representatives, dtype=np.int64)
    new_representatives = inverse[old_representatives[order]]
    new_shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)[order]
    periodic = PeriodicMeshTopology(
        lifted, topology.cell, new_representatives, new_shifts
    )
    permuted = CellMesh(
        np.asarray(mesh.coordinates)[order],
        new_blocks,
        vertex_global_ids=np.asarray(mesh.vertex_global_ids, dtype=np.int64)[order],
        periodic_topology=periodic,
    )
    return permuted, np.asarray(regions, dtype=np.int32)[cell_order]


def _material_cell_ids(mesh: CellMesh, /) -> np.ndarray:
    return np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
    )


def _native_result(
    profile: _Profile,
    source: NativePeriodicSource,
    mesh: CellMesh,
    regions: np.ndarray | None,
    /,
) -> Any:
    specification = _make_specification(source, mesh, profile, regions)
    return (
        NativeMeshingProvider(NativeMeshingOptions("periodic_delaunay"))
        .plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )


def _profile_record(profile: _Profile, /) -> dict[str, object]:
    payload = _payload(profile)
    revision = _source_revision(payload)
    source_id = _SOURCE_IDS[profile.case]
    point_numerators = np.asarray(payload["fractional_seed_numerators"], dtype=np.int64)
    point_ids = _point_ids(point_numerators, profile.count)
    record: dict[str, object] = {
        "profile_id": profile.profile_id,
        "case": profile.case,
        "source_id": source_id,
        "source_revision": revision,
        "authored_input": payload,
        "source_provenance": {
            "migration_record": "tools/meshing_qualification/native_case_migration.json#/workflow_contracts/periodic-system",
            "donor_artifact": "tools/meshing_qualification/scenarios/periodic.py::_periodic_source_payload",
            "donor_commit": "c56614a9fe52c4a8f56484874e91fde6498ec9ff",
            "carrier_representation": "exact-algebraic",
            "native_coordinate_carrier": "binary64-exact-for-authored-dyadic-values",
            "resolution": profile.count,
            "fractional_sites": int(point_ids.size),
            "point_id_rule": "100 + 16*(8/count*site_i) + (8/count*site_j)",
            "stable_source_point_ids": point_ids.tolist(),
        },
        "producer_route": "NativeMeshingProvider(periodic_delaunay)",
        "oracle_tolerances": {
            "fractional_site_projection_absolute": _SOURCE_COORDINATE_TOLERANCE,
            "quotient_measure_absolute": _MEASURE_TOLERANCE,
            "winding_and_orbit_identity": "exact",
            "field_transfer_residual": _TRANSFER_TOLERANCE,
        },
    }
    stage = "native_dependency_preflight"
    try:
        meshcore = require_meshcore()
        vectors = np.asarray(payload["vectors"], dtype=np.float64)
        cell = PeriodicCell(vectors)
        points = (point_numerators.astype(np.float64) / profile.count) @ vectors
        expected_fractional = tuple(
            (Fraction(int(row[0]), profile.count), Fraction(int(row[1]), profile.count))
            for row in point_numerators
        )
        if _fractional_rows(points) != expected_fractional:
            raise RuntimeError(
                "Authored periodic point coordinates differ from their exact rational fractional-site oracle."
            )
        record["exact_input_site_oracle"] = {
            "fractional_sites_match_exact_rationals": True,
            "exact_point_coordinates": True,
        }
        orbits = PeriodicPointOrbits(cell, points, point_ids)
        if not np.allclose(
            np.asarray(orbits.fractional_points, dtype=np.float64),
            np.asarray([[float(value) for value in row] for row in expected_fractional]),
            rtol=0.0,
            atol=_SOURCE_COORDINATE_TOLERANCE,
        ):
            raise RuntimeError(
                f"PeriodicPointOrbits fractional-coordinate projection exceeds {_SOURCE_COORDINATE_TOLERANCE}."
            )
        # This independent source-construction route checks stable IDs and permutation invariance.
        point_mesh = periodic_delaunay_mesh(orbits).mesh
        permutation = np.arange(point_ids.size - 1, -1, -1, dtype=np.int64)
        permuted_orbits = PeriodicPointOrbits(
            cell, points[permutation], point_ids[permutation]
        )
        permuted_point_mesh = periodic_delaunay_mesh(permuted_orbits).mesh
        if point_mesh.mesh_id != permuted_point_mesh.mesh_id:
            raise RuntimeError(
                "Periodic Delaunay output changed under authored point-row permutation."
            )
        source, source_mesh, source_regions = _mesh_source(
            profile, payload, cell, points, point_ids
        )
        if profile.material_feature:
            if source_regions is None:
                raise RuntimeError(
                    "Material-feature profile has no authored source-region table."
                )
            source_fractional = _fractional_coordinates(source_mesh)
            authored_simplices = payload["simplices"]
            authored_shifts = payload["shifts"]
            expected_triangles = tuple(
                tuple(
                    (
                        Fraction(int(point_numerators[site, 0]), profile.count)
                        + int(shift[0]),
                        Fraction(int(point_numerators[site, 1]), profile.count)
                        + int(shift[1]),
                    )
                    for site, shift in zip(simplex, shifts, strict=True)
                )
                for simplex, shifts in zip(
                    authored_simplices, authored_shifts, strict=True
                )
            )
            actual_triangles = tuple(
                tuple(source_fractional[vertex] for vertex in vertices)
                for _, vertices in _cells(source_mesh)
            )
            if actual_triangles != expected_triangles or not np.array_equal(
                source_regions,
                np.asarray(payload["cell_regions"], dtype=np.int32),
            ):
                raise RuntimeError(
                    "Represented periodic source does not preserve the donor simplices, windings, and cell-region map."
                )
            record["material_source_input_oracle"] = {
                "authored_simplices_and_windings_match_exact_coordinates": True,
                "authored_region_assignment_matches": True,
                "source_triangle_count": len(expected_triangles),
            }
            source_cell_ids = _material_cell_ids(source_mesh)
            source_region_by_id = {
                int(identifier): int(code)
                for identifier, code in zip(source_cell_ids, source_regions, strict=True)
            }
            source_feature_ids = _source_feature_ids(source_mesh)
            source_measure, source_material, _ = _measure_oracles(
                source_mesh, source_region_by_id
            )
            if source_measure != Fraction(1) or source_material != {
                0: Fraction(1, 2),
                1: Fraction(1, 2),
            }:
                raise RuntimeError(
                    f"Authored exact material source differs from its half-domain oracle: area={source_measure}, materials={source_material}."
                )
            source_interface = _material_interface_oracle(
                source_mesh, source_region_by_id
            )
            expected_interface = sqrt(17.0) / 2.0
            if abs(source_interface["length"] - expected_interface) > _MEASURE_TOLERANCE:
                raise RuntimeError(
                    "Authored material interface differs from exact sqrt(17)/2 length."
                )
            source_edge_map = periodic_entity_orbit_map(source_mesh, 1)
            feature_rows = np.flatnonzero(
                np.isin(
                    np.asarray(source_edge_map.lifted_entity_ids, dtype=np.int64),
                    source_feature_ids,
                )
            )
            feature_orbits = {
                int(value)
                for value in np.asarray(
                    source_edge_map.quotient_entity_ids, dtype=np.int64
                )[
                    np.asarray(source_edge_map.lifted_to_quotient, dtype=np.int64)[
                        feature_rows
                    ]
                ]
            }
            if feature_orbits != set(source_interface["interface_quotient_ids"]):
                raise RuntimeError(
                    "Authored material feature IDs do not match the exact region-interface quotient."
                )
            record["source_material_interface"] = {
                "authored_feature_edge_ids": source_feature_ids.tolist(),
                "source_material_measure_exact": {
                    str(code): str(value) for code, value in source_material.items()
                },
                "source_interface_quotient_ids": source_interface[
                    "interface_quotient_ids"
                ],
                "source_interface_length": source_interface["length"],
                "source_interface_length_exact_oracle": "sqrt(17)/2",
            }
        else:
            source_cell_ids = np.zeros((0,), dtype=np.int64)
            source_region_by_id = None
            source_feature_ids = np.zeros((0,), dtype=np.int64)
        source_area, _, source_cell_count = _measure_oracles(
            source_mesh, source_region_by_id
        )
        if abs(float(source_area) - 1.0) > _MEASURE_TOLERANCE:
            raise RuntimeError(
                f"Authored source quotient measure {float(source_area)} differs from exact lattice area 1."
            )
        record["source_binding_id"] = source.binding_id
        record["authored_source_mesh_id"] = source_mesh.mesh_id
        record["authored_source_cell_count"] = source_cell_count
        record["authored_source_measure_exact"] = str(source_area)
        stage = "native_provider_execution"
        specification = _make_specification(source, source_mesh, profile, source_regions)
        result = _native_result(profile, source, source_mesh, source_regions)
        accepted = accepted_result_record(result)
        if result.provider.name != "phydrax-native":
            raise RuntimeError(f"Unexpected periodic producer {result.provider.name!r}.")
        quotient = _assert_quotient(result.mesh, point_ids)
        target_area, _, target_cell_count = _measure_oracles(result.mesh)
        if abs(float(target_area) - 1.0) > _MEASURE_TOLERANCE:
            raise RuntimeError(
                f"Adapted periodic quotient measure {float(target_area)} differs from exact lattice measure 1."
            )
        generation = _exact_generation_oracle(result, specification)
        material_record = None
        if profile.material_feature:
            if source_regions is None:
                raise RuntimeError(
                    "Material-feature profile has no physical source-region data."
                )
            material_record = _check_material_mapping(
                result, source_mesh, source_regions, source_cell_ids
            )
        stage = "periodic_permutation_provider_execution"
        if profile.material_feature:
            if source_regions is None:
                raise RuntimeError(
                    "Material permutation attempt has no authored source-region data."
                )
            permuted_mesh, permuted_regions = _permuted_material_mesh(
                source_mesh, source_regions
            )
            permuted_source = NativePeriodicSource(
                permuted_mesh,
                source.source_id,
                source.source_revision,
                cell_regions=permuted_regions,
            )
        else:
            permuted_orbits = PeriodicPointOrbits(
                cell, points[permutation], point_ids[permutation]
            )
            permuted_source = NativePeriodicSource(
                permuted_orbits,
                source.source_id,
                source.source_revision,
            )
            permuted_mesh = point_mesh
            permuted_regions = None
        permuted_result = _native_result(
            profile, permuted_source, permuted_mesh, permuted_regions
        )
        accepted_result_record(permuted_result)
        if (
            result.mesh.mesh_id != permuted_result.mesh.mesh_id
            or result.mesh.periodic_topology.periodic_topology_id
            != permuted_result.mesh.periodic_topology.periodic_topology_id
        ):
            raise RuntimeError(
                "Native periodic provider output changed under source-row permutation."
            )
        if profile.material_feature:
            first_material = _material_values(result)
            second_material = _material_values(permuted_result)
            if first_material != second_material:
                raise RuntimeError(
                    "Material-ID mapping changed under source-row permutation."
                )
        stage = "marked_bisection_lifecycle"
        lifecycle, final, solution = _marked_bisection_lifecycle(
            result, point_ids, profile.material_feature
        )
        stage = "cold_reconstruction"
        cold = _cold_reconstruction(final, solution, point_ids, profile.material_feature)
        stage = "final_result_record"
        record.update(
            {
                "status": "passed",
                "route": result.provider.name,
                "result": accepted,
                "meshcore_identity": meshcore,
                "runtime": runtime_identity(),
                "quotient": quotient,
                "lattice_measure": {
                    "exact_oracle": "1",
                    "measured_exact_binary_area": str(target_area),
                    "absolute_tolerance": _MEASURE_TOLERANCE,
                },
                "construction": {
                    "source_cells": source_cell_count,
                    "target_cells": target_cell_count,
                    "exact_generation_oracle": generation,
                },
                "permutation_invariance": {
                    "point_delaunay_mesh_id": point_mesh.mesh_id,
                    "provider_mesh_id": result.mesh.mesh_id,
                    "preserved": True,
                },
                "marked_bisection_lifecycle": lifecycle,
                "cold_reconstruction": cold,
                "material_interface": material_record,
            }
        )
        return record
    except Exception as error:
        record.update(
            {
                "status": "failed",
                "failure": {
                    "stage": stage,
                    "exception_type": type(error).__name__,
                    "message": str(error),
                },
                "attempted_source_profile": profile.profile_id,
                "attempted_source_id": source_id,
                "attempted_source_revision": revision,
            }
        )
        return record


def native_periodic_profile_matrix() -> dict[str, object]:
    migration = json.loads(_MIGRATION.read_text(encoding="utf-8"))
    workflow = next(
        item
        for item in migration["workflow_contracts"]
        if item["workflow"] == "periodic-system"
    )
    source_profiles = tuple(workflow["profiles"])
    records: dict[str, dict[str, object]] = {}
    for profile_id in source_profiles:
        profile = _PROFILES.get(profile_id)
        if profile is None:
            records[profile_id] = {
                "profile_id": profile_id,
                "status": "failed",
                "failure": {
                    "stage": "migration_source_mapping",
                    "exception_type": "MissingProfileSourceMapping",
                    "message": "The migration source profile has no matching authored periodic fixture mapping in this module.",
                },
            }
        else:
            records[profile_id] = _profile_record(profile)
    passed = all(record.get("status") == "passed" for record in records.values())
    return {
        "status": "passed" if passed else "failed",
        "workflow": "periodic-system",
        "migration_record": "tools/meshing_qualification/native_case_migration.json#/workflow_contracts/periodic-system",
        "producer_route": "NativeMeshingProvider(periodic_delaunay)",
        "required_profile_count": len(source_profiles),
        "profile_record_count": len(records),
        "profiles": records,
        "oracle_tolerances": {
            "fractional_site_projection_absolute": _SOURCE_COORDINATE_TOLERANCE,
            "quotient_measure_absolute": _MEASURE_TOLERANCE,
            "winding_and_orbit_identity": "exact",
            "field_transfer_residual": _TRANSFER_TOLERANCE,
        },
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4 or sys.argv[1] != "read":
        raise SystemExit("Usage: native_periodic_profiles read <directory> <expected>")
    print(
        json.dumps(_read_cold(Path(sys.argv[2]), json.loads(sys.argv[3])), sort_keys=True)
    )
