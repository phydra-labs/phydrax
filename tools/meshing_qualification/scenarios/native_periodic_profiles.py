from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax.discretization import CellBlock, CellMesh
from phydrax.discretization._cell_mesh import SimplicialConnectivity
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization._periodic_exact import exact_periodic_element
from phydrax.discretization._periodic_topology import PeriodicMeshTopology
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
from phydrax.meshing._lineage import VertexInterpolationStencil
from phydrax.meshing._organization import RegionRole
from phydrax.meshing._periodic import (
    periodic_delaunay_mesh,
    periodic_entity_orbit_map,
    periodic_orbit_closure,
    PeriodicPointOrbits,
    transfer_periodic_p1_vertices,
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


def _fractional_coordinates(mesh: CellMesh, /) -> tuple[tuple[Fraction, Fraction], ...]:
    return _fractional_rows(np.asarray(mesh.coordinates, dtype=np.float64))


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


def _source_feature_ids(mesh: CellMesh, /) -> np.ndarray:
    connectivity = mesh.connectivity
    if (
        not isinstance(connectivity, SimplicialConnectivity)
        or connectivity.dimension != 2
    ):
        raise RuntimeError("Periodic material features require triangle edge incidence.")
    edge_ids: list[int] = np.asarray(
        mesh.entity_set(1).entity_ids, dtype=np.int64
    ).tolist()
    edge_rows: list[list[int]] = np.asarray(
        connectivity.entities[1], dtype=np.int64
    ).tolist()
    fractional = _fractional_coordinates(mesh)
    selected: list[int] = []
    for edge_id, endpoints in zip(edge_ids, edge_rows, strict=True):
        x0 = fractional[endpoints[0]][0]
        x1 = fractional[endpoints[1]][0]
        x0 -= x0.numerator // x0.denominator
        x1 -= x1.numerator // x1.denominator
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
        size_compliance=SizeCompliancePolicy(relative_tolerance=0.5),
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
    fractional: tuple[tuple[Fraction, Fraction], ...],
    /,
) -> Fraction:
    a, b, c = (fractional[index] for index in triangle)
    determinant = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return determinant / 2


def _measure_oracles(
    mesh: CellMesh,
    cell_regions: dict[int, int] | None = None,
    /,
) -> tuple[Fraction, dict[int, Fraction], int]:
    fractional = _fractional_coordinates(mesh)
    area = Fraction(0)
    material_measures: dict[int, Fraction] = {}
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
    connectivity = mesh.connectivity
    if (
        not isinstance(connectivity, SimplicialConnectivity)
        or connectivity.dimension != 2
    ):
        raise RuntimeError(
            "Periodic material interfaces require triangle edge incidence."
        )
    edge_rows: list[list[int]] = np.asarray(
        connectivity.entities[1], dtype=np.int64
    ).tolist()
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
        x = x0 - x0.numerator // x0.denominator
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


def _exact_stencil(
    source: CellMesh, target: CellMesh, /
) -> tuple[VertexInterpolationStencil, dict[int, tuple[Fraction, Fraction, Fraction]]]:
    source_fractional = _fractional_coordinates(source)
    target_fractional = _fractional_coordinates(target)
    target_map = periodic_entity_orbit_map(target, 0)
    source_ids = np.asarray(source.vertex_global_ids, dtype=np.int64)
    target_ids = np.asarray(target_map.lifted_entity_ids, dtype=np.int64)
    target_row_by_id = {
        int(identifier): row
        for row, identifier in enumerate(
            np.asarray(target.vertex_global_ids, dtype=np.int64)
        )
    }
    source_cells = sorted(_cells(source), key=lambda record: record[0])
    triangles = [
        (cell_id, tuple(vertices), tuple(source_fractional[index] for index in vertices))
        for cell_id, vertices in source_cells
    ]
    source_ids_by_target = np.empty((target_ids.size, 3), dtype=np.int64)
    exact_rows: list[tuple[Fraction, Fraction, Fraction]] = []
    exact_by_target: dict[int, tuple[Fraction, Fraction, Fraction]] = {}
    for target_row, identifier in enumerate(target_ids.tolist()):
        point = target_fractional[target_row_by_id[int(identifier)]]
        candidates: list[
            tuple[int, int, int, tuple[int, ...], tuple[Fraction, Fraction, Fraction]]
        ] = []
        for cell_id, vertex_rows, triangle in triangles:
            a, b, c = triangle
            ab = (b[0] - a[0], b[1] - a[1])
            ac = (c[0] - a[0], c[1] - a[1])
            denominator = ab[0] * ac[1] - ab[1] * ac[0]
            if denominator == 0:
                continue
            for shift_x in range(-2, 3):
                for shift_y in range(-2, 3):
                    shifted = (point[0] + shift_x, point[1] + shift_y)
                    relative = (shifted[0] - a[0], shifted[1] - a[1])
                    weight_b = (relative[0] * ac[1] - relative[1] * ac[0]) / denominator
                    weight_c = (ab[0] * relative[1] - ab[1] * relative[0]) / denominator
                    weight_a = Fraction(1) - weight_b - weight_c
                    weights = (weight_a, weight_b, weight_c)
                    if min(weights) >= 0:
                        candidates.append(
                            (cell_id, shift_x, shift_y, vertex_rows, weights)
                        )
        if not candidates:
            raise RuntimeError(
                f"Target periodic vertex ID {identifier} lies outside every exact source triangle image."
            )
        _, _, _, vertex_rows, weights = min(
            candidates, key=lambda item: (item[0], item[1], item[2])
        )
        source_ids_by_target[target_row] = source_ids[
            np.asarray(vertex_rows, dtype=np.int64)
        ]
        exact_rows.append(weights)
        exact_by_target[int(identifier)] = weights
    stencil = VertexInterpolationStencil(
        source.entity_set(0).entity_set_id,
        target.entity_set(0).entity_set_id,
        target_ids,
        source_ids_by_target,
        np.asarray(
            [[float(value) for value in row] for row in exact_rows], dtype=np.float64
        ),
        np.ones(source_ids_by_target.shape, dtype=np.bool_),
        exact_weights=tuple(exact_rows),
    )
    return stencil, exact_by_target


def _periodic_scalar(point: tuple[Fraction, Fraction], /) -> Fraction:
    x, y = point
    x -= x.numerator // x.denominator
    y -= y.numerator // y.denominator
    x_term, y_term = x * (1 - x), y * (1 - y)
    return 1 + x_term + 2 * y_term + x_term * y_term


def _field_transfer_oracle(source: CellMesh, target: CellMesh, /) -> dict[str, Any]:
    stencil, exact_weights_by_target = _exact_stencil(source, target)
    source_map = periodic_entity_orbit_map(source, 0)
    target_map = periodic_entity_orbit_map(target, 0)
    source_fractional = _fractional_coordinates(source)
    source_representatives = np.asarray(
        source_map.representative_lifted_rows, dtype=np.int64
    )
    quotient_source = np.asarray(
        [_periodic_scalar(source_fractional[int(row)]) for row in source_representatives],
        dtype=object,
    )
    quotient_source_float = np.asarray(
        [float(value) for value in quotient_source], dtype=np.float64
    )
    source_by_id = {
        int(identifier): quotient_source[int(orbit)] * int(sign)
        for identifier, orbit, sign in zip(
            np.asarray(source_map.lifted_entity_ids, dtype=np.int64),
            np.asarray(source_map.lifted_to_quotient, dtype=np.int64),
            np.asarray(source_map.orientation_signs, dtype=np.int64),
            strict=True,
        )
    }
    transferred = transfer_periodic_p1_vertices(
        source, target, stencil, quotient_source_float
    )
    exact_lifted: dict[int, Fraction] = {}
    for identifier, weights in exact_weights_by_target.items():
        stencil_row = int(
            np.flatnonzero(
                np.asarray(stencil.target_global_ids, dtype=np.int64) == identifier
            )[0]
        )
        source_row_ids = np.asarray(stencil.source_global_ids, dtype=np.int64)[
            stencil_row
        ]
        exact_lifted[identifier] = sum(
            (
                weight * source_by_id[int(source_id)]
                for weight, source_id in zip(weights, source_row_ids, strict=True)
            ),
            Fraction(0),
        )
    target_ids = np.asarray(target_map.lifted_entity_ids, dtype=np.int64)
    target_orbit = np.asarray(target_map.lifted_to_quotient, dtype=np.int64)
    representatives = np.asarray(target_map.representative_lifted_rows, dtype=np.int64)
    quotient_expected = np.asarray(
        [float(exact_lifted[int(target_ids[int(row)])]) for row in representatives],
        dtype=np.float64,
    )
    residual = float(
        np.max(
            np.abs(np.asarray(transferred, dtype=np.float64) - quotient_expected),
            initial=0.0,
        )
    )
    if residual > _TRANSFER_TOLERANCE:
        raise RuntimeError(
            f"Periodic P1 field transfer residual {residual} exceeds {_TRANSFER_TOLERANCE}."
        )
    for quotient_row in range(target_map.quotient_entity_ids.size):
        members = np.flatnonzero(target_orbit == quotient_row)
        exact_values = {exact_lifted[int(target_ids[row])] for row in members}
        if len(exact_values) != 1:
            raise RuntimeError(
                "Exact target field values do not commute with periodic vertex identification."
            )
    constants = transfer_periodic_p1_vertices(
        source,
        target,
        stencil,
        np.ones((source_map.quotient_entity_ids.size,), dtype=np.float64),
    )
    constant_residual = float(
        np.max(np.abs(np.asarray(constants, dtype=np.float64) - 1.0), initial=0.0)
    )
    if constant_residual > _TRANSFER_TOLERANCE:
        raise RuntimeError(
            f"Periodic P1 transfer does not preserve constants: residual={constant_residual}."
        )

    source_fractional = _fractional_coordinates(source)
    target_fractional = _fractional_coordinates(target)
    source_integral = Fraction(0)
    for _, vertices in _cells(source):
        area = _fractional_area(vertices, source_fractional)
        values = [
            source_by_id[int(source_ids)]
            for source_ids in np.asarray(source.vertex_global_ids)[
                np.asarray(vertices, dtype=np.int64)
            ]
        ]
        source_integral += area * sum(values, Fraction(0)) / 3
    source_actual_by_id = {
        int(identifier): Fraction.from_float(
            float(quotient_source_float[int(orbit)] * int(sign))
        )
        for identifier, orbit, sign in zip(
            np.asarray(source_map.lifted_entity_ids, dtype=np.int64),
            np.asarray(source_map.lifted_to_quotient, dtype=np.int64),
            np.asarray(source_map.orientation_signs, dtype=np.int64),
            strict=True,
        )
    }
    source_actual_integral = Fraction(0)
    for _, vertices in _cells(source):
        area = _fractional_area(vertices, source_fractional)
        values = [
            source_actual_by_id[int(identifier)]
            for identifier in np.asarray(source.vertex_global_ids)[
                np.asarray(vertices, dtype=np.int64)
            ]
        ]
        source_actual_integral += area * sum(values, Fraction(0)) / 3
    target_integral = Fraction(0)
    target_values_by_id = {
        int(identifier): exact_lifted[int(identifier)] for identifier in target_ids
    }
    for _, vertices in _cells(target):
        area = _fractional_area(vertices, target_fractional)
        values = [
            target_values_by_id[int(np.asarray(target.vertex_global_ids)[row])]
            for row in vertices
        ]
        target_integral += area * sum(values, Fraction(0)) / 3
    target_actual_by_id = {
        int(identifier): Fraction.from_float(float(transferred[int(orbit)] * int(sign)))
        for identifier, orbit, sign in zip(
            target_ids,
            target_orbit,
            np.asarray(target_map.orientation_signs, dtype=np.int64),
            strict=True,
        )
    }
    target_actual_integral = Fraction(0)
    for _, vertices in _cells(target):
        area = _fractional_area(vertices, target_fractional)
        values = [
            target_actual_by_id[int(identifier)]
            for identifier in np.asarray(target.vertex_global_ids)[
                np.asarray(vertices, dtype=np.int64)
            ]
        ]
        target_actual_integral += area * sum(values, Fraction(0)) / 3
    exact_conservation_residual = abs(float(target_integral - source_integral))
    actual_conservation_residual = abs(
        float(target_actual_integral - source_actual_integral)
    )
    if (
        max(exact_conservation_residual, actual_conservation_residual)
        > _TRANSFER_TOLERANCE
    ):
        raise RuntimeError(
            "Periodic P1 transfer violates exact/algebraic conservation: "
            f"exact_residual={exact_conservation_residual}, "
            f"binary64_action_residual={actual_conservation_residual}."
        )
    return {
        "source_quotient_dofs": int(source_map.quotient_entity_ids.size),
        "target_quotient_dofs": int(target_map.quotient_entity_ids.size),
        "stencil_id": stencil.stencil_id,
        "exact_p1_commuting_residual": residual,
        "constant_preservation_residual": constant_residual,
        "exact_integral_source": str(source_integral),
        "exact_integral_target": str(target_integral),
        "exact_conservative_transfer_residual": exact_conservation_residual,
        "conservative_transfer_residual": actual_conservation_residual,
        "binary64_integral_source": str(source_actual_integral),
        "binary64_integral_target": str(target_actual_integral),
        "tolerance": _TRANSFER_TOLERANCE,
    }


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
        if target_cell_count <= source_cell_count:
            raise RuntimeError(
                f"Periodic metric adaptation did not refine this source: source_cells={source_cell_count}, target_cells={target_cell_count}."
            )
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
        stage = "exact_periodic_p1_transfer"
        transfer = _field_transfer_oracle(source_mesh, result.mesh)
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
                "adaptation": {
                    "source_cells": source_cell_count,
                    "target_cells": target_cell_count,
                    "refined": target_cell_count > source_cell_count,
                    "orbit_synchronized": True,
                },
                "permutation_invariance": {
                    "point_delaunay_mesh_id": point_mesh.mesh_id,
                    "provider_mesh_id": result.mesh.mesh_id,
                    "preserved": True,
                },
                "compatible_field_transfer": transfer,
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
