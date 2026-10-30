from __future__ import annotations

from fractions import Fraction
from typing import Any

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization import PolyhedralConnectivity
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.discretization.finite_volume import (
    HybridDiffusionBoundary,
    HybridMimeticDiffusion,
    UnstructuredFiniteVolumePlan,
)
from phydrax.discretization.vem import prepare_polyhedral_h1_virtual_element_3d
from phydrax.geometry import CommonRefinementStatus, prepare_common_refinement
from phydrax.linalg import (
    ConjugateGradient,
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
)
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingScope,
    RegionSeed,
    UniformSizeControl,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._controls import RegionControl
from phydrax.meshing._organization import RegionRole
from phydrax.meshing._periodic import periodic_entity_orbit_map, PeriodicPointOrbits
from phydrax.meshing._plc_source import PiecewiseLinearComplex
from phydrax.meshing._polyhedral_generation import NativePolyhedralSchedule
from phydrax.meshing._scope import MeshingEntityKind
from phydrax.meshing._sizing import SizeControlStrength
from phydrax.meshing.providers import NativeMeshingOptions, NativeMeshingProvider
from phydrax.meshing.providers._native_periodic import NativePeriodicSource
from phydrax.meshing.providers._native_sources import NativePolyhedralSource

from .._runtime import accepted_result_record, runtime_identity


_CONTRACT = SpatialCoordinateContract.si()
_SOURCE_REVISION = "c56614a9fe52c4a8f56484874e91fde6498ec9ff"
_PROFILE_MATRIX = (
    ("polyhedral-material-L", "material-L", 1),
    ("polyhedral-material-L:larger", "material-L", 2),
    ("polyhedral-periodic-cube", "periodic-cube", 1),
    ("polyhedral-periodic-cube:larger", "periodic-cube", 2),
    ("polyhedral-stretched-L", "stretched-L", 1),
    ("polyhedral-stretched-L:larger", "stretched-L", 2),
)
_TOLERANCES = {
    "region_measure_relative": 1.0e-10,
    "fv_affine_error": 2.0e-8,
    "fv_normalized_residual": 2.0e-8,
    "vem_solution_error": 2.0e-8,
    "remap_content_absolute": 1.0e-12,
    "remap_transpose_absolute": 1.0e-12,
}


def _periodic_point_orbits(source: NativePeriodicSource, /) -> PeriodicPointOrbits:
    domain = source.domain
    if not isinstance(domain, PeriodicPointOrbits):
        raise TypeError(
            "This polyhedral periodic workflow requires authored point-orbit input."
        )
    return domain


def _material_l_complex(name: str, scale: float, /) -> PiecewiseLinearComplex:
    """Recreate the original three-voxel L PLC with stable current source IDs."""
    face_loops = (
        (0, 2, 3, 1),
        (4, 5, 7, 6),
        (0, 1, 5, 4),
        (2, 6, 7, 3),
        (0, 4, 6, 2),
        (1, 3, 7, 5),
    )
    vertices: list[tuple[float, float, float]] = []
    vertex_positions: dict[tuple[int, int, int], int] = {}
    facets_by_vertices: dict[tuple[int, ...], list[tuple[tuple[int, ...], int]]] = {}
    for cube, region in (((0, 0, 0), 0), ((1, 0, 0), 1), ((0, 1, 0), 0)):
        corners: list[int] = []
        for corner in range(8):
            integer_point = (
                cube[0] + (corner & 1),
                cube[1] + ((corner >> 1) & 1),
                cube[2] + ((corner >> 2) & 1),
            )
            if integer_point not in vertex_positions:
                vertex_positions[integer_point] = len(vertices)
                vertices.append(
                    (
                        scale * integer_point[0],
                        float(integer_point[1]),
                        float(integer_point[2]),
                    )
                )
            corners.append(vertex_positions[integer_point])
        for loop in face_loops:
            face = tuple(corners[index] for index in loop)
            facets_by_vertices.setdefault(tuple(sorted(face)), []).append((face, region))

    polygons: list[np.ndarray] = []
    facet_regions: list[tuple[int, int]] = []
    for entries in facets_by_vertices.values():
        loop, region = entries[0]
        other = entries[1][1] if len(entries) == 2 else -1
        if other != region:
            polygons.append(np.asarray(loop, dtype=np.int64))
            facet_regions.append((other, region))
    region_pairs, polygon_facets = np.unique(
        np.asarray(facet_regions, dtype=np.int64), axis=0, return_inverse=True
    )
    vertex_ids = tuple(
        f"{name}:vertex:{row}:{tuple(float(value) for value in point)}"
        for row, point in enumerate(vertices)
    )
    facet_ids = tuple(
        f"{name}:facet:region-pair:{int(pair[0])}:{int(pair[1])}" for pair in region_pairs
    )
    return PiecewiseLinearComplex(
        np.asarray(vertices, dtype=np.float64),
        polygons,
        polygon_facets,
        region_pairs,
        ("left", "right"),
        segments=np.empty((0, 2), dtype=np.int64),
        vertex_ids=vertex_ids,
        vertex_occurrence_paths=((),) * len(vertex_ids),
        facet_ids=facet_ids,
        facet_occurrence_paths=((),) * len(facet_ids),
        segment_ids=(),
        segment_occurrence_paths=(),
        region_occurrence_paths=((), ()),
    )


def _l_sites(
    name: str, scale: float, resolution: int, /
) -> tuple[np.ndarray, tuple[str, ...]]:
    denominator = 1 << (resolution - 1).bit_length()
    sites = np.column_stack(
        (
            np.full(resolution, 0.5 * scale, dtype=np.float64),
            (np.arange(resolution, dtype=np.float64) + 0.5) * (2.0 / denominator),
            np.full(resolution, 0.5, dtype=np.float64),
        )
    )
    return sites, tuple(f"{name}:power-site:{row}" for row in range(resolution))


def _l_source(
    profile: str, geometry_kind: str, resolution: int, /
) -> tuple[
    NativePolyhedralSource, VolumeMeshingSpec, NativeMeshingOptions, dict[str, object]
]:
    scale = 2.0 if geometry_kind == "stretched-L" else 1.0
    source_id = f"polyhedral-{geometry_kind}"
    complex_ = _material_l_complex(source_id, scale)
    sites, site_ids = _l_sites(source_id, scale, resolution)
    weights = np.zeros((sites.shape[0],), dtype=np.float64)
    revision = canonical_fingerprint(
        {
            "complex": complex_.complex_id,
            "sites": sites.tolist(),
            "coordinate_unit": "meter",
        }
    )
    source = NativePolyhedralSource(
        complex_,
        source_id,
        revision,
        sites=sites,
        weights=weights,
        site_ids=site_ids,
    )
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    region_id_by_name = {
        complex_.region_ids[row]: int(complex_.region_entity_ids[row])
        for row in range(len(complex_.region_ids))
    }
    region_controls = tuple(
        RegionControl(
            MeshingScope(
                source.source_id,
                source.source_revision,
                MeshingEntityKind.GEOMETRY,
                3,
                source.scope_entity_set_id(3),
                np.asarray((region_id_by_name[region],), dtype=np.int64),
            ),
            region,
            f"{region}-material",
            RegionRole.SOLID,
        )
        for region in complex_.region_ids
    )
    seeds = (
        RegionSeed(
            np.asarray((0.5 * scale, 0.5, 0.5), dtype=np.float64),
            "left",
            "left-material",
            RegionRole.SOLID,
        ),
        RegionSeed(
            np.asarray((1.5 * scale, 0.5, 0.5), dtype=np.float64),
            "right",
            "right-material",
            RegionRole.SOLID,
        ),
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("polyhedron",))),
        scope,
        VolumeFillStrategy.POLYHEDRAL,
        size_controls=(
            UniformSizeControl(
                scope,
                10.0 * scale,
                strength=SizeControlStrength.SOFT,
            ),
        ),
        region_controls=region_controls,
        region_seeds=seeds,
    )
    options = NativeMeshingOptions(
        "plc_restricted_power",
        polyhedral_schedule=NativePolyhedralSchedule(require_every_site=True),
    )
    region_volume_scale = Fraction(2) if scale == 1.0 else Fraction(4)
    exact_regions = {"left": region_volume_scale, "right": region_volume_scale / 2}
    source_record: dict[str, object] = {
        "kind": "analytic-generated-fixture",
        "profile_id": profile,
        "locator": "examples/native_polyhedral_meshing.py::material_domain",
        "donor_registry": "tools/_meshing_cases.py::prepare_polyhedral_case",
        "donor_commit": _SOURCE_REVISION,
        "complex_id": complex_.complex_id,
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "source_binding_id": source.binding_id,
        "geometry": "three unit voxels in an L, with a two-region material interface",
        "stretch_x": scale,
        "resolution": resolution,
        "site_ids": list(site_ids),
        "sites_m": sites.tolist(),
        "weights": weights.tolist(),
        "region_ids": list(complex_.region_ids),
        "facet_ids": list(complex_.facet_ids),
        "facet_region_pairs": np.asarray(complex_.facet_regions, dtype=np.int64).tolist(),
        "expected_region_measures_exact_m3": {
            region: str(measure) for region, measure in exact_regions.items()
        },
        "soft_size_control_m": 10.0 * scale,
        "request_id": specification.specification_id,
    }
    return source, specification, options, source_record


def _adapt_l_source(
    geometry_kind: str,
    base: NativePolyhedralSource,
    /,
) -> tuple[
    NativePolyhedralSource, VolumeMeshingSpec, NativeMeshingOptions, dict[str, object]
]:
    sites = np.asarray(base.sites, dtype=np.float64)
    base_ids = tuple(base.site_ids)
    midpoint_sites = 0.5 * (sites[:-1] + sites[1:])
    added_ids = tuple(
        f"{base.source_id}:inserted-power-site:y-{float(point[1]):.17g}"
        for point in midpoint_sites
    )
    target_sites = np.concatenate((sites, midpoint_sites), axis=0)
    target_ids = (*base_ids, *added_ids)
    order = np.argsort(target_sites[:, 1], kind="stable")
    target_sites = target_sites[order]
    target_ids = tuple(target_ids[int(row)] for row in order)
    complex_ = base.complex
    target_revision = canonical_fingerprint(
        {
            "complex": complex_.complex_id,
            "sites": target_sites.tolist(),
            "coordinate_unit": "meter",
            "parent_source": base.source_revision,
            "operation": "insert-one-site-at-each-adjacent-y-bisector",
        }
    )
    target = NativePolyhedralSource(
        complex_,
        base.source_id,
        target_revision,
        sites=target_sites,
        weights=np.zeros((target_sites.shape[0],), dtype=np.float64),
        site_ids=target_ids,
    )
    # Rebuild the request against the adapted source's exact revision and scope.
    scale = 2.0 if geometry_kind == "stretched-L" else 1.0
    domain_scope = MeshingScope(
        target.source_id,
        target.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        target.scope_entity_set_id(2),
        target.scope_entity_ids(2),
    )
    region_id_by_name = {
        complex_.region_ids[row]: int(complex_.region_entity_ids[row])
        for row in range(len(complex_.region_ids))
    }
    controls = tuple(
        RegionControl(
            MeshingScope(
                target.source_id,
                target.source_revision,
                MeshingEntityKind.GEOMETRY,
                3,
                target.scope_entity_set_id(3),
                np.asarray((region_id_by_name[region],), dtype=np.int64),
            ),
            region,
            f"{region}-material",
            RegionRole.SOLID,
        )
        for region in complex_.region_ids
    )
    seeds = (
        RegionSeed(
            np.asarray((0.5 * scale, 0.5, 0.5)), "left", "left-material", RegionRole.SOLID
        ),
        RegionSeed(
            np.asarray((1.5 * scale, 0.5, 0.5)),
            "right",
            "right-material",
            RegionRole.SOLID,
        ),
    )
    adapted_specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("polyhedron",))),
        domain_scope,
        VolumeFillStrategy.POLYHEDRAL,
        size_controls=(
            UniformSizeControl(
                domain_scope,
                10.0 * scale,
                strength=SizeControlStrength.SOFT,
            ),
        ),
        region_controls=controls,
        region_seeds=seeds,
    )
    options = NativeMeshingOptions(
        "plc_restricted_power",
        polyhedral_schedule=NativePolyhedralSchedule(require_every_site=True),
    )
    exact_regions = {
        "left": Fraction(4 if scale == 2.0 else 2),
        "right": Fraction(2 if scale == 2.0 else 1),
    }
    record: dict[str, object] = {
        "kind": "native-restricted-power-site-insertion",
        "parent_source_id": base.source_id,
        "parent_source_revision": base.source_revision,
        "source_id": target.source_id,
        "source_revision": target.source_revision,
        "source_binding_id": target.binding_id,
        "operation": "insert one authored site at each adjacent source-site y-bisector; retain every parent site ID and weight",
        "parent_site_ids": list(base_ids),
        "inserted_site_ids": list(added_ids),
        "site_ids": list(target_ids),
        "sites_m": target_sites.tolist(),
        "weights": np.zeros((target_sites.shape[0],), dtype=np.float64).tolist(),
        "analytic_region_measures_exact_m3": {
            region: str(measure) for region, measure in exact_regions.items()
        },
        "request_id": adapted_specification.specification_id,
    }
    return target, adapted_specification, options, record


def _periodic_source(
    revision: str = "r1", /, *, refined_sites: int | None = None
) -> tuple[
    NativePeriodicSource, VolumeMeshingSpec, NativeMeshingOptions, dict[str, object]
]:
    cell = PeriodicCell(np.eye(3, dtype=np.float64))
    original_ids = np.asarray((101, 202), dtype=np.int64)
    original_fractional = np.asarray(
        ((0.25, 0.5, 0.5), (0.75, 0.5, 0.5)), dtype=np.float64
    )
    if refined_sites is None:
        fractional = original_fractional
        point_ids = original_ids
        source_revision = revision
        operation = "original prescribed two-site periodic cube"
    else:
        if refined_sites not in (4, 8):
            raise ValueError(
                "The periodic adaptation profiles are the original 4/8-site levels."
            )
        x_coordinates = np.arange(1, refined_sites, dtype=np.float64) / refined_sites
        fractional = np.column_stack(
            (
                x_coordinates,
                np.full(x_coordinates.size, 0.5),
                np.full(x_coordinates.size, 0.5),
            )
        )
        ids: list[int] = []
        next_new_id = 1001
        for x in x_coordinates.tolist():
            if x == 0.25:
                ids.append(101)
            elif x == 0.75:
                ids.append(202)
            else:
                ids.append(next_new_id)
                next_new_id += 1
        point_ids = np.asarray(ids, dtype=np.int64)
        source_revision = canonical_fingerprint(
            {
                "parent_revision": revision,
                "periodic_cell": cell.cell_id,
                "fractional_points": fractional.tolist(),
                "point_ids": point_ids.tolist(),
                "weights": [0.0] * point_ids.size,
            }
        )
        operation = f"retain original site orbits and insert periodic x-axis sites at the authored {refined_sites}-site level"
    points = np.asarray(cell.cartesian(fractional), dtype=np.float64)
    weights = np.zeros((point_ids.size,), dtype=np.float64)
    orbits = PeriodicPointOrbits(cell, points, point_ids)
    source = NativePeriodicSource(
        orbits,
        "polyhedral-domain",
        source_revision,
        site_weights=weights,
    )
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        3,
        source.scope_entity_set_id(3),
        source.scope_entity_ids(3),
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("polyhedron",))),
        scope,
        VolumeFillStrategy.POLYHEDRAL,
        size_controls=(),
    )
    options = NativeMeshingOptions("periodic_power")
    record: dict[str, object] = {
        "kind": "fully-periodic-three-axis-cube",
        "locator": "tools/_meshing_cases.py::_prepare_periodic_polyhedral_case",
        "donor_commit": _SOURCE_REVISION,
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "source_binding_id": source.binding_id,
        "periodic_cell_id": cell.cell_id,
        "periodic_lattice_vectors_m": np.asarray(cell.vectors, dtype=np.float64).tolist(),
        "fractional_sites": fractional.tolist(),
        "sites_m": points.tolist(),
        "site_ids": point_ids.tolist(),
        "weights": weights.tolist(),
        "operation": operation,
        "exact_quotient_measure_m3": "1",
        "request_id": specification.specification_id,
    }
    return source, specification, options, record


def _adapt_periodic_source(
    resolution_scale: int, base: NativePeriodicSource, /
) -> tuple[
    NativePeriodicSource, VolumeMeshingSpec, NativeMeshingOptions, dict[str, object]
]:
    refined_sites = 4 if resolution_scale == 1 else 8
    target, specification, options, record = _periodic_source(
        base.source_revision, refined_sites=refined_sites
    )
    base_orbits = _periodic_point_orbits(base)
    target_orbits = _periodic_point_orbits(target)
    base_rows = {
        int(identifier): np.asarray(point, dtype=np.float64)
        for identifier, point in zip(
            np.asarray(base_orbits.point_ids, dtype=np.int64),
            np.asarray(base_orbits.points, dtype=np.float64),
            strict=True,
        )
    }
    target_rows = {
        int(identifier): np.asarray(point, dtype=np.float64)
        for identifier, point in zip(
            np.asarray(target_orbits.point_ids, dtype=np.int64),
            np.asarray(target_orbits.points, dtype=np.float64),
            strict=True,
        )
    }
    record["parent_source_id"] = base.source_id
    record["parent_source_revision"] = base.source_revision
    record["parent_site_ids"] = sorted(base_rows)
    record["inserted_site_ids"] = sorted(set(target_rows) - set(base_rows))
    record["retained_parent_sites_exact"] = all(
        np.array_equal(base_rows[identifier], target_rows[identifier])
        for identifier in base_rows
    )
    return target, specification, options, record


def _polyhedral_cell_volumes(mesh: phx.discretization.CellMesh, /) -> dict[int, float]:
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("The volume oracle requires native polyhedral connectivity.")
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    face_vertices = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    cell_face_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    cell_faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    cell_face_signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.float64)
    cell_vertex_offsets = np.asarray(connectivity.cell_vertex_offsets, dtype=np.int64)
    cell_vertex_values = np.asarray(connectivity.cell_vertex_values, dtype=np.int64)
    cell_ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    volumes: dict[int, float] = {}
    for cell, identifier in enumerate(cell_ids.tolist()):
        vertex_start, vertex_stop = map(int, cell_vertex_offsets[cell : cell + 2])
        origin = np.mean(
            coordinates[cell_vertex_values[vertex_start:vertex_stop]], axis=0
        )
        signed_volume = 0.0
        face_start, face_stop = map(int, cell_face_offsets[cell : cell + 2])
        for face, sign in zip(
            cell_faces[face_start:face_stop],
            cell_face_signs[face_start:face_stop],
            strict=True,
        ):
            start, stop = map(int, face_offsets[int(face) : int(face) + 2])
            polygon = coordinates[face_vertices[start:stop]] - origin
            for offset in range(1, polygon.shape[0] - 1):
                signed_volume += float(
                    sign
                    * np.dot(
                        polygon[0],
                        np.cross(polygon[offset], polygon[offset + 1]),
                    )
                    / 6.0
                )
        if not np.isfinite(signed_volume) or signed_volume <= 0.0:
            raise RuntimeError(
                f"Polyhedral cell {identifier} has nonpositive independently measured volume {signed_volume}."
            )
        volumes[int(identifier)] = signed_volume
    return volumes


def _material_cell_regions(
    result: phx.meshing.CellMeshingResult, expected_names: tuple[str, ...], /
) -> np.ndarray:
    connectivity = result.mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Material assignment requires canonical polyhedral connectivity.")
    cell_ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    assignments = np.full((cell_ids.size,), -1, dtype=np.int64)
    zones = {
        zone.name: zone
        for zone in result.zones
        if zone.role is phx.meshing.MeshZoneRole.REGION
    }
    if set(zones) != set(expected_names):
        raise RuntimeError(
            f"Native output region zones {sorted(zones)} differ from authored regions {expected_names}."
        )
    for region, name in enumerate(expected_names):
        zone = zones[name]
        if (
            zone.material_id != f"{name}-material"
            or zone.region_role is not RegionRole.SOLID
        ):
            raise RuntimeError(
                f"Output zone {name!r} lost its authored material identity."
            )
        selected = np.isin(cell_ids, np.asarray(zone.scope.entity_ids, dtype=np.int64))
        if not np.any(selected) or np.any(assignments[selected] >= 0):
            raise RuntimeError(
                f"Output material zone {name!r} is empty or overlaps another zone."
            )
        assignments[selected] = region
    if np.any(assignments < 0):
        raise RuntimeError("Native material zones do not assign every polyhedral cell.")
    certification = result.certification
    coverage = None if certification is None else certification.coverage
    if coverage is None:
        raise RuntimeError(
            "The native material result omitted exact source-coverage evidence."
        )
    certified_regions = np.asarray(coverage.cell_regions, dtype=np.int64)
    if not np.array_equal(assignments, certified_regions):
        raise RuntimeError(
            "Native zone membership differs from certified source-region membership."
        )
    return assignments


def _profile_geometry_oracle(
    result: phx.meshing.CellMeshingResult,
    source: NativePolyhedralSource | NativePeriodicSource,
    kind: str,
    expected_regions: dict[str, Fraction],
    /,
) -> tuple[dict[str, object], np.ndarray]:
    mesh = result.mesh
    if not isinstance(mesh.connectivity, PolyhedralConnectivity):
        raise TypeError("The native provider did not publish PolyhedralConnectivity.")
    if not mesh.blocks or any(block.cell_kind != "polyhedron" for block in mesh.blocks):
        raise RuntimeError(
            "The native provider did not preserve the required polyhedron family."
        )
    volumes_by_id = _polyhedral_cell_volumes(mesh)
    cell_ids = np.asarray(mesh.connectivity.cell_global_ids, dtype=np.int64)
    cell_volumes = np.asarray([volumes_by_id[int(identifier)] for identifier in cell_ids])
    total_volume = float(np.sum(cell_volumes, dtype=np.float64))
    exact_total = sum(expected_regions.values(), Fraction(0))
    volume_tolerance = _TOLERANCES["region_measure_relative"]
    if abs(total_volume - float(exact_total)) > volume_tolerance * max(
        1.0, float(exact_total)
    ):
        raise RuntimeError(
            f"Independent face-loop volume {total_volume} differs from exact source measure {exact_total}."
        )
    if kind != "periodic-cube":
        assert isinstance(source, NativePolyhedralSource)
        certification = result.certification
        coverage = None if certification is None else certification.coverage
        if coverage is None:
            raise RuntimeError(
                "Native L-source generation omitted its independent region coverage certificate."
            )
        certified = {
            name: measure
            for name, measure in zip(
                coverage.region_ids, coverage.achieved_region_measures, strict=True
            )
        }
        if set(certified) != set(expected_regions):
            raise RuntimeError(
                f"Certified source region identities {sorted(certified)} differ from {sorted(expected_regions)}."
            )
        measured_regions = {name: Fraction(0) for name in expected_regions}
        assignments = _material_cell_regions(result, tuple(expected_regions))
        region_names = tuple(expected_regions)
        for region_index, volume in zip(assignments, cell_volumes, strict=True):
            measured_regions[region_names[int(region_index)]] += Fraction.from_float(
                float(volume)
            )
        for region, exact_measure in expected_regions.items():
            certified_measure = certified.get(region)
            if certified_measure is None or abs(
                float(certified_measure) - float(exact_measure)
            ) > volume_tolerance * max(1.0, float(exact_measure)):
                raise RuntimeError(
                    f"Independent exact source-region oracle disagrees for {region}: "
                    f"{certified_measure=} {exact_measure=} ."
                )
            if abs(
                float(measured_regions[region]) - float(exact_measure)
            ) > volume_tolerance * max(1.0, float(exact_measure)):
                raise RuntimeError(
                    f"Output face-loop region measure for {region} differs from exact source measure {exact_measure}."
                )
        source_facet_ids = set(map(int, source.complex.facet_entity_ids))
        overlapped_facet_ids = {
            int(overlap.source_facet_id) for overlap in coverage.facet_source_overlaps
        }
        if (
            coverage.source_facet_count != len(source_facet_ids)
            or coverage.covered_source_facet_count != coverage.source_facet_count
            or overlapped_facet_ids != source_facet_ids
        ):
            raise RuntimeError(
                "The exact source-boundary oracle did not cover every authored PLC facet exactly."
            )
        return (
            {
                "output_family": sorted({block.cell_kind for block in mesh.blocks}),
                "analytic_region_measures_exact_m3": {
                    name: str(measure) for name, measure in expected_regions.items()
                },
                "certified_region_measures_m3": {
                    name: float(value)
                    for name, value in certified.items()
                    if value is not None
                },
                "independent_face_loop_region_measures_m3": {
                    name: float(value) for name, value in measured_regions.items()
                },
                "independent_total_measure_m3": total_volume,
                "source_facet_count": int(coverage.source_facet_count),
                "covered_source_facet_count": int(coverage.covered_source_facet_count),
                "source_coverage_id": coverage.certificate_id,
                "coverage_method": str(coverage.coverage_method),
                "volume_tolerance_relative": volume_tolerance,
            },
            assignments,
        )

    assert isinstance(source, NativePeriodicSource)
    source_orbits = _periodic_point_orbits(source)
    topology = mesh.periodic_topology
    if topology is None:
        raise RuntimeError(
            "Native periodic power publication omitted its quotient topology."
        )
    point_ids = np.asarray(source_orbits.point_ids, dtype=np.int64)
    cell_orbits = periodic_entity_orbit_map(mesh, 3)
    if not np.array_equal(
        np.sort(np.asarray(cell_orbits.quotient_entity_ids, dtype=np.int64)),
        np.sort(point_ids),
    ):
        raise RuntimeError(
            "Periodic quotient-cell identities differ from authored power-site orbit IDs."
        )
    boundary_faces = np.asarray(
        topology.quotient.entity_sets[2].subset("boundary").mask, dtype=np.bool_
    )
    if np.any(boundary_faces):
        raise RuntimeError(
            "Periodic quotient surface retained an uncovered exterior face orbit."
        )
    analytic_measure = float(source_orbits.cell.cell_measure)
    if abs(total_volume - analytic_measure) > volume_tolerance * max(
        1.0, analytic_measure
    ):
        raise RuntimeError(
            "Independent periodic polyhedral volume differs from the exact lattice measure."
        )
    return (
        {
            "output_family": sorted({block.cell_kind for block in mesh.blocks}),
            "analytic_region_measures_exact_m3": {
                name: str(measure) for name, measure in expected_regions.items()
            },
            "independent_quotient_measure_m3": total_volume,
            "analytic_lattice_measure_m3": analytic_measure,
            "periodic_cell_id": source_orbits.cell.cell_id,
            "periodic_topology_id": topology.periodic_topology_id,
            "site_ids": point_ids.tolist(),
            "quotient_cell_site_ids": np.asarray(
                cell_orbits.quotient_entity_ids, dtype=np.int64
            ).tolist(),
            "periodic_vertex_orbits": int(
                periodic_entity_orbit_map(mesh, 0).quotient_entity_ids.size
            ),
            "periodic_face_orbits": int(
                periodic_entity_orbit_map(mesh, 2).quotient_entity_ids.size
            ),
            "uncovered_quotient_boundary_faces": int(np.count_nonzero(boundary_faces)),
            "volume_tolerance_relative": volume_tolerance,
        },
        np.zeros((cell_ids.size,), dtype=np.int64),
    )


def _polyhedral_fv(
    result: phx.meshing.CellMeshingResult, periodic: bool, /
) -> dict[str, object]:
    discretization = UnstructuredFiniteVolumePlan.from_cell_mesh(result.mesh).prepare()
    if periodic:
        content = float(np.sum(np.asarray(discretization.cell_volumes, dtype=np.float64)))
        error = abs(content - 1.0)
        if error > _TOLERANCES["region_measure_relative"]:
            raise RuntimeError(f"Periodic FV constant-content oracle failed: {error=}.")
        return {
            "method": "UnstructuredFiniteVolumePlan.from_cell_mesh",
            "consumer": "polyhedral FV geometry and constant cell-content integral",
            "cell_count": int(discretization.cell_volumes.shape[0]),
            "constant_density": 1.0,
            "integrated_content": content,
            "analytic_periodic_content": 1.0,
            "absolute_content_error": error,
            "tolerance": _TOLERANCES["region_measure_relative"],
        }

    boundary_faces = np.flatnonzero(np.asarray(discretization.neighbor_cells) < 0)
    centers = np.asarray(discretization.face_centers, dtype=np.float64)
    dirichlet = {
        int(face): float(
            1.0 + 0.2 * centers[face, 0] - 0.3 * centers[face, 1] + 0.4 * centers[face, 2]
        )
        for face in boundary_faces.tolist()
    }
    boundary = HybridDiffusionBoundary(discretization, dirichlet=dirichlet)
    system = HybridMimeticDiffusion(discretization)
    solved = system.solve(1.0, boundary)
    cell_count = int(discretization.cell_volumes.shape[0])
    cell_values = np.asarray(solved.value[:cell_count], dtype=np.float64)
    exact = (
        1.0
        + 0.2 * np.asarray(discretization.cell_centers)[:, 0]
        - 0.3 * np.asarray(discretization.cell_centers)[:, 1]
        + 0.4 * np.asarray(discretization.cell_centers)[:, 2]
    )
    operator, rhs = system.linear_system(1.0, boundary)
    residual = float(jnp.linalg.norm(operator.operator.mv(solved.value) - rhs))
    rhs_norm = float(jnp.linalg.norm(rhs))
    error = float(np.max(np.abs(cell_values - exact), initial=0.0))
    normalized_residual = residual / max(1.0, rhs_norm)
    if (
        not bool(jnp.all(solved.successful))
        or error > _TOLERANCES["fv_affine_error"]
        or normalized_residual > _TOLERANCES["fv_normalized_residual"]
    ):
        raise RuntimeError(
            f"Native polyhedral mimetic FV solve failed its affine oracle: {error=} {normalized_residual=} ."
        )
    return {
        "method": "HybridMimeticDiffusion",
        "cell_count": cell_count,
        "face_count": int(discretization.face_measures.shape[0]),
        "successful": bool(jnp.all(solved.successful)),
        "maximum_affine_error": error,
        "residual_norm": residual,
        "normalized_residual": normalized_residual,
        "exact_field": "u=1+0.2x-0.3y+0.4z",
        "tolerance_affine_error": _TOLERANCES["fv_affine_error"],
        "tolerance_normalized_residual": _TOLERANCES["fv_normalized_residual"],
    }


def _polyhedral_vem(
    result: phx.meshing.CellMeshingResult, periodic: bool, /
) -> dict[str, object]:
    vem = prepare_polyhedral_h1_virtual_element_3d(result.mesh)
    if periodic:
        constant = jnp.ones(
            (vem.dof_count,), dtype=jnp.asarray(vem.evidence.cell_volumes).dtype
        )
        defect = float(jnp.max(jnp.abs(vem.mv(constant)), initial=0.0))
        if defect > _TOLERANCES["vem_solution_error"]:
            raise RuntimeError(
                f"Periodic polyhedral VEM constant nullspace oracle failed: {defect=}."
            )
        return {
            "method": "prepare_polyhedral_h1_virtual_element_3d",
            "consumer": "periodic polyhedral VEM stiffness on the constant quotient field",
            "dofs": vem.dof_count,
            "constant_nullspace_defect": defect,
            "minimum_volume": vem.evidence.minimum_volume,
            "minimum_projector_rank_margin": vem.evidence.minimum_rank_margin,
            "maximum_polynomial_reproduction_defect": vem.evidence.maximum_reproduction_defect,
            "tolerance": _TOLERANCES["vem_solution_error"],
        }

    connectivity = result.mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Polyhedral VEM requires root PolyhedralConnectivity.")
    offsets = np.asarray(connectivity.cell_vertex_offsets, dtype=np.int64)
    vertices = np.asarray(connectivity.cell_vertex_values, dtype=np.int64)
    lumped = np.zeros((connectivity.vertex_count,), dtype=np.float64)
    for cell, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        local_vertices = vertices[int(start) : int(stop)]
        np.add.at(
            lumped,
            local_vertices,
            float(np.asarray(vem.evidence.cell_volumes)[cell]) / local_vertices.size,
        )
    if np.any(~np.isfinite(lumped)) or np.any(lumped <= 0.0):
        raise RuntimeError(
            "VEM reaction solve requires positive mass at every polyhedral vertex."
        )
    mass = jnp.asarray(lumped)
    base = vem.as_linear_operator()
    operator = FunctionLinearOperator(
        lambda values: base.mv(values) + mass * values,
        source=base.source,
        target=base.target,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        ),
    )
    solved = phx.linalg.solve(
        LinearSystem(operator),
        mass,
        policy=LinearSolvePolicy(ConjugateGradient()),
    )
    values = np.asarray(solved.value, dtype=np.float64)
    error = float(np.max(np.abs(values - 1.0), initial=0.0))
    residual = float(jnp.linalg.norm(operator.mv(solved.value) - mass))
    normalized_residual = residual / max(1.0, float(jnp.linalg.norm(mass)))
    if (
        not bool(jnp.all(solved.successful))
        or error > _TOLERANCES["vem_solution_error"]
        or normalized_residual > _TOLERANCES["vem_solution_error"]
    ):
        raise RuntimeError(
            f"Native polyhedral VEM reaction solve failed its constant solution oracle: {error=} {normalized_residual=} ."
        )
    return {
        "method": "degree-one polyhedral H1 VEM reaction-diffusion",
        "dofs": int(values.size),
        "successful": bool(jnp.all(solved.successful)),
        "maximum_constant_solution_error": error,
        "normalized_residual": normalized_residual,
        "exact_field": "u=1 for -Delta(u)+u=1 with natural boundary flux",
        "minimum_volume": vem.evidence.minimum_volume,
        "minimum_projector_rank_margin": vem.evidence.minimum_rank_margin,
        "maximum_polynomial_reproduction_defect": vem.evidence.maximum_reproduction_defect,
        "tolerance_solution_error": _TOLERANCES["vem_solution_error"],
        "tolerance_normalized_residual": _TOLERANCES["vem_solution_error"],
    }


def _site_representation(
    result: phx.meshing.CellMeshingResult,
    source: NativePolyhedralSource | NativePeriodicSource,
    periodic: bool,
    /,
) -> dict[str, object]:
    if periodic:
        assert isinstance(source, NativePeriodicSource)
        source_orbits = _periodic_point_orbits(source)
        orbit_map = periodic_entity_orbit_map(result.mesh, 3)
        input_ids = np.asarray(source_orbits.point_ids, dtype=np.int64)
        output_ids = np.asarray(orbit_map.quotient_entity_ids, dtype=np.int64)
        if not np.array_equal(np.sort(input_ids), np.sort(output_ids)):
            raise RuntimeError(
                "Periodic source-site IDs were not retained as quotient cell orbits."
            )
        periodic_topology = result.mesh.periodic_topology
        if periodic_topology is None:
            raise RuntimeError("Periodic power result omitted its quotient topology.")
        return {
            "status": "passed",
            "input_site_orbit_ids": input_ids.tolist(),
            "represented_quotient_cell_ids": output_ids.tolist(),
            "site_orbit_count": int(output_ids.size),
            "periodic_topology_id": periodic_topology.periodic_topology_id,
        }

    assert isinstance(source, NativePolyhedralSource)
    connectivity = result.mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Restricted-power output lost polyhedral connectivity.")
    cell_ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    attributes = {attribute.name: attribute for attribute in result.attributes}
    site_attribute = attributes.get("power_site_index")
    # MeshingScope stores sorted entity IDs; attribute values follow that order.
    if site_attribute is None or not np.array_equal(
        np.asarray(site_attribute.scope.entity_ids, dtype=np.int64), np.sort(cell_ids)
    ):
        raise RuntimeError(
            "Restricted-power output omitted cell-aligned power-site ownership."
        )
    indices = np.asarray(site_attribute.values, dtype=np.int64)
    if indices.shape != cell_ids.shape:
        raise RuntimeError("Restricted-power site-index values are not cell aligned.")
    represented = set(indices.tolist())
    if represented != set(range(len(source.site_ids))):
        raise RuntimeError(
            "Native restricted-power regeneration did not represent every authored site."
        )
    labels = {label.name for label in result.labels}
    missing = {f"power-site:{site_id}" for site_id in source.site_ids} - labels
    if missing:
        raise RuntimeError(
            f"Native output omitted stable power-site labels: {sorted(missing)}."
        )
    return {
        "status": "passed",
        "input_site_ids": list(source.site_ids),
        "represented_site_indices": sorted(represented),
        "represented_site_count": len(represented),
        "cell_count": int(cell_ids.size),
    }


def _run_native_source(
    source: NativePolyhedralSource | NativePeriodicSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    /,
) -> phx.meshing.CellMeshingResult:
    plan = NativeMeshingProvider(options).plan(
        source,
        specification,
        coordinate_contract=_CONTRACT,
    )
    return plan.execute()


def _record_value(value: Any, /) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, np.ndarray):
        return _record_value(value.tolist())
    if isinstance(value, np.generic):
        return _record_value(value.item())
    if isinstance(value, dict):
        return {str(key): _record_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_record_value(item) for item in value]
    if hasattr(value, "value"):
        return _record_value(value.value)
    return {"type": type(value).__name__}


def _capture_gate(profile: dict[str, Any], gate: str, operation: Any, /) -> Any:
    profile["attempted_gates"].append(gate)
    try:
        evidence = operation()
    except Exception as error:
        profile["gates"][gate] = {
            "status": "failed",
            "error_type": type(error).__name__,
            "detail": str(error),
        }
        return None
    profile["gates"][gate] = {"status": "passed", "evidence": _record_value(evidence)}
    return evidence


def _run_output(
    profile: dict[str, Any],
    result: phx.meshing.CellMeshingResult,
    source: NativePolyhedralSource | NativePeriodicSource,
    kind: str,
    expected_regions: dict[str, Fraction],
    label: str,
    /,
) -> np.ndarray | None:
    _capture_gate(
        profile,
        f"{label}_native_acceptance",
        lambda: accepted_result_record(result),
    )
    internal: dict[str, object] = {}

    def geometry() -> dict[str, object]:
        evidence, assignments = _profile_geometry_oracle(
            result, source, kind, expected_regions
        )
        internal["assignments"] = assignments
        return evidence

    _capture_gate(profile, f"{label}_family_measure_source_coverage", geometry)
    _capture_gate(
        profile,
        f"{label}_power_site_orbit_identity",
        lambda: _site_representation(result, source, kind == "periodic-cube"),
    )
    _capture_gate(
        profile,
        f"{label}_finite_volume_consumption",
        lambda: _polyhedral_fv(result, kind == "periodic-cube"),
    )
    _capture_gate(
        profile,
        f"{label}_vem_consumption",
        lambda: _polyhedral_vem(result, kind == "periodic-cube"),
    )
    assignments = internal.get("assignments")
    return None if assignments is None else np.asarray(assignments, dtype=np.int64)


def _common_refinement_remap(
    source: phx.meshing.CellMeshingResult,
    target: phx.meshing.CellMeshingResult,
    source_regions: np.ndarray,
    target_regions: np.ndarray,
    periodic: bool,
    /,
) -> dict[str, object]:
    if source.mesh.topology_id == target.mesh.topology_id:
        raise RuntimeError(
            "The authored remap target is nested-identical to its source mesh."
        )
    common = prepare_common_refinement(source.mesh, target.mesh)
    if common.status is not CommonRefinementStatus.SUCCESS:
        raise RuntimeError(
            f"Production common refinement refused the actual source/target pair: "
            f"{common.status.value}: {common.evidence.reason}."
        )
    source_rows = np.asarray(common.source_cells, dtype=np.int64)
    target_rows = np.asarray(common.target_cells, dtype=np.int64)
    overlap_volumes = np.asarray(common.volumes, dtype=np.float64)
    source_measures = np.asarray(common.source_measures, dtype=np.float64)
    target_measures = np.asarray(common.target_measures, dtype=np.float64)
    if (
        source_rows.size == 0
        or np.any(source_rows < 0)
        or np.any(source_rows >= source_regions.size)
        or np.any(target_rows < 0)
        or np.any(target_rows >= target_regions.size)
    ):
        raise RuntimeError(
            "Common-refinement CSR entries are empty or outside the authored cell tables."
        )
    if periodic:
        source_cell_values = np.ones(source_measures.shape, dtype=np.float64)
    else:
        source_cell_values = np.where(source_regions == 0, 2.0, 5.0)
        if np.any(source_regions[source_rows] != target_regions[target_rows]):
            raise RuntimeError(
                "Conservative native overlap transferred content across the material interface."
            )
    values = source_cell_values[source_rows]
    transferred_mass = np.bincount(
        target_rows,
        weights=overlap_volumes * values,
        minlength=target_measures.size,
    )
    transferred = transferred_mass / target_measures
    target_values = transferred
    if periodic:
        inventory_by_region = {
            "material-0": {
                "before": float(np.dot(source_cell_values, source_measures)),
                "after": float(np.dot(target_values, target_measures)),
            }
        }
    else:
        target_cell_values = np.where(target_regions == 0, 2.0, 5.0)
        inventory_by_region = {}
        for region, name in enumerate(("left", "right")):
            before = float(
                np.dot(
                    source_cell_values[source_regions == region],
                    source_measures[source_regions == region],
                )
            )
            after = float(
                np.dot(
                    target_values[target_regions == region],
                    target_measures[target_regions == region],
                )
            )
            inventory_by_region[name] = {"before": before, "after": after}
            if abs(before - after) > _TOLERANCES["remap_content_absolute"]:
                raise RuntimeError(
                    f"Conservative overlap remap changed {name} inventory by {abs(before - after)}."
                )
        if (
            np.max(np.abs(target_values - target_cell_values), initial=0.0)
            > _TOLERANCES["remap_content_absolute"]
        ):
            raise RuntimeError(
                "Exact material-region constants were not retained by the overlap remap."
            )
    content_defects = {
        name: abs(values_["after"] - values_["before"])
        for name, values_ in inventory_by_region.items()
    }
    if max(content_defects.values(), default=0.0) > _TOLERANCES["remap_content_absolute"]:
        raise RuntimeError(
            f"Native conservative remap content oracle failed: {content_defects}."
        )

    target_cotangent = np.linspace(0.25, 1.25, target_measures.size, dtype=np.float64)
    transpose = np.bincount(
        source_rows,
        weights=(overlap_volumes / target_measures[target_rows])
        * target_cotangent[target_rows],
        minlength=source_measures.size,
    )
    forward_pairing = float(np.dot(target_cotangent, transferred))
    transpose_pairing = float(np.dot(transpose, source_cell_values))
    transpose_defect = abs(forward_pairing - transpose_pairing)
    if transpose_defect > _TOLERANCES["remap_transpose_absolute"]:
        raise RuntimeError(
            f"Common-refinement transpose identity failed: {transpose_defect=}."
        )

    overlap_pairs = np.unique(np.column_stack((source_rows, target_rows)), axis=0)
    source_fanout = np.bincount(overlap_pairs[:, 0], minlength=source_measures.size)
    target_fanin = np.bincount(overlap_pairs[:, 1], minlength=target_measures.size)
    nonnested = bool(np.any(source_fanout > 1) and np.any(target_fanin > 1))
    if not nonnested:
        raise RuntimeError(
            "The generated site-insertion target did not exercise overlaps in both directions; "
            "the required remap is not genuinely non-nested."
        )
    return {
        "status": "passed",
        "method": "phydrax.geometry.prepare_common_refinement",
        "semantics": "physical overlap-volume cell-average transfer",
        "source_topology_id": source.mesh.topology_id,
        "target_topology_id": target.mesh.topology_id,
        "source_cells": int(source_measures.size),
        "target_cells": int(target_measures.size),
        "overlap_pieces": int(overlap_volumes.size),
        "nonnested": nonnested,
        "material_or_periodic_content": inventory_by_region,
        "absolute_content_defects": content_defects,
        "maximum_relative_source_coverage_defect": common.evidence.maximum_relative_source_defect,
        "maximum_relative_target_coverage_defect": common.evidence.maximum_relative_target_defect,
        "transpose_pairing_defect": transpose_defect,
        "content_tolerance_absolute": _TOLERANCES["remap_content_absolute"],
        "transpose_tolerance_absolute": _TOLERANCES["remap_transpose_absolute"],
        "common_refinement_id": common.refinement_id,
    }


def _profile_remap(
    source: phx.meshing.CellMeshingResult,
    target: phx.meshing.CellMeshingResult,
    kind: str,
    source_regions: np.ndarray | None,
    target_regions: np.ndarray | None,
    /,
) -> dict[str, object]:
    periodic = kind == "periodic-cube"
    resolved_source_regions = (
        np.zeros((source.mesh.connectivity.cell_count,), dtype=np.int64)
        if periodic
        else (
            _material_cell_regions(source, ("left", "right"))
            if source_regions is None
            else source_regions
        )
    )
    resolved_target_regions = (
        np.zeros((target.mesh.connectivity.cell_count,), dtype=np.int64)
        if periodic
        else (
            _material_cell_regions(target, ("left", "right"))
            if target_regions is None
            else target_regions
        )
    )
    return _common_refinement_remap(
        source,
        target,
        resolved_source_regions,
        resolved_target_regions,
        periodic,
    )


def _profile_entry(profile_id: str, kind: str, scale: int, /) -> dict[str, Any]:
    return {
        "profile_id": profile_id,
        "geometry_profile": kind,
        "resolution_scale": scale,
        "status": "failed",
        "attempted_gates": [],
        "gates": {},
        "tolerances": dict(_TOLERANCES),
    }


def native_polyhedral() -> dict[str, object]:
    """Exercise the complete original native polyhedral source/profile matrix."""
    profiles: dict[str, dict[str, Any]] = {}
    for profile_id, kind, resolution_scale in _PROFILE_MATRIX:
        profile = _profile_entry(profile_id, kind, resolution_scale)
        profiles[profile_id] = profile
        if kind == "periodic-cube":
            base_source, base_spec, base_options, base_input = _capture_gate(
                profile,
                "original_periodic_source_construction",
                lambda: _periodic_source(),
            ) or (None, None, None, None)
            if (
                base_source is None
                or base_spec is None
                or base_options is None
                or base_input is None
            ):
                continue
            profile["source_input"] = base_input
            target_parts = _capture_gate(
                profile,
                "periodic_site_orbit_insertion_input",
                lambda: _adapt_periodic_source(resolution_scale, base_source),
            )
            if target_parts is None:
                target_source = target_spec = target_options = None
                target_input = None
            else:
                target_source, target_spec, target_options, target_input = target_parts
                profile["adapted_source_input"] = target_input
        else:
            parts = _capture_gate(
                profile,
                "original_analytic_l_source_construction",
                lambda: _l_source(profile_id, kind, 4 * resolution_scale),
            )
            if parts is None:
                continue
            base_source, base_spec, base_options, base_input = parts
            profile["source_input"] = base_input
            adapted_parts = _capture_gate(
                profile,
                "restricted_power_site_insertion_input",
                lambda: _adapt_l_source(kind, base_source),
            )
            if adapted_parts is None:
                target_source = target_spec = target_options = None
                target_input = None
            else:
                target_source, target_spec, target_options, target_input = adapted_parts
                profile["adapted_source_input"] = target_input
        if base_source is None or base_spec is None or base_options is None:
            continue

        if kind == "periodic-cube":
            expected_regions = {"material-0": Fraction(1)}
        else:
            stretch = 2 if kind == "stretched-L" else 1
            expected_regions = {
                "left": Fraction(2 * stretch),
                "right": Fraction(stretch),
            }

        base_result = _capture_gate(
            profile,
            "original_source_native_generation",
            lambda: _run_native_source(base_source, base_spec, base_options),
        )
        base_regions = None
        if base_result is not None:
            base_regions = _run_output(
                profile,
                base_result,
                base_source,
                kind,
                expected_regions,
                "original",
            )
        target_regions = None
        if target_source is None or target_spec is None or target_options is None:
            profile["gates"]["native_site_regeneration"] = {
                "status": "failed",
                "detail": "The profile-specific adapted source could not be constructed; no substitute source was used.",
            }
            profile["attempted_gates"].append("native_site_regeneration")
            target_result = None
        else:
            target_result = _capture_gate(
                profile,
                "adapted_source_native_generation",
                lambda: _run_native_source(target_source, target_spec, target_options),
            )
            base_ids = (
                set(
                    map(
                        int,
                        np.asarray(
                            _periodic_point_orbits(base_source).point_ids, dtype=np.int64
                        ),
                    )
                )
                if isinstance(base_source, NativePeriodicSource)
                else set(base_source.site_ids)
            )
            target_ids = (
                set(
                    map(
                        int,
                        np.asarray(
                            _periodic_point_orbits(target_source).point_ids,
                            dtype=np.int64,
                        ),
                    )
                )
                if isinstance(target_source, NativePeriodicSource)
                else set(target_source.site_ids)
            )
            identity_evidence = {
                "parent_site_ids": sorted(base_ids),
                "target_site_ids": sorted(target_ids),
                "retained_parent_site_ids": sorted(base_ids & target_ids),
                "inserted_site_ids": sorted(target_ids - base_ids),
                "source_revision_changed": base_source.source_revision
                != target_source.source_revision,
            }
            _capture_gate(
                profile,
                "source_bound_site_regeneration_identity",
                lambda: _validate_regeneration_identity(
                    base_source, target_source, identity_evidence
                ),
            )
            _capture_gate(
                profile,
                "native_site_regeneration",
                lambda: _native_site_regeneration_evidence(
                    base_result, target_result, identity_evidence
                ),
            )
            if target_result is not None:
                target_regions = _run_output(
                    profile,
                    target_result,
                    target_source,
                    kind,
                    expected_regions,
                    "adapted",
                )
        profile["profile_controls"] = {
            "original_site_count": (
                int(np.asarray(base_source.sites).shape[0])
                if isinstance(base_source, NativePolyhedralSource)
                else int(np.asarray(_periodic_point_orbits(base_source).point_ids).size)
            ),
            "resolution_scale": resolution_scale,
            "adapted_site_count": (
                None
                if target_source is None
                else (
                    int(target_source.sites.shape[0])
                    if isinstance(target_source, NativePolyhedralSource)
                    else int(
                        np.asarray(_periodic_point_orbits(target_source).point_ids).size
                    )
                )
            ),
            "mandatory_family": "polyhedron",
            "fill_strategy": "polyhedral",
            "resource_controls": "none",
        }
        if base_result is not None and target_result is not None:
            _capture_gate(
                profile,
                "nonnested_conservative_remap_and_transpose",
                lambda: _profile_remap(
                    base_result,
                    target_result,
                    kind,
                    base_regions,
                    target_regions,
                ),
            )
        else:
            profile["attempted_gates"].append(
                "nonnested_conservative_remap_and_transpose"
            )
            profile["gates"]["nonnested_conservative_remap_and_transpose"] = {
                "status": "failed",
                "detail": "The real source/target native output required for conservative remap was not produced; no fallback mesh was substituted.",
            }
        profile["status"] = (
            "passed"
            if all(gate["status"] == "passed" for gate in profile["gates"].values())
            and len(profile["gates"]) == len(profile["attempted_gates"])
            else "failed"
        )

    return {
        "status": "passed"
        if all(profile["status"] == "passed" for profile in profiles.values())
        else "failed",
        "workflow": "original nonconvex material-L, stretched-L and periodic-cube sources -> native restricted/periodic power polyhedra -> FV/VEM consumption -> site insertion and physical common-refinement remap",
        "source_registry": "tools/meshing_qualification/native_case_migration.json#polyhedral-system",
        "source_provenance": {
            "historical_source_tree_commit": _SOURCE_REVISION,
            "material_L_fixture": "examples/native_polyhedral_meshing.py::material_domain",
            "case_registry": "tools/_meshing_cases.py::prepare_polyhedral_case",
            "profile_resolution_scales": {
                name: scale for name, _, scale in _PROFILE_MATRIX
            },
            "resource_quotas": "not supplied",
        },
        "profiles": profiles,
        "runtime": runtime_identity(),
    }


def _validate_regeneration_identity(
    base: NativePolyhedralSource | NativePeriodicSource,
    target: NativePolyhedralSource | NativePeriodicSource,
    evidence: dict[str, object],
    /,
) -> dict[str, object]:
    if not evidence["source_revision_changed"]:
        raise RuntimeError("Site insertion did not create a distinct source revision.")
    if not evidence["inserted_site_ids"]:
        raise RuntimeError("Site insertion did not add a stable source-site identity.")
    if not evidence["retained_parent_site_ids"]:
        raise RuntimeError(
            "Site insertion did not retain any parent source-site identity."
        )
    if type(base) is not type(target):
        raise TypeError("Adapted and parent source authority types differ.")
    if base.source_id != target.source_id:
        raise RuntimeError("Site insertion changed the source family's stable source ID.")
    if isinstance(base, NativePolyhedralSource) and isinstance(
        target, NativePolyhedralSource
    ):
        if base.complex.complex_id != target.complex.complex_id:
            raise RuntimeError(
                "Restricted-power site insertion changed the authored L PLC."
            )
        base_rows = {
            identifier: (np.asarray(point), float(weight))
            for identifier, point, weight in zip(
                base.site_ids, base.sites, base.weights, strict=True
            )
        }
        target_rows = {
            identifier: (np.asarray(point), float(weight))
            for identifier, point, weight in zip(
                target.site_ids, target.sites, target.weights, strict=True
            )
        }
        for identifier in base_rows:
            if (
                identifier not in target_rows
                or not np.array_equal(
                    base_rows[identifier][0], target_rows[identifier][0]
                )
                or base_rows[identifier][1] != target_rows[identifier][1]
            ):
                raise RuntimeError(
                    f"Adapted restricted-power source changed parent site {identifier!r}."
                )
    elif isinstance(base, NativePeriodicSource) and isinstance(
        target, NativePeriodicSource
    ):
        base_orbits = _periodic_point_orbits(base)
        target_orbits = _periodic_point_orbits(target)
        if base_orbits.cell.cell_id != target_orbits.cell.cell_id:
            raise RuntimeError(
                "Periodic site insertion changed the authored lattice quotient."
            )
        base_rows = {
            int(identifier): np.asarray(point)
            for identifier, point in zip(
                base_orbits.point_ids, base_orbits.points, strict=True
            )
        }
        target_rows = {
            int(identifier): np.asarray(point)
            for identifier, point in zip(
                target_orbits.point_ids, target_orbits.points, strict=True
            )
        }
        for identifier in base_rows:
            if identifier not in target_rows or not np.array_equal(
                base_rows[identifier], target_rows[identifier]
            ):
                raise RuntimeError(
                    f"Adapted periodic source changed parent orbit {identifier}."
                )
    return {
        **evidence,
        "retained_parent_site_coordinates_and_weights_exact": True,
        "source_geometry_identity_preserved": True,
    }


def _native_site_regeneration_evidence(
    base_result: phx.meshing.CellMeshingResult | None,
    target_result: phx.meshing.CellMeshingResult | None,
    identity_evidence: dict[str, object],
    /,
) -> dict[str, object]:
    if base_result is None or target_result is None:
        raise RuntimeError(
            "The base and inserted-site native generations must both publish results."
        )
    base_connectivity = base_result.mesh.connectivity
    target_connectivity = target_result.mesh.connectivity
    if not isinstance(base_connectivity, PolyhedralConnectivity) or not isinstance(
        target_connectivity, PolyhedralConnectivity
    ):
        raise TypeError(
            "Native site regeneration must preserve polyhedral connectivity at both epochs."
        )
    base_count = int(base_connectivity.cell_count)
    target_count = int(target_connectivity.cell_count)
    if (
        target_result.mesh.topology_id == base_result.mesh.topology_id
        or target_count <= base_count
    ):
        raise RuntimeError(
            "Inserted source sites did not regenerate a distinct, finer polyhedral mesh."
        )
    return {
        **identity_evidence,
        "base_topology_id": base_result.mesh.topology_id,
        "target_topology_id": target_result.mesh.topology_id,
        "base_cell_count": base_count,
        "target_cell_count": target_count,
        "inserted_site_mesh_regeneration": True,
    }
