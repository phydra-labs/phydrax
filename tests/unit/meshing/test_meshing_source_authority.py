"""Reject self-certified native closures with foreign original authorities."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import meshcore_available
from phydrax._trainable import ArrayRole, resolve_array_roles
from phydrax.discretization import CellGeometrySpec, ExactPlcCellGeometrySource
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.geometry._mesh_certificates import PiecewiseLinearDomain, SourceBoundaryQuery
from phydrax.geometry._meshing_domain import (
    _PhysicalCurveMap,
    MeshingDomain,
    MeshingDomainBoundarySource,
)
from phydrax.lifecycle._meshing_source_families import native_family_audit_policy
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    validate_meshing_source_closure,
    write_meshing_source_closure,
)
from phydrax.meshing._assembly import MeshPart
from phydrax.meshing._contracts import VolumeMeshingSpec
from phydrax.meshing._result import CellMeshingResult
from phydrax.meshing._trace import MeshingStageKind
from phydrax.meshing._volume_generation import declared_plc_domain
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_publication import (
    NativeCertificationRequest,
    publish_native_result,
)
from phydrax.meshing.providers._native_sources import (
    NativeLayerCoreSource,
    NativePlcSource,
)


if TYPE_CHECKING:
    from phydrax.discretization._spaces import DiscreteFieldSpace
    from phydrax.lifecycle._meshing_field_records import (
        MeshingCoefficientIdentityBank,
    )


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="native source publication unavailable"
)
M = phx.meshing


def _closure(
    result: CellMeshingResult,
    source: NativePlcSource | NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
) -> dict[str, object]:
    report = result.certification
    if report is None:
        raise AssertionError(
            "An actual native publication must retain its positive source theorem."
        )
    return {
        "certification_inputs": report.request,
        "report": report,
        "associations": result.associations,
        "generation_part": MeshPart("generation", result),
        "generation_source": source,
        "generation_specification": specification,
        "generation_options": options,
    }


def _self_certified_carrier(
    original: CellMeshingResult,
    source: NativePlcSource | NativeLayerCoreSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    geometry: CellGeometrySpec,
    domain: PiecewiseLinearDomain,
    query: SourceBoundaryQuery | None,
) -> CellMeshingResult:
    report = original.certification
    if report is None:
        raise AssertionError("The actual original carrier must be certified.")
    request = report.request
    provenance = json.loads(original.provenance.content_json)
    assert provenance["kind"] == "mapping"
    values = dict(provenance["items"])
    plan = M.NativeMeshingProvider(options).plan(
        source, specification, coordinate_contract=original.coordinate_contract
    )
    values["source"], values["plan"] = source.binding_id, plan.plan_id
    construction = tuple(
        stage
        for stage in original.trace.stages
        if stage.stage
        not in (MeshingStageKind.GEOMETRY_AUDIT, MeshingStageKind.CERTIFICATION)
    )
    result = publish_native_result(
        original.mesh,
        original.coordinate_contract,
        original.compliance,
        construction,
        original.provider,
        values,
        NativeCertificationRequest(
            request.schedule,
            source.source_id,
            source.source_revision,
            specification.limits,
            domain=domain,
            cell_regions=None
            if request.cell_regions is None
            else np.asarray(request.cell_regions),
            fidelity_source=query,
            fidelity_tolerance=request.fidelity_tolerance,
            junction_vertices=request.junction_vertices,
            scoped_fidelity=request.scoped_fidelity,
        ),
        audit_policy=native_family_audit_policy(source, specification, original.mesh),
        derivative_mode=original.derivative_mode,
        enforced_limits=original.runtime.enforced_limits,
        unenforced_limits=original.runtime.unenforced_limits,
        geometry=geometry,
        boundary=original.boundary,
        patches=original.patches,
        zones=original.zones,
        labels=original.labels,
        attributes=original.attributes,
        associations=original.associations,
    )
    assert (
        result.audit.passed
        and result.certification is not None
        and result.certification.passed
    )
    return result


@pytest.fixture(scope="module")
def plc_generation() -> tuple[
    CellMeshingResult, NativePlcSource, VolumeMeshingSpec, NativeMeshingOptions
]:
    from tests.unit.meshing.test_meshing_restart import _native_family_generation

    part, source, specification, options = _native_family_generation("plc")
    assert isinstance(part.carrier, CellMeshingResult)
    assert isinstance(source, NativePlcSource) and isinstance(
        specification, VolumeMeshingSpec
    )
    assert isinstance(part.carrier.geometry.exact_source, ExactPlcCellGeometrySource)
    validate_meshing_source_closure(
        _closure(part.carrier, source, specification, options)
    )
    return part.carrier, source, specification, options


def _foreign_plc_source(
    source: ExactPlcCellGeometrySource, change: str
) -> ExactPlcCellGeometrySource:
    points, triangles, segments = (
        np.asarray(source.source_points),
        np.asarray(source.source_triangles),
        np.asarray(source.source_segments),
    )
    triangle_ids, segment_ids = (
        np.asarray(source.source_triangle_ids),
        np.asarray(source.source_segment_ids),
    )
    triangle_bounds, segment_bounds = (
        np.asarray(source.source_triangle_bounds),
        np.asarray(source.source_segment_bounds),
    )
    identity, revision = source.domain_source_id, source.domain_source_revision
    match change:
        case "identity":
            identity += "-foreign"
        case "revision":
            revision += "-stale"
        case "points":
            points = np.concatenate(
                (points, np.asarray(((0.25, 0.25, 0.25),), dtype=np.float64))
            )
        case "triangles":
            triangles, triangle_ids = (
                np.concatenate((triangles, triangles[:1])),
                np.concatenate((triangle_ids, triangle_ids[:1])),
            )
            triangle_bounds = np.concatenate((triangle_bounds, triangle_bounds[:1]))
        case "triangle-ids":
            triangle_ids = triangle_ids + 100
        case "segments":
            segments, segment_ids = (
                np.concatenate((segments, segments[:1])),
                np.concatenate(
                    (segment_ids, np.asarray((segment_ids.max() + 1,), dtype=np.int64))
                ),
            )
            segment_bounds = np.concatenate((segment_bounds, segment_bounds[:1]))
        case "segment-ids":
            segment_ids = segment_ids + 100
        case "triangle-bounds":
            triangle_bounds = triangle_bounds + 0.01
        case "segment-bounds":
            segment_bounds = segment_bounds + 0.01
        case _:
            raise AssertionError(f"Unknown original bank change {change!r}.")
    return ExactPlcCellGeometrySource(
        points,
        triangles,
        segments,
        source.vertex_strata,
        source.vertex_rows,
        source.vertex_parameters,
        domain_source_id=identity,
        domain_source_revision=revision,
        source_triangle_ids=triangle_ids,
        source_triangle_bounds=triangle_bounds,
        source_segment_ids=segment_ids,
        source_segment_bounds=segment_bounds,
        maximum_work=source.maximum_work,
        maximum_bits=source.maximum_bits,
    )


@pytest.mark.parametrize(
    "change",
    (
        "identity",
        "revision",
        "points",
        "triangles",
        "triangle-ids",
        "segments",
        "segment-ids",
        "triangle-bounds",
        "segment-bounds",
    ),
)
def test_archive_refuses_self_certified_foreign_plc_source(
    tmp_path: Path,
    plc_generation: tuple[
        CellMeshingResult, NativePlcSource, VolumeMeshingSpec, NativeMeshingOptions
    ],
    change: str,
) -> None:
    original, source, specification, options = plc_generation
    exact = original.geometry.exact_source
    assert isinstance(exact, ExactPlcCellGeometrySource)
    altered = _foreign_plc_source(exact, change)
    geometry = CellGeometrySpec.plc(original.mesh, altered)
    assert geometry.source_coordinates() == original.geometry.source_coordinates()
    assert original.certification is not None and isinstance(
        original.certification.request.domain, PiecewiseLinearDomain
    )
    forged = _self_certified_carrier(
        original,
        source,
        specification,
        options,
        geometry,
        original.certification.request.domain,
        None,
    )
    with pytest.raises(
        ValueError, match="Registered exact PLC geometry|exact PLC source does not bind"
    ):
        write_meshing_source_closure(
            tmp_path / "foreign-plc", _closure(forged, source, specification, options)
        )


def _cube_domain(
    *, parameter_scale: float = 1.0, source_revision: str = "authored-box-planes"
) -> MeshingDomain:
    topology = reference_cell_topology("hexahedron")
    points = np.asarray(topology.vertices, dtype=np.float64)
    edges = topology.entities[1]
    lookup = {tuple(sorted(edge)): index for index, edge in enumerate(edges)}
    curve_endpoints: dict[int, tuple[int, int]] = {}
    patches = []
    for face_index, face in enumerate(topology.entities[2]):
        scale = parameter_scale if face_index == 1 else 1.0
        uv = np.asarray(
            ((0.0, 0.0), (1.0 / scale, 0.0), (1.0 / scale, 1.0), (0.0, 1.0)),
            dtype=np.float64,
        )
        uses = []
        for position, start in enumerate(face):
            end = face[(position + 1) % 4]
            edge = lookup[tuple(sorted((start, end)))]
            owner = curve_endpoints.setdefault(edge, (start, end))
            forward = owner == (start, end)
            first, last = (
                (position, (position + 1) % 4)
                if forward
                else ((position + 1) % 4, position)
            )
            curve = phx.geometry.LineCurve(uv[first], uv[last] - uv[first])
            uses.append(
                phx.geometry.PatchCurveUse(
                    edge, curve, 0.0 if forward else 1.0, 1.0 if forward else 0.0
                )
            )
        plane = phx.geometry.PlanePatch(
            points[face[0]],
            scale * (points[face[1]] - points[face[0]]),
            points[face[3]] - points[face[0]],
        )
        patches.append(phx.geometry.MeshingSurfacePatch(plane, (tuple(uses),)))
    return MeshingDomain(
        tuple(patches),
        tuple(
            phx.geometry.MeshingDomainCurve(*curve_endpoints[index])
            for index in range(len(edges))
        ),
        8,
        source_id="authored-layer-box",
        source_revision=source_revision,
        regions=(
            phx.geometry.MeshingDomainRegion(
                "fluid", tuple((index, 1) for index in range(6))
            ),
        ),
    )


def _cube_query(
    domain: MeshingDomain, *, parameter_scale: float = 1.0
) -> MeshingDomainBoundarySource:
    topology = reference_cell_topology("hexahedron")
    points = np.asarray(topology.vertices, dtype=np.float64)
    cells = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    boundary = np.asarray(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=np.int64)
    chains = []
    for patch, face in enumerate(topology.entities[2]):
        scale = parameter_scale if patch == 1 else 1.0
        charts = np.asarray(
            ((0.0, 0.0), (1.0 / scale, 0.0), (1.0 / scale, 1.0), (0.0, 1.0)),
            dtype=np.float64,
        )
        provenance = []
        for index, use in enumerate(domain.patches[patch].loops[0]):
            if not isinstance(use, phx.geometry.PatchCurveUse):
                raise AssertionError(
                    "The authored plane box has four exact line coedges."
                )
            provenance.append((0, index, use.first, use.last))
        count = charts.shape[0]
        chains.append(
            (
                patch,
                charts,
                np.asarray(points[list(face)]),
                cells,
                boundary,
                np.asarray(provenance, dtype=np.float64),
                np.zeros((count,), dtype=np.bool_),
                np.empty((0,), dtype=np.int64),
                np.empty((0, 2), dtype=np.int64),
                np.empty((0, 2), dtype=np.int64),
            )
        )
    return MeshingDomainBoundarySource(
        domain, tuple(range(6)), resolution=4, chart_triangulations=tuple(chains)
    )


@pytest.fixture(scope="module", params=("represented", "parametric"))
def layer_generation(
    request: pytest.FixtureRequest,
) -> tuple[
    CellMeshingResult, NativeLayerCoreSource, VolumeMeshingSpec, NativeMeshingOptions
]:
    topology = reference_cell_topology("hexahedron")
    points = np.asarray(topology.vertices, dtype=np.float64)
    if request.param == "represented":
        outer = M.PiecewiseLinearComplex(
            points,
            topology.entities[2],
            np.arange(6, dtype=np.int64),
            np.tile(np.asarray((-1, 0), dtype=np.int64), (6, 1)),
            ("fluid",),
        )
        domain = declared_plc_domain(outer, "authored-layer-box")
        physical = _cube_domain(source_revision=domain.source_revision)
    else:
        domain = _cube_domain()
        physical = domain
    query = _cube_query(physical)
    boundary = M.MeshingScope(
        domain.source_id,
        domain.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        physical.entity_set_id(2),
        physical.scope_indices(2),
    )
    # Closed inward layers retain all six original boundary faces; every core
    # face is an explicitly shared generated cap, not an invented source face.
    triangles = np.asarray(
        tuple(
            triangle
            for face in topology.entities[2]
            for triangle in ((face[0], face[2], face[1]), (face[0], face[3], face[2]))
        ),
        dtype=np.int64,
    )
    wall = phx.discretization.CellMesh.from_triangles(points, triangles)
    indices = np.repeat(np.arange(6, dtype=np.int32), 2)
    names = tuple(physical.entity_id(2, int(index)) for index in indices)
    association = M.GeometryAssociation(
        M.GeometryAssociationKind.SURFACE,
        domain.source_id,
        domain.source_revision,
        wall.entity_set(2).entity_set_id,
        wall.entity_set(2).entity_ids,
        names,
        np.zeros(12, dtype=np.float64),
        exact=True,
        source_dimensions=np.full(12, 2, dtype=np.int32),
        source_indices=indices,
        orientations=np.full(12, -1, dtype=np.int8),
    )
    volume_scope = M.MeshingScope(
        domain.source_id,
        domain.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        3,
        physical.entity_set_id(3),
        physical.scope_indices(3),
    )
    control = M.BoundaryLayerControl(
        boundary,
        M.LayerSchedule.geometric(2, 0.1, growth_rate=1.0),
        route=M.BoundaryLayerRoute.ADVANCING,
        volume_scope=volume_scope,
    )
    layers = M.prepare_boundary_layers(
        wall, control, wall_association=association, source_domain=domain
    )
    if layers.cap is None:
        raise AssertionError("Closed inward layers must retain their actual cap.")
    cap_rows = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in layers.cap.blocks]
    )
    core = M.PiecewiseLinearComplex(
        layers.cap.coordinates,
        tuple(cap_rows),
        np.repeat(np.arange(6, dtype=np.int64), 2),
        np.tile(np.asarray((0, -1), dtype=np.int64), (6, 1)),
        ("fluid",),
        boundary="fixed",
    )
    source = NativeLayerCoreSource(
        layers,
        core,
        domain.source_id,
        domain.source_revision,
        vertex_layer_ids=layers.cap_vertices,
        cap_polygon_ids=np.arange(12, dtype=np.int64),
        layer_regions=np.zeros(
            sum(block.cell_count for block in layers.mesh.blocks), dtype=np.int64
        ),
        region_ids=("fluid",),
        core_region_map=np.asarray((0,), dtype=np.int64),
        source_boundary_scope=boundary,
        source_domain=domain,
        fidelity_source=query,
        core_facet_source_ids=np.full(6, -1, dtype=np.int64),
    )
    query_arrays = [
        leaf
        for leaf in jax.tree_util.tree_leaves(query)
        if isinstance(leaf, (jax.Array, np.ndarray))
    ]
    dynamic_arrays = {
        id(leaf)
        for leaf in jax.tree_util.tree_leaves(source)
        if isinstance(leaf, (jax.Array, np.ndarray))
    }
    assert query_arrays and all(id(leaf) in dynamic_arrays for leaf in query_arrays)
    roles = resolve_array_roles(source)
    assert not roles.violations and all(role is ArrayRole.FIXED for role in roles.roles)
    specification = M.VolumeMeshingSpec(
        M.CellMeshingTarget(
            3, 3, M.CellFamilyPolicy(required=("prism", "tetrahedron"), allow_mixed=True)
        ),
        boundary,
        M.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            M.UniformSizeControl(boundary, 2.0, strength=M.SizeControlStrength.SOFT),
        ),
    )
    options = NativeMeshingOptions("layer_core")
    result = (
        M.NativeMeshingProvider(options)
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )
    validate_meshing_source_closure(_closure(result, source, specification, options))
    return result, source, specification, options


@pytest.mark.parametrize("change", ("query", "domain"))
def test_archive_refuses_self_certified_foreign_layer_authority(
    tmp_path: Path,
    layer_generation: tuple[
        CellMeshingResult, NativeLayerCoreSource, VolumeMeshingSpec, NativeMeshingOptions
    ],
    change: str,
) -> None:
    original, source, specification, options = layer_generation
    assert original.certification is not None
    domain = original.certification.request.domain
    assert isinstance(domain, PiecewiseLinearDomain)
    query = source.fidelity_source
    if isinstance(source.source_domain, PiecewiseLinearDomain):
        if change == "query":
            query = _cube_query(
                _cube_domain(parameter_scale=2.0, source_revision=source.source_revision),
                parameter_scale=2.0,
            )
        else:
            domain = PiecewiseLinearDomain(
                domain.vertices,
                domain.facets[::-1],
                domain.facet_regions[::-1],
                domain.region_ids,
                source_id=domain.source_id,
            )
    else:
        assert isinstance(source.source_domain, MeshingDomain)
        query = _cube_query(_cube_domain(parameter_scale=2.0), parameter_scale=2.0)
        if change == "domain":
            source = NativeLayerCoreSource(
                source.layers,
                source.complex,
                source.source_id,
                source.source_revision,
                vertex_layer_ids=source.vertex_layer_ids,
                cap_polygon_ids=source.cap_polygon_ids,
                layer_regions=source.layer_regions,
                region_ids=source.region_ids,
                core_region_map=source.core_region_map,
                source_boundary_scope=source.source_boundary_scope,
                source_domain=source.source_domain,
                fidelity_source=query,
                core_facet_source_ids=source.core_facet_source_ids,
            )
    forged = _self_certified_carrier(
        original, source, specification, options, original.geometry, domain, query
    )
    with pytest.raises(ValueError, match="Layer certificate"):
        write_meshing_source_closure(
            tmp_path / "foreign-layer", _closure(forged, source, specification, options)
        )


@pytest.fixture(scope="module")
def original_curved_brep_domain() -> MeshingDomain:
    geometry = phx.geometry
    lower = geometry.BSplineCurve.bezier(
        np.asarray(((0.0, 0.0), (0.5, 0.2), (1.0, 0.0)), dtype=np.float64),
        np.ones(3, dtype=np.float64),
    )
    upper = geometry.BSplineCurve.bezier(
        np.asarray(((1.0, 0.05), (0.5, 0.25), (0.0, 0.05)), dtype=np.float64),
        np.ones(3, dtype=np.float64),
    )
    loop = geometry.ProfileLoop(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 0.05), (0.0, 0.05)),
        (lower, geometry.ProfileLine(), upper, geometry.ProfileLine()),
    )
    profile = geometry.PlanarProfile(
        geometry.ProfilePlane(x_axis=(0.0, 1.0, 0.0), y_axis=(0.0, 0.0, 1.0)),
        loop,
    )
    model = geometry.brep_extrusion(
        profile,
        (1.0, 0.0, 0.0),
        coordinate_contract=phx.SpatialCoordinateContract.si(),
        source_id="curved-periodic-narrow-gap",
    )
    return MeshingDomain.from_brep(model)


def test_original_curved_brep_domain_archive_fresh_process(
    tmp_path: Path,
    original_curved_brep_domain: MeshingDomain,
) -> None:
    domain = original_curved_brep_domain
    receipt = write_meshing_source_closure(tmp_path / "curved-domain", domain)
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert type(restored) is MeshingDomain
    np.testing.assert_array_equal(
        restored.corner_points.view(np.uint64), domain.corner_points.view(np.uint64)
    )
    assert type(restored.curve_atlas.mapping) is type(domain.curve_atlas.mapping)
    assert type(restored.brep_authority) is type(domain.brep_authority)
    code = """
import json
import sys
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.geometry.brep._model import BRepModel
from phydrax.lifecycle._meshing_sources import read_meshing_source_closure
domain = read_meshing_source_closure(sys.argv[1], expected_content_id=sys.argv[2])
if type(domain) is not MeshingDomain or type(domain.brep_authority) is not BRepModel:
    raise TypeError("Fresh source restoration lost its exact domain or B-Rep owner.")
print(json.dumps([domain.source_id, domain.source_revision, domain.domain_id,
                  domain.authority_id, type(domain.curve_atlas.mapping).__name__,
                  domain.brep_authority.model_id]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(receipt.path), receipt.content_id],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == [
        domain.source_id,
        domain.source_revision,
        domain.domain_id,
        domain.authority_id,
        type(domain.curve_atlas.mapping).__name__,
        domain.authority_id,
    ]


def test_original_curved_brep_geometry_domain_archive(
    tmp_path: Path,
    original_curved_brep_domain: MeshingDomain,
) -> None:
    from phydrax.geometry.brep._model import BRepModel

    model = original_curved_brep_domain.brep_authority
    if type(model) is not BRepModel or model.geometry is None:
        raise AssertionError("Original source requires its full B-Rep geometry owner.")
    domain = MeshingDomain.from_brep_geometry(
        model.geometry,
        model.patches,
        np.asarray(model.orientation),
        source_id=model.source_id,
        source_revision=model.source_revision,
    )
    receipt = write_meshing_source_closure(tmp_path / "curved-geometry-domain", domain)
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert type(restored) is MeshingDomain
    assert type(restored.brep_authority) is type(model.geometry)
    assert restored.authority_id == model.geometry.geometry_id
    np.testing.assert_array_equal(
        restored.corner_points.view(np.uint64), domain.corner_points.view(np.uint64)
    )


@pytest.mark.parametrize(
    "change", ("corners", "edges", "missing-owner", "foreign-domain")
)
def test_original_curved_brep_domain_archive_refuses_wrong_domain(
    tmp_path: Path,
    original_curved_brep_domain: MeshingDomain,
    change: str,
) -> None:
    domain = original_curved_brep_domain
    match change:
        case "corners":
            changed = eqx.tree_at(
                lambda value: value.corner_points,
                domain,
                np.nextafter(domain.corner_points, np.inf),
            )
        case "edges":
            mapping = domain.curve_atlas.mapping
            if type(mapping) is not _PhysicalCurveMap:
                raise AssertionError(
                    "Original B-Rep edges must retain their physical source owner."
                )
            changed = eqx.tree_at(
                lambda value: value.curve_atlas.mapping,
                domain,
                eqx.tree_at(
                    lambda value: value.ranges,
                    mapping,
                    np.asarray(mapping.ranges) + 0.01,
                ),
            )
        case "missing-owner":
            changed = eqx.tree_at(lambda value: value.brep_authority, domain, None)
        case "foreign-domain":
            foreign = MeshingDomain(
                domain.patches,
                domain.curves,
                domain.corner_count,
                source_id=domain.source_id + "-foreign",
                source_revision=domain.source_revision,
                regions=domain.regions,
                tolerance=domain.tolerance,
                accuracy=domain.accuracy,
                source_indices=domain.source_indices,
                source_occurrences=domain.source_occurrences,
                region_source_indices=domain.region_source_indices,
                region_source_occurrences=domain.region_source_occurrences,
                source_kinds=domain.source_kinds,
                authority_id=domain.authority_id,
            )
            changed = eqx.tree_at(
                lambda value: value.brep_authority,
                foreign,
                domain.brep_authority,
                is_leaf=lambda value: value is None,
            )
        case _:
            raise AssertionError(f"Unknown authored domain change {change!r}.")
    with pytest.raises(ValueError, match="freshly certified native source authority"):
        write_meshing_source_closure(tmp_path / f"wrong-domain-{change}", changed)


@pytest.mark.parametrize("change", ("missing-field", "obsolete-field", "unknown-type"))
def test_current_native_recipe_requires_exact_registered_type_and_fieldset(
    change: str,
) -> None:
    from phydrax._model._structure import (
        model_structure_recipe,
        validate_model_structure_recipe,
    )
    from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts
    from phydrax.meshing._controls import RegionSeed

    register_meshing_source_artifacts()
    source = RegionSeed(
        np.asarray([0.25, 0.25, 0.25]),
        "solid",
        "material",
        M.RegionRole.SOLID,
    )
    recipe = model_structure_recipe(source)
    if change == "missing-field":
        recipe["items"].pop()
    elif change == "obsolete-field":
        recipe["items"].append({"kind": "literal", "value": None})
    else:
        recipe["type"] = "phydrax.meshing:UnregisteredNativeSource"
    with pytest.raises(ValueError):
        validate_model_structure_recipe(recipe)


@pytest.mark.parametrize(
    "field,value",
    (
        ("optimizer_method_id", "foreign-policy"),
        ("optimizer_evaluations", -1),
        ("native_work_units", 4),
    ),
)
def test_current_metric_evidence_archive_refuses_changed_policy_counts_and_work(
    field: str, value: object
) -> None:
    from copy import copy

    from phydrax.meshing._tetra_metric import (
        MetricRemeshingEvidence,
        MetricRemeshingStatus,
    )

    evidence = MetricRemeshingEvidence(
        MetricRemeshingStatus.COMPLETE,
        passes=1,
        counts=(0, 0, 0, 0),
        rejected_operations=0,
        work_units=3,
        lengths=np.asarray([1.0]),
        quality=np.asarray([1.0]),
        optimizer_method_id="actual-policy",
    )
    validate_meshing_source_closure(evidence)
    corrupted = copy(evidence)
    object.__setattr__(corrupted, field, value)
    with pytest.raises(ValueError):
        validate_meshing_source_closure(corrupted)


def _coefficient_identity_space(
    name: str,
    count: int,
    components: tuple[int, ...] = (),
) -> DiscreteFieldSpace:
    from phydrax.discretization._spaces import DiscreteFieldSpace, TensorDofLayout
    from phydrax.linalg import ArraySpace

    return DiscreteFieldSpace(
        name,
        "coefficient-metadata-test-support",
        TensorDofLayout(("row",), (count,), component_shape=components),
        ArraySpace((count, *components)),
        representation="point_value",
    )


def _coefficient_identity_bank(
    space: DiscreteFieldSpace,
) -> MeshingCoefficientIdentityBank:
    from phydrax.lifecycle._meshing_field_records import (
        _coefficient_id_records,
        MeshingCoefficientIdentityBank,
    )

    return MeshingCoefficientIdentityBank(
        space.field_space_id,
        tuple(
            (value.ordinal, value.component, value.coefficient_id)
            for value in _coefficient_id_records(space)
        ),
    )


def test_coefficient_identity_banks_preserve_measured_counts_under_default_manifest_bound() -> (
    None
):
    from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
    from phydrax._model._structure import (
        _canonical_recipe_payload,
        model_from_structure_recipe,
        model_recipe_array_inventory,
        model_recipe_from_wire_json,
        model_recipe_wire_json,
        model_structure_recipe,
    )
    from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts

    register_meshing_source_artifacts()
    counts = (125, 729, 729, 512, 512, 1944, 1944, 1728, 1728)
    spaces = tuple(
        _coefficient_identity_space(f"field-{index}", count)
        for index, count in enumerate(counts)
    )
    spaces += (_coefficient_identity_space("euler", 64, (5,)),)
    banks = tuple(_coefficient_identity_bank(space) for space in spaces)
    recipe = model_structure_recipe(banks)
    assert (
        len(_canonical_recipe_payload(recipe))
        < DEFAULT_ARRAY_ARCHIVE_LIMITS.max_manifest_bytes
    )
    assert model_recipe_array_inventory(recipe, prefix="coefficient-identities") == ()
    wire = model_recipe_wire_json(recipe)
    restored = model_from_structure_recipe(model_recipe_from_wire_json(wire))
    assert restored == banks
    assert [len(bank.rows) for bank in restored] == [*counts, 320]
    for bank, space in zip(restored, spaces, strict=True):
        bank.require_field_space(space)
        assert not any(
            isinstance(value, (jax.Array, np.ndarray)) for value in jax.tree.leaves(bank)
        )


@pytest.mark.parametrize(
    "change", ("duplicate", "reordered", "missing", "foreign-id", "foreign-space")
)
def test_coefficient_identity_bank_refuses_incomplete_or_foreign_owner_rows(
    change: str,
) -> None:
    from phydrax.lifecycle._meshing_field_records import MeshingCoefficientIdentityBank

    space = _coefficient_identity_space("physical-field", 4, (2,))
    bank = _coefficient_identity_bank(space)
    rows, identity = bank.rows, bank.field_space_id
    if change == "duplicate":
        rows = (rows[0], *rows)
    elif change == "reordered":
        rows = tuple(reversed(rows))
    elif change == "missing":
        rows = rows[:-1]
    elif change == "foreign-id":
        rows = ((rows[0][0], rows[0][1], "foreign-scientific-coefficient"), *rows[1:])
    else:
        identity = "foreign-field-space"
    with pytest.raises(ValueError):
        MeshingCoefficientIdentityBank(identity, rows).require_field_space(space)


@pytest.mark.parametrize(
    "change", ("unknown-field", "noncanonical-json", "oversize", "deep-json")
)
def test_coefficient_identity_codec_refuses_noncanonical_or_unbounded_payload(
    change: str,
) -> None:
    from phydrax._array_archive import (
        ArrayArchiveCorruptionError,
        DEFAULT_ARRAY_ARCHIVE_LIMITS,
    )
    from phydrax.lifecycle._meshing_field_records import (
        _decode_coefficient_identity_bank,
        _encode_coefficient_identity_bank,
    )

    bank = _coefficient_identity_bank(_coefficient_identity_space("physical-field", 2))
    payload = dict(_encode_coefficient_identity_bank(bank))
    if change == "unknown-field":
        payload["coefficient_ids"] = "obsolete-carrier"
    elif change == "noncanonical-json":
        payload["rows_json"] = " " + payload["rows_json"]
    elif change == "oversize":
        payload["rows_json"] = " " * (DEFAULT_ARRAY_ARCHIVE_LIMITS.max_manifest_bytes + 1)
    else:
        payload["rows_json"] = "[" * 17 + "0" + "]" * 17
    with pytest.raises(
        ArrayArchiveCorruptionError if change == "deep-json" else ValueError
    ):
        _decode_coefficient_identity_bank(payload)
