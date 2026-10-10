"""Native lossless B-Rep persistence and explicit intersection-curve export policy.

The intersection reference is closed-form: a plane at height ``h`` cuts a unit
sphere in the circle of radius ``sqrt(1 - h^2)``.
"""

import json
import math
import zipfile
from collections.abc import Callable
from dataclasses import replace
from fractions import Fraction
from functools import cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._array_archive import ArrayArchiveLimits
from phydrax._external_resource import ResourceLimits
from phydrax.geometry._atlas import CurveTrimLoop, TrimDomain
from phydrax.geometry.brep._curve_approximation_bounds import (
    ApproximationDeviationError,
    certify_coupled_approximation,
)
from phydrax.geometry.brep._curve_approximation_topology import (
    BranchApproximationTopologyResourceError,
    certify_branch_approximation_topology,
    certify_branch_trim_separation,
)
from phydrax.geometry.brep._patches import BSplineCurve, LineCurve
from phydrax.geometry.brep._placement import materialize_brep_occurrences
from phydrax.interchange._cad import CadCoverage
from tests._support.cad_models import (
    native_assembly,
    native_empty_model,
    native_intersection_cap,
    native_rational_spline_face,
    native_repeated_assembly,
    native_rooted_sector_face,
    native_solid,
)


interchange = phx.interchange
brep = phx.geometry.brep
HEIGHT = 0.3
FIT = interchange.CadCurveFitPolicy(
    tolerance=1e-7, maximum_control_points=512, check_samples=1025
)


@pytest.mark.parametrize(
    "changes",
    [
        {"entity_counts": (("SURFACE", -1),)},
        {"entity_counts": (("SURFACE", True),)},
        {"entity_counts": (("SURFACE", 1), ("SURFACE", 2))},
        {"entity_counts": (("", 1),)},
        {"pcurves_honored": -1},
        {"pcurves_exact": True},
        {"pcurves_fitted": 0.5},
        {"maximum_fit_error": float("nan")},
        {"maximum_fit_error": float("inf")},
        {"maximum_fit_error": 1.0},
    ],
)
def test_cad_coverage_refuses_invalid_provider_evidence(
    changes: dict[str, object],
) -> None:
    values = dict(
        entity_counts=(("SURFACE", 1),),
        pcurves_honored=0,
        pcurves_exact=0,
        pcurves_fitted=0,
        maximum_fit_error=0.0,
        degenerate_edges=0,
        natural_boundaries=0,
    )
    with pytest.raises(ValueError):
        CadCoverage(**(values | changes))  # ty: ignore[invalid-argument-type]


def test_import_identity_binds_actual_coverage_evidence(tmp_path: Path) -> None:
    model = native_solid("box")
    destination = tmp_path / "coverage.step"
    interchange.write_step(model, destination)
    policy = interchange.CadImportPolicy(
        model.coordinate_contract,
        ResourceLimits(
            max_bytes=1 << 24,
            max_depth=64,
            max_nodes=200_000,
            max_attributes=4_000_000,
            max_losses=0,
        ),
        tessellation=brep.BRepTessellationPolicy(realize=False),
    )
    result = interchange.read_step(destination, policy, trusted_root=tmp_path)
    changed = replace(
        result,
        coverage=replace(
            result.coverage,
            natural_boundaries=result.coverage.natural_boundaries + 1,
        ),
    )
    assert changed.result_id != result.result_id
    assert changed.model.model_id == result.model.model_id
    assert changed.source_digest == result.source_digest


@pytest.mark.parametrize(
    "field", ("receipt", "report", "approximations", "failed-report")
)
def test_export_result_refuses_unbound_provider_output(
    field: str, tmp_path: Path
) -> None:
    result = interchange.write_step(native_empty_model(), tmp_path / "empty.step")
    value = None
    if field == "failed-report":
        field = "report"
        value = interchange.AdapterReport(
            interchange.AdapterStatus.MALFORMED_SOURCE,
            "phydrax-brep",
            "step",
            source_id=result.report.source_id,
            target_id=result.report.target_id,
        )
    with pytest.raises((TypeError, ValueError)):
        replace(result, **{field: value})


@cache
def _branch() -> brep.IntersectionCurve:
    sphere = brep.SurfaceRegion(
        brep.SpherePatch([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], 1.0),
        np.asarray([[0.0, -math.pi / 2], [2 * math.pi, math.pi / 2]]),
    )
    plane = brep.SurfaceRegion(
        brep.PlanePatch([-2, -2, HEIGHT], [4, 0, 0], [0, 4, 0]),
        np.asarray([[0.0, 0.0], [1.0, 1.0]]),
    )
    result = brep.intersect_surface_regions(sphere, plane)
    assert result.complete and len(result.curves) == 1
    return result.curves[0]


def _with_intersection_trim(
    model: brep.BRepModel, *, split_reversed: bool = False
) -> brep.BRepModel:
    """The model with face 0 trimmed by an exact intersection-curve loop."""
    branch = _branch()
    first, last = branch.parameter_interval
    middle = 0.5 * (first + last)
    curves = (
        [
            brep.IntersectionPCurve(
                branch, "second", first=middle, last=last, reversed=True
            ),
            brep.IntersectionPCurve(
                branch, "second", first=first, last=middle, reversed=True
            ),
        ]
        if split_reversed
        else [branch.p_curve("second")]
    )
    loop = CurveTrimLoop(curves, tolerance=0.2)
    domains = (TrimDomain(loop), *model.trim_domains[1:])
    return brep.BRepModel(
        patches=model.patches,
        parameter_bounds=model.parameter_bounds,
        orientation=model.orientation,
        trim_domains=domains,
        topology=model.topology,
        coordinate_contract=model.coordinate_contract,
        mesh_vertices=model.mesh_vertices,
        mesh_faces=model.mesh_faces,
        triangle_face_ids=model.triangle_face_ids,
        triangle_parameters=model.triangle_parameters,
        tessellation_deviation_bounds=model.tessellation_deviation_bounds,
        tessellation_normal_bounds=model.tessellation_normal_bounds,
        mesh_vertex_source_dimensions=model.mesh_vertex_source_dimensions,
        mesh_vertex_source_indices=model.mesh_vertex_source_indices,
        mesh_vertex_parameters=model.mesh_vertex_parameters,
        mesh_chart_restriction_vertices=model.mesh_chart_restriction_vertices,
        mesh_chart_restriction_edges=model.mesh_chart_restriction_edges,
        mesh_chart_restriction_endpoint_parameters=(
            model.mesh_chart_restriction_endpoint_parameters
        ),
        mesh_chart_restriction_parameters=model.mesh_chart_restriction_parameters,
        coedge_deviation_bounds=model.coedge_deviation_bounds,
        triangle_occurrence_ids=model.triangle_occurrence_ids,
        vertex_occurrence_ids=model.vertex_occurrence_ids,
        physical_tags=model.physical_tags,
        report=model.report,
        geometry=model.geometry,
    )


@pytest.mark.parametrize("name", ["box", "sphere", "torus", "plate", "assembly"])
def test_archive_round_trip_preserves_complete_model_identity(
    name: str, tmp_path: Path
) -> None:
    model = native_assembly() if name == "assembly" else native_solid(name)
    receipt = interchange.save_brep_archive(model, tmp_path / "model.phx")
    restored = interchange.load_brep_archive(receipt.path)
    assert restored.model_id == model.model_id == receipt.model_id
    assert restored.tessellation_id == model.tessellation_id
    assert restored.chart_restriction_ids == model.chart_restriction_ids
    assert restored.geometry is not None and model.geometry is not None
    assert restored.geometry.geometry_id == model.geometry.geometry_id
    assert restored.geometry.occurrences == model.geometry.occurrences
    assert restored.geometry.assembly_containers == model.geometry.assembly_containers
    assert restored.coordinate_contract.spatial_id == model.coordinate_contract.spatial_id
    assert restored.source_revision == model.source_revision
    np.testing.assert_array_equal(
        restored.tessellation_deviation_bounds, model.tessellation_deviation_bounds
    )
    np.testing.assert_array_equal(
        restored.tessellation_normal_bounds, model.tessellation_normal_bounds
    )
    np.testing.assert_array_equal(
        restored.mesh_vertex_source_dimensions, model.mesh_vertex_source_dimensions
    )
    np.testing.assert_array_equal(
        restored.mesh_vertex_source_indices, model.mesh_vertex_source_indices
    )
    np.testing.assert_array_equal(
        restored.mesh_vertex_parameters, model.mesh_vertex_parameters
    )
    np.testing.assert_array_equal(
        restored.mesh_chart_restriction_vertices,
        model.mesh_chart_restriction_vertices,
    )
    np.testing.assert_array_equal(
        restored.mesh_chart_restriction_edges,
        model.mesh_chart_restriction_edges,
    )
    np.testing.assert_array_equal(
        restored.mesh_chart_restriction_endpoint_parameters,
        model.mesh_chart_restriction_endpoint_parameters,
    )
    np.testing.assert_array_equal(
        restored.mesh_chart_restriction_parameters,
        model.mesh_chart_restriction_parameters,
    )
    np.testing.assert_array_equal(
        restored.coedge_deviation_bounds, model.coedge_deviation_bounds
    )
    np.testing.assert_array_equal(
        restored.triangle_occurrence_ids, model.triangle_occurrence_ids
    )
    np.testing.assert_array_equal(
        restored.vertex_occurrence_ids, model.vertex_occurrence_ids
    )
    assert restored.report.curve_surface_tolerance == model.report.curve_surface_tolerance
    assert restored.report.curve_surface_scale == model.report.curve_surface_scale


def test_archive_preserves_materialized_world_source_and_pose(tmp_path: Path) -> None:
    world = materialize_brep_occurrences(
        native_repeated_assembly(),
        tessellation=brep.BRepTessellationPolicy(realize=False),
    ).model
    restored = interchange.load_brep_archive(
        interchange.save_brep_archive(world, tmp_path / "world.phx").path
    )
    assert restored.model_id == world.model_id
    assert restored.geometry is not None and world.geometry is not None
    assert restored.geometry.geometry_id == world.geometry.geometry_id
    for face, (before, after) in enumerate(
        zip(world.patches, restored.patches, strict=True)
    ):
        bounds = np.asarray(world.parameter_bounds[face])
        parameters = np.asarray(
            (0.25 * bounds[0] + 0.75 * bounds[1], 0.75 * bounds[0] + 0.25 * bounds[1])
        )
        np.testing.assert_array_equal(
            after.evaluate(parameters),
            before.evaluate(parameters),
            err_msg=f"face {face}",
        )


@pytest.mark.parametrize(
    "distance", [0.125, 0.1], ids=["binary-radius", "exact-nonbinary-radius"]
)
def test_archived_placed_offset_sphere_retains_native_containment(
    distance: float,
    tmp_path: Path,
) -> None:
    contract = phx.SpatialCoordinateContract.si()
    tessellation = brep.BRepTessellationPolicy(realize=False)
    policy = interchange.CadImportPolicy(
        contract,
        ResourceLimits(4194304, 64, 20000, 2000000, 16),
        tessellation=tessellation,
    )
    text = (
        "CASCADE Topology V3, (c) Open Cascade\nLocations 0\nCurve2ds 0\n"
        "Curves 0\nPolygon3D 0\nPolygonOnTriangulations 0\nSurfaces 1\n"
        f"11 {distance!r} 4 0 0 0 0 0 1 1 0 0 0 1 0 1\n"
        "Triangulations 0\nTShapes 3\nFa\n0 1e-8 1 0\n0101000\n*\n"
        "Sh\n0101000\n+3 0 *\nSo\n0101000\n+2 0 *\n+1 0\n"
    )
    source = interchange.decode_brep_text_bytes(
        text.encode("ascii"),
        policy,
        source_length_unit=contract.length_unit,
    ).model
    if source.geometry is None:
        raise ValueError("The authored sphere lost its complete native geometry.")
    geometry = source.geometry
    rotation = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64
    )
    translation = np.asarray((4.0, 2.0, 1.0), dtype=np.float64)
    occurrence = brep.BRepOccurrence(("offset-sphere",), 0, rotation, translation)
    placed_geometry = brep.BRepGeometry(
        vertex_points=geometry.vertex_points,
        curves=geometry.curves,
        edge_curves=geometry.edge_curves,
        edge_ranges=geometry.edge_ranges,
        edge_vertices=geometry.edge_vertices,
        pcurves=geometry.pcurves,
        coedge_edges=geometry.coedge_edges,
        coedge_senses=geometry.coedge_senses,
        face_loops=geometry.face_loops,
        shell_faces=geometry.shell_faces,
        shell_orientations=geometry.shell_orientations,
        solid_shells=geometry.solid_shells,
        occurrences=(occurrence,),
        assembly_containers=(),
        vertex_roots=geometry.vertex_roots,
        edge_endpoint_roots=geometry.edge_endpoint_roots,
        coedge_endpoint_roots=geometry.coedge_endpoint_roots,
    )
    placed = brep.assemble_brep_model(
        placed_geometry,
        source.patches,
        source.parameter_bounds,
        source.orientation,
        source.physical_tags,
        coordinate_contract=contract,
        source_id=source.source_id,
        source_format=source.report.source_format,
        source_digest=source.source_digest,
        import_policy_id=source.import_policy_id,
        tessellation=tessellation,
    )
    world = materialize_brep_occurrences(placed, tessellation=tessellation).model
    restored = interchange.load_brep_archive(
        interchange.save_brep_archive(world, tmp_path / "placed-offset.phx").path,
    )
    query = brep.prepare_brep_query(restored)
    points = jnp.asarray(
        translation + np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)), dtype=np.float64),
        dtype=jnp.float64,
    )
    result = query.contains(points)
    np.testing.assert_array_equal(result.inside, (True, False))
    np.testing.assert_array_equal(result.unresolved, (False, False))
    expected = (Fraction(1) + Fraction(distance), Fraction(1) - Fraction(distance))
    lower, upper = (
        np.asarray(result.distance_lower_bounds),
        np.asarray(result.distance_upper_bounds),
    )
    for row, exact in enumerate(expected):
        assert Fraction(float(lower[row])) <= exact <= Fraction(float(upper[row])), (
            f"point {row}"
        )
    assert np.all(upper - lower < 1.0e-12)
    assert restored.source_revision == world.source_revision
    assert restored.geometry is not None and world.geometry is not None
    assert restored.geometry.geometry_id == world.geometry.geometry_id


def test_archive_empty_exact_model_preserves_zero_shapes_and_lineage(
    tmp_path: Path,
) -> None:
    native = native_empty_model()
    receipt = interchange.save_brep_archive(native, tmp_path / "empty.phx")
    restored = interchange.load_brep_archive(receipt.path)
    assert restored.model_id == native.model_id
    assert restored.source_revision == native.source_revision
    assert restored.tessellation_id == native.tessellation_id
    assert restored.geometry is not None and native.geometry is not None
    assert restored.geometry.geometry_id == native.geometry.geometry_id
    assert restored.geometry.occurrences == ()
    assert restored.geometry.vertex_points.shape == (0, 3)
    assert restored.geometry.edge_ranges.shape == (0, 2)
    assert restored.patches == ()
    assert restored.parameter_bounds.shape == (0, 2, 2)
    assert restored.mesh_vertices.shape == (0, 3)
    assert restored.mesh_faces.shape == (0, 3)
    assert restored.tessellation_deviation_bounds.shape == (0,)
    assert restored.tessellation_normal_bounds.shape == (0,)
    assert restored.mesh_vertex_source_dimensions.shape == (0,)
    assert restored.mesh_vertex_source_indices.shape == (0,)
    assert restored.mesh_vertex_parameters.shape == (0, 2)
    assert restored.coedge_deviation_bounds.shape == (0,)
    assert restored.triangle_occurrence_ids.shape == (0,)
    assert restored.vertex_occurrence_ids.shape == (0,)


def test_archive_is_deterministic(tmp_path: Path) -> None:
    model = native_solid("cylinder")
    first = interchange.save_brep_archive(model, tmp_path / "a.phx")
    second = interchange.save_brep_archive(model, tmp_path / "b.phx")
    assert first.path.read_bytes() == second.path.read_bytes()


def test_archive_publication_is_exclusive_unless_replacement_is_requested(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "model.phx"
    original = native_solid("box")
    replacement = native_solid("sphere")
    interchange.save_brep_archive(original, destination)
    unchanged = destination.read_bytes()
    with pytest.raises(FileExistsError):
        interchange.save_brep_archive(replacement, destination)
    assert destination.read_bytes() == unchanged
    interchange.save_brep_archive(replacement, destination, mode="atomic_replace")
    assert interchange.load_brep_archive(destination).model_id == replacement.model_id


def test_archive_preserves_intersection_curve_trims(tmp_path: Path) -> None:
    model = _with_intersection_trim(native_solid("box"))
    restored = interchange.load_brep_archive(
        interchange.save_brep_archive(model, tmp_path / "trim.phx").path
    )
    assert restored.model_id == model.model_id
    assert restored.tessellation_id == model.tessellation_id
    domain = restored.trim_domains[0]
    assert domain is not None and isinstance(domain.outer, CurveTrimLoop)
    curve = domain.outer.curves[0]
    assert isinstance(curve, brep.IntersectionPCurve)
    assert curve.curve.branch_id == _branch().branch_id
    assert curve.first == _branch().parameter_interval[0]
    assert curve.last == _branch().parameter_interval[1]
    assert curve.reversed is False
    parameters = jnp.asarray([0.0, 0.25, 0.75, 1.0]) * curve.parameter_interval[1]
    np.testing.assert_array_equal(
        curve.evaluate(parameters), _branch().p_curve("second").evaluate(parameters)
    )


def test_archive_preserves_rational_carrier_closure(tmp_path: Path) -> None:
    model = native_rational_spline_face()
    receipt = interchange.save_brep_archive(model, tmp_path / "spline.phx")
    restored = interchange.load_brep_archive(receipt.path)
    assert restored.geometry is not None and model.geometry is not None
    assert restored.geometry.geometry_id == model.geometry.geometry_id
    assert restored.model_id == model.model_id
    for field in ("control_points", "weights", "u_knots", "v_knots"):
        np.testing.assert_array_equal(
            getattr(restored.patches[0], field), getattr(model.patches[0], field)
        )


def test_archive_preserves_certified_vertex_and_endpoint_source_definitions(
    tmp_path: Path,
) -> None:
    model = native_rooted_sector_face()
    restored = interchange.load_brep_archive(
        interchange.save_brep_archive(model, tmp_path / "rooted.phx").path
    )
    assert restored.geometry is not None and model.geometry is not None
    assert restored.geometry.geometry_id == model.geometry.geometry_id
    vertex = restored.geometry.vertex_roots[0]
    original = model.geometry.vertex_roots[0]
    assert vertex is not None and original is not None
    assert vertex.root_id == original.root_id
    point, bound, certified = vertex.evaluate()
    assert certified and bound < 1e-6
    np.testing.assert_allclose(point, [1.0, 0.0, 0.0], rtol=0.0, atol=1e-12)
    for edge, endpoint in ((0, 0), (2, 1)):
        expression = restored.geometry.edge_endpoint_roots[edge][endpoint]
        assert expression is not None and vertex.supports_endpoint(expression)
        lower, upper = expression.parameter_enclosure()
        assert lower <= restored.geometry.edge_ranges[edge, endpoint] <= upper
    area = float(brep.prepare_brep_query(restored).measures.face_areas[0])
    assert area == pytest.approx(0.75 * np.pi, rel=1e-9)


def test_archive_preserves_reversed_intersection_trim_slices(tmp_path: Path) -> None:
    model = _with_intersection_trim(native_solid("box"), split_reversed=True)
    restored = interchange.load_brep_archive(
        interchange.save_brep_archive(model, tmp_path / "slices.phx").path
    )
    domain = restored.trim_domains[0]
    if domain is None or not isinstance(domain.outer, CurveTrimLoop):
        raise TypeError(
            "Restored reversed intersection slices require a curve trim loop."
        )
    first, last = _branch().parameter_interval
    middle = 0.5 * (first + last)
    slices: list[brep.IntersectionPCurve] = []
    for curve in domain.outer.curves:
        if not isinstance(curve, brep.IntersectionPCurve):
            raise TypeError(
                "Restored reversed slices must retain exact intersection p-curves."
            )
        slices.append(curve)
    assert tuple((curve.first, curve.last, curve.reversed) for curve in slices) == (
        (middle, last, True),
        (first, middle, True),
    )
    for curve in slices:
        parameters = curve.first + jnp.asarray([0.0, 0.3, 0.8, 1.0]) * (
            curve.last - curve.first
        )
        expected = (
            _branch().evaluate(curve.first + curve.last - parameters).second_parameters
        )
        np.testing.assert_array_equal(curve.evaluate(parameters), expected)


def _mutated_archive(
    source: Path, destination: Path, mutate: Callable[[dict], None]
) -> Path:
    with (
        zipfile.ZipFile(source) as incoming,
        zipfile.ZipFile(destination, "w") as outgoing,
    ):
        for item in incoming.infolist():
            payload = incoming.read(item)
            if item.filename == "manifest.json":
                manifest = json.loads(payload)
                mutate(manifest)
                payload = json.dumps(manifest, sort_keys=True).encode()
            outgoing.writestr(item, payload)
    return destination


def _definition(manifest: dict, value: dict) -> dict:
    while set(value) == {"definition"}:
        value = manifest["definitions"][value["definition"]]
    return value


@pytest.mark.parametrize(
    "location", ["report", "geometry", "topology", "patch", "occurrence", "contract"]
)
def test_archive_refuses_unknown_nested_fields(location: str, tmp_path: Path) -> None:
    source = interchange.save_brep_archive(
        native_solid("box"), tmp_path / "source.phx"
    ).path

    def mutate(manifest: dict) -> None:
        if location == "patch":
            record = manifest["patches"][0]
        elif location == "occurrence":
            record = manifest["geometry"]["occurrences"][0]
        elif location == "contract":
            record = manifest["report"]["coordinate_contract"]
        else:
            record = manifest[location]
        record["version"] = 1

    tampered = _mutated_archive(source, tmp_path / "unknown.phx", mutate)
    with pytest.raises(ValueError):
        interchange.load_brep_archive(tampered)


def test_archive_refuses_altered_intersection_proof(tmp_path: Path) -> None:
    source = interchange.save_brep_archive(
        _with_intersection_trim(native_solid("box")), tmp_path / "source.phx"
    ).path

    def mutate(manifest: dict) -> None:
        loop = _definition(manifest, manifest["trim_domains"][0]["outer"])
        trim = _definition(manifest, loop["curves"][0])
        branch = _definition(manifest, trim["curve"])
        branch["certified"][0] = False

    tampered = _mutated_archive(source, tmp_path / "proof.phx", mutate)
    with pytest.raises(ValueError):
        interchange.load_brep_archive(tampered)


def test_archive_refuses_coerced_trim_orientation(tmp_path: Path) -> None:
    source = interchange.save_brep_archive(
        _with_intersection_trim(native_solid("box")), tmp_path / "source.phx"
    ).path

    def mutate(manifest: dict) -> None:
        loop = _definition(manifest, manifest["trim_domains"][0]["outer"])
        trim = _definition(manifest, loop["curves"][0])
        trim["reversed"] = "false"

    tampered = _mutated_archive(source, tmp_path / "orientation.phx", mutate)
    with pytest.raises(ValueError):
        interchange.load_brep_archive(tampered)


def test_branch_archive_deduplicates_exact_support_under_bounded_manifest(
    tmp_path: Path,
) -> None:
    model = native_intersection_cap(realize=False)
    limits = ArrayArchiveLimits(max_manifest_bytes=65_536)
    receipt = interchange.save_brep_archive(
        model, tmp_path / "deduplicated.phx", limits=limits
    )
    with zipfile.ZipFile(receipt.path) as container:
        assert container.getinfo("manifest.json").file_size <= limits.max_manifest_bytes
        manifest = json.loads(container.read("manifest.json"))
    branches = [
        definition
        for definition in manifest["definitions"].values()
        if definition.get("kind") == "intersection-curve"
    ]
    assert len(branches) == 1
    restored = interchange.load_brep_archive(receipt.path, limits=limits)
    assert restored.geometry is not None and model.geometry is not None
    assert restored.geometry.geometry_id == model.geometry.geometry_id
    assert restored.model_id == model.model_id
    assert restored.tessellation_id == model.tessellation_id
    branch = restored.geometry.curves[0]
    pcurve = restored.geometry.pcurves[0]
    if not isinstance(branch, brep.IntersectionCurve) or not isinstance(
        pcurve, brep.IntersectionPCurve
    ):
        raise TypeError("The archive must restore exact branch and p-curve carriers.")
    assert branch is pcurve.curve
    parameters = jnp.linspace(*branch.parameter_interval, 33)
    points = np.asarray(branch.evaluate(parameters).point)
    np.testing.assert_allclose(points[:, 2], HEIGHT, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.norm(points[:, :2], axis=1), np.sqrt(1.0 - HEIGHT**2), atol=1e-12
    )


@pytest.mark.parametrize("failure", ["cycle", "dangling", "orphan"])
def test_archive_definition_dag_refuses_invalid_closure(
    failure: str, tmp_path: Path
) -> None:
    source = interchange.save_brep_archive(
        native_solid("box"), tmp_path / "source.phx"
    ).path

    def mutate(manifest: dict) -> None:
        identifier = "0" * 64
        if failure == "cycle":
            manifest["definitions"][identifier] = {
                "kind": "cycle",
                "child": {"definition": identifier},
            }
            manifest["patches"][0] = {"definition": identifier}
        elif failure == "dangling":
            manifest["patches"][0] = {"definition": identifier}
        else:
            from phydrax.interchange._cad_archive import _definition_id

            payload = {"kind": "unused-source", "value": "unreachable"}
            manifest["definitions"][_definition_id(payload)] = payload
        manifest["definitions"] = dict(sorted(manifest["definitions"].items()))

    tampered = _mutated_archive(source, tmp_path / "bad-dag.phx", mutate)
    with pytest.raises(ValueError):
        interchange.load_brep_archive(tampered)


def test_archive_keeps_exact_identity_separate_from_tessellation(tmp_path: Path) -> None:
    model = native_solid("sphere")
    assert model.geometry is not None
    retessellated = brep.assemble_brep_model(
        model.geometry,
        model.patches,
        model.parameter_bounds,
        model.orientation,
        model.physical_tags,
        coordinate_contract=model.coordinate_contract,
        source_id=model.source_id,
        source_format=model.report.source_format,
        source_digest=model.source_digest,
        import_policy_id=model.import_policy_id,
        tessellation=brep.BRepTessellationPolicy(
            linear_deflection=0.02, angular_deflection=0.2, trim_samples_per_edge=17
        ),
    )
    first = interchange.save_brep_archive(model, tmp_path / "first.phx")
    second = interchange.save_brep_archive(retessellated, tmp_path / "second.phx")
    assert first.model_id == second.model_id
    assert first.geometry_id == second.geometry_id
    assert first.tessellation_id != second.tessellation_id
    for receipt in (first, second):
        restored = interchange.load_brep_archive(receipt.path)
        assert restored.model_id == receipt.model_id
        assert restored.tessellation_id == receipt.tessellation_id


def test_tampered_archive_is_refused(tmp_path: Path) -> None:
    path = interchange.save_brep_archive(native_solid("box"), tmp_path / "box.phx").path
    tampered = tmp_path / "tampered.phx"
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(tampered, "w") as target:
        for item in source.infolist():
            payload = source.read(item)
            if item.filename == "manifest.json":
                manifest = json.loads(payload)
                manifest["physical_tags"][0] = "cylinder"
                payload = json.dumps(manifest, indent=2, sort_keys=True).encode()
            target.writestr(item, payload)
    with pytest.raises(ValueError):
        interchange.load_brep_archive(tampered)


def test_archive_decode_is_bounded(tmp_path: Path) -> None:
    path = interchange.save_brep_archive(
        native_solid("sphere"), tmp_path / "sphere.phx"
    ).path
    with pytest.raises(Exception, match="limit"):
        interchange.load_brep_archive(
            path, limits=ArrayArchiveLimits(max_manifest_bytes=1024)
        )


def test_intersection_curve_approximation_certifies_continuous_coupled_bounds() -> None:
    policy = FIT
    approximation = interchange.approximate_intersection_curve(_branch(), policy)
    continuous = approximation.continuous_evidence
    assert continuous.distance_bound <= policy.tolerance * approximation.distance_scale
    assert continuous.parameter_bound <= policy.tolerance * approximation.parameter_scale
    assert (
        continuous.first_correspondence_bound
        <= policy.tolerance * approximation.distance_scale
    )
    assert (
        continuous.second_correspondence_bound
        <= policy.tolerance * approximation.distance_scale
    )
    assert approximation.certification_cells <= policy.maximum_certificate_cells
    assert approximation.certification_peak_bytes <= policy.maximum_certificate_bytes
    assert approximation.topology_evidence.trim_closure_kinds == (
        "open-native-period-lift",
        "closed",
    )
    parameters = jnp.linspace(0.0, float(_branch().num_charts), 257)
    points = np.asarray(approximation.curve.evaluate(parameters))
    np.testing.assert_allclose(
        np.linalg.norm(points[:, :2], axis=1), math.sqrt(1 - HEIGHT**2), atol=2e-7
    )
    np.testing.assert_allclose(points[:, 2], HEIGHT, atol=2e-7)
    plane = _branch().second.patch
    on_plane = np.asarray(
        plane.evaluate(approximation.second_pcurve.evaluate(parameters))
    )
    np.testing.assert_allclose(on_plane, points, atol=5e-7)


def _between_check_loop(curve: BSplineCurve) -> BSplineCurve:
    """A genuine cubic excursion between two original policy check points."""
    first, middle, last = 0.01, 0.0125, 0.015

    def split(points: np.ndarray, parameter: float) -> tuple[np.ndarray, np.ndarray]:
        rows = [points]
        while rows[-1].shape[0] > 1:
            rows.append((1.0 - parameter) * rows[-1][:-1] + parameter * rows[-1][1:])
        return np.asarray([row[0] for row in rows]), np.asarray(
            [row[-1] for row in reversed(rows)]
        )

    def restrict(points: np.ndarray, lower: float, upper: float) -> np.ndarray:
        left = split(points, upper)[0]
        return left if lower == 0.0 else split(left, lower / upper)[1]

    segments: list[tuple[float, float, np.ndarray]] = []
    for piece in curve.bezier_pieces():
        lower, upper = piece.parameter_bounds[0]
        homogeneous = np.asarray(piece.homogeneous_controls)
        points = homogeneous[:, :-1] / homogeneous[:, -1:]
        if lower < first < last < upper:
            prefix = restrict(points, 0.0, (first - lower) / (upper - lower))
            suffix = restrict(points, (last - lower) / (upper - lower), 1.0)
            start, end = prefix[-1], suffix[0]
            excursion = np.asarray(
                (start, start + (0.1, 0.0, 0.0), start + (0.0, 0.1, 0.0), start)
            )
            bridge = np.asarray(
                (start, start + (end - start) / 3, start + 2 * (end - start) / 3, end)
            )
            segments.extend(
                (
                    (lower, first, prefix),
                    (first, middle, excursion),
                    (middle, last, bridge),
                    (last, upper, suffix),
                )
            )
        else:
            segments.append((lower, upper, points))
    controls = np.concatenate(
        (segments[0][2], *(points[1:] for _, _, points in segments[1:]))
    )
    knots = np.concatenate(
        (
            np.full(4, segments[0][0]),
            *(np.full(3, upper) for _, upper, _ in segments[:-1]),
            np.full(4, segments[-1][1]),
        )
    )
    return BSplineCurve(controls, np.ones(controls.shape[0]), knots, 3)


def test_between_check_excursion_cannot_obtain_continuous_or_topology_certificate() -> (
    None
):
    source = _branch()
    approximation = interchange.approximate_intersection_curve(source, FIT)
    adversary = _between_check_loop(approximation.curve)
    parameters = np.linspace(*source.parameter_interval, FIT.check_samples)
    assert not np.any((parameters > 0.01) & (parameters < 0.015))
    assert np.asarray(adversary.control_points).shape[0] <= FIT.maximum_control_points
    np.testing.assert_allclose(
        np.asarray(adversary.evaluate(jnp.asarray(parameters))),
        np.asarray(approximation.curve.evaluate(jnp.asarray(parameters))),
        rtol=0.0,
        atol=1.0e-12,
    )
    with pytest.raises(ApproximationDeviationError) as deviation:
        certify_coupled_approximation(
            source,
            adversary,
            approximation.first_pcurve,
            approximation.second_pcurve,
            distance_tolerance=FIT.tolerance * approximation.distance_scale,
            parameter_tolerance=FIT.tolerance * approximation.parameter_scale,
            maximum_cells=FIT.maximum_certificate_cells,
            maximum_depth=FIT.maximum_certificate_depth,
            maximum_bytes=FIT.maximum_certificate_bytes,
        )
    assert deviation.value.lower_bound > deviation.value.tolerance
    with pytest.raises(BranchApproximationTopologyResourceError):
        certify_branch_approximation_topology(
            source,
            adversary,
            approximation.first_pcurve,
            approximation.second_pcurve,
            maximum_cells=FIT.maximum_certificate_cells,
            maximum_depth=FIT.maximum_certificate_depth,
            maximum_bytes=FIT.maximum_certificate_bytes,
        )


def test_trim_contact_cannot_obtain_separation_certificate() -> None:
    source = _branch()
    approximation = interchange.approximate_intersection_curve(source, FIT)
    center = np.asarray((0.5, 0.5))
    clear = LineCurve(center, (0.01, 0.0))
    evidence = certify_branch_trim_separation(
        approximation.topology_evidence,
        clear,
        side="second",
        first=0.0,
        last=1.0,
        maximum_cells=FIT.maximum_certificate_cells,
        maximum_depth=FIT.maximum_certificate_depth,
        maximum_bytes=FIT.maximum_certificate_bytes,
    )
    assert evidence.separation_lower_bound > 0.0
    endpoint = np.asarray(approximation.second_pcurve.control_points)[0]
    contact = LineCurve(endpoint, center - endpoint)
    with pytest.raises(BranchApproximationTopologyResourceError):
        certify_branch_trim_separation(
            approximation.topology_evidence,
            contact,
            side="second",
            first=0.0,
            last=1.0,
            maximum_cells=FIT.maximum_certificate_cells,
            maximum_depth=FIT.maximum_certificate_depth,
            maximum_bytes=FIT.maximum_certificate_bytes,
        )


def test_archive_preserves_native_branch_edge_and_pcurve_incidence(
    tmp_path: Path,
) -> None:
    model = native_intersection_cap(realize=False)
    receipt = interchange.save_brep_archive(model, tmp_path / "branch.phx")
    restored = interchange.load_brep_archive(receipt.path)
    assert restored.geometry is not None and model.geometry is not None
    assert restored.geometry.geometry_id == model.geometry.geometry_id
    assert restored.tessellation_id == model.tessellation_id
    branch = restored.geometry.curves[0]
    pcurve = restored.geometry.pcurves[0]
    assert isinstance(branch, brep.IntersectionCurve)
    assert isinstance(pcurve, brep.IntersectionPCurve)
    assert pcurve.curve.branch_id == branch.branch_id
    parameters = jnp.linspace(*branch.parameter_interval, 33)
    points = np.asarray(branch.evaluate(parameters).point)
    on_plane = np.asarray(restored.patches[0].evaluate(pcurve.evaluate(parameters)))
    np.testing.assert_allclose(on_plane, points, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(points[:, 2], HEIGHT, rtol=0.0, atol=1e-12)
    assert np.all(np.isfinite(restored.coedge_deviation_bounds))
    assert (
        np.max(restored.coedge_deviation_bounds)
        <= restored.report.curve_surface_tolerance
    )


def test_step_substitutes_real_branch_topology_under_explicit_policy(
    tmp_path: Path,
) -> None:
    model = native_intersection_cap()
    assert model.geometry is not None
    branch = model.geometry.curves[0]
    assert isinstance(branch, brep.IntersectionCurve)
    with pytest.raises(interchange.CadInterchangeError) as exact:
        interchange.write_step(model, tmp_path / "exact-branch.step")
    assert exact.value.refusal.reason == "inexact-export"
    assert "edge:0" in exact.value.refusal.chain
    policy = interchange.CadExportPolicy(intersection_approximation=FIT)
    result = interchange.write_step(
        model, tmp_path / "bounded-branch.step", policy=policy
    )
    assert result.report.status == interchange.AdapterStatus.DECLARED_LOSS
    assert tuple(item.branch_id for item in result.approximations) == (branch.branch_id,)
    restored = interchange.decode_step_bytes(
        Path(result.receipt.destination).read_bytes(),
        interchange.CadImportPolicy(
            model.coordinate_contract,
            ResourceLimits(
                max_bytes=1 << 24,
                max_depth=64,
                max_nodes=200_000,
                max_attributes=4_000_000,
                max_losses=0,
            ),
        ),
    ).model
    assert restored.geometry is not None
    assert isinstance(restored.geometry.curves[0], brep.BSplineCurve)
    assert isinstance(restored.geometry.pcurves[0], brep.BSplineCurve)
    assert restored.geometry.edge_vertices == ((0, 0),)
    assert restored.geometry.coedge_edges == (0,)
    assert restored.geometry.coedge_senses == model.geometry.coedge_senses
    np.testing.assert_array_equal(
        restored.geometry.edge_ranges, model.geometry.edge_ranges
    )
    parameters = jnp.linspace(*branch.parameter_interval, 129)
    written = np.asarray(restored.geometry.curves[0].evaluate(parameters))
    expected = np.asarray(branch.evaluate(parameters).point)
    np.testing.assert_allclose(written, expected, rtol=0.0, atol=2e-7)
    area = float(brep.prepare_brep_query(restored).measures.face_areas[0])
    assert area == pytest.approx(np.pi * (1.0 - HEIGHT**2), rel=2e-6)
    # The source remains an exact branch model, not a forged fitted native ID.
    assert model.geometry.curves[0] is branch


def test_trim_only_intersection_without_edge_incidence_is_not_exported(
    tmp_path: Path,
) -> None:
    model = _with_intersection_trim(native_solid("box"))
    with pytest.raises(interchange.CadInterchangeError) as caught:
        interchange.write_step(model, tmp_path / "exact.step")
    assert caught.value.refusal.reason == "inexact-export"
    assert "face:0" in caught.value.refusal.chain
    policy = interchange.CadExportPolicy(intersection_approximation=FIT)
    with pytest.raises(interchange.CadInterchangeError) as sampled:
        interchange.write_step(model, tmp_path / "approximate.step", policy=policy)
    assert sampled.value.refusal.reason == "inexact-export"
    assert not (tmp_path / "approximate.step").exists()


def test_continuous_certificate_budget_refuses_before_external_publication(
    tmp_path: Path,
) -> None:
    model = native_intersection_cap()
    bounded = interchange.CadCurveFitPolicy(
        tolerance=FIT.tolerance,
        maximum_control_points=FIT.maximum_control_points,
        check_samples=FIT.check_samples,
        maximum_certificate_cells=1,
    )
    path = tmp_path / "exhausted-proof.step"
    with pytest.raises(interchange.CadInterchangeError) as refusal:
        interchange.write_step(
            model,
            path,
            policy=interchange.CadExportPolicy(intersection_approximation=bounded),
        )
    assert refusal.value.refusal.reason == "limit"
    assert "edge:0" in refusal.value.refusal.chain
    assert not path.exists()
