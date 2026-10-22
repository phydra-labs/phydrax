"""Exercise IGES carriers, bounded decoding and independent fixtures."""

from __future__ import annotations

import builtins
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax import SpatialCoordinateContract
from phydrax.geometry import PolygonTrimLoop, TrimDomain
from phydrax.geometry.brep import (
    brep_box,
    BRepModel,
    BSplineCurve,
    BSplineSurfacePatch,
    IntersectionCurve,
    OffsetSurface,
    PlanePatch,
    prepare_brep_query,
)
from phydrax.interchange import (
    _iges,
    AdapterStatus,
    CadCurveFitPolicy,
    CadExportPolicy,
    CadExportResult,
    CadImportPolicy,
    CadInterchangeError,
    decode_iges_bytes,
    load_brep_archive,
    read_iges,
    ResourceLimits,
    save_brep_archive,
    write_iges,
)
from phydrax.interchange._cad import decode_bounded_approximation_descriptor
from phydrax.units import MILLIMETER
from tests._support.cad_models import (
    CONTRACT,
    EXACT_VOLUMES,
    geometry_of,
    native_assembly,
    native_empty_model,
    native_intersection_cap,
    native_rational_spline_face,
    native_repeated_assembly,
    native_solid,
    solid_volumes,
    SOLIDS,
)


LIMITS = ResourceLimits(
    max_bytes=1 << 22,
    max_depth=32,
    max_nodes=20_000,
    max_attributes=2_000_000,
    max_losses=16,
)
POLICY = CadImportPolicy(CONTRACT, LIMITS)

# Preserve the pre-refactor IGES qualification envelopes: these are empirical
# acceptance thresholds, not certified bounds. Source: this module's native
# solid/assembly and independent OCCT measure comparisons.
_VOLUME_RELATIVE_TOLERANCE = 2.0e-6
_OCP_MEASURE_RELATIVE_TOLERANCE = 1.0e-7
# Source: the independent IGES 5.3 section 4.30 planar-offset fixture below.
_OFFSET_AREA_RELATIVE_TOLERANCE = 1.0e-10
# For coordinates up to 2 m, 1e-14 allows about 45 float64 eps of roundoff
# through the offset evaluation. Retain the fixture's pre-refactor envelope.
_OFFSET_COORDINATE_ABSOLUTE_TOLERANCE = 1.0e-14
# At 3 mm, 1e-18 is below three float64 ulps; retain the unit-scaling envelope.
_MILLIMETER_COORDINATE_ABSOLUTE_TOLERANCE = 1.0e-18


def _independent_iges(
    title: str, global_record: str, entities: Sequence[tuple[int, int, str]]
) -> bytes:
    """Lay out handwritten ``(type, form, parameters)`` entities as fixed IGES 5.3."""
    global_lines = [
        f"{global_record[start : start + 72]:72s}G{index + 1:7d}"
        for index, start in enumerate(range(0, len(global_record), 72))
    ]
    directory_lines = []
    parameter_lines = []
    for index, (kind, form, parameters) in enumerate(entities):
        directory = 2 * index + 1
        pointer = index + 1
        first = (kind, pointer, 0, 0, 0, 0, 0, 0, 0)
        second = (kind, 0, 0, 1, form, 0, 0)
        directory_lines.extend(
            (
                "".join(f"{value:8d}" for value in first) + f"D{directory:7d}",
                "".join(f"{value:8d}" for value in second)
                + f"{'SOURCE':8s}{0:8d}D{directory + 1:7d}",
            )
        )
        parameter_lines.append(f"{parameters:64s}{directory:8d}P{pointer:7d}")
    termination = f"S{1:7d}G{len(global_lines):7d}D{len(directory_lines):7d}P{len(parameter_lines):7d}"
    records = [
        f"{title:72s}S{1:7d}",
        *global_lines,
        *directory_lines,
        *parameter_lines,
        f"{termination:72s}T{1:7d}",
    ]
    return ("\n".join(records) + "\n").encode("ascii")


# IGES 5.3 section 4.30 planar offset source: an offset plane and four lines
# of a 2 m x 1 m rectangle at directory pointers D11, D13, D15 and D17.
_OFFSET_SUPPORT = (
    (116, 0, "116,0,0,0.5;"),
    (123, 0, "123,0,0,1;"),
    (123, 0, "123,1,0,0;"),
    (190, 1, "190,1,3,5;"),
    (140, 0, "140,0,0,1,0.25,7;"),
    (110, 0, "110,0,0,0.75,2,0,0.75;"),
    (110, 0, "110,2,0,0.75,2,1,0.75;"),
    (110, 0, "110,2,1,0.75,0,1,0.75;"),
    (110, 0, "110,0,1,0.75,0,0,0.75;"),
)
_OFFSET_GLOBAL = (
    "1H,,1H;,6HPHYDRA,11Hoffset.iges,6HPHYDRA,7Hfixture,"
    "32,38,6,308,15,6HPHYDRA,1.0,6,1HM,1,0.01,"
    "15H20261004.000000,0.000001,2.0,6HPHYDRA,6HPHYDRA,"
    "11,0,15H20261004.000000;"
)


def _offset_face(composites: Sequence[str], *, millimeters: bool = False) -> bytes:
    """Bound the offset plane by the last of ``composites`` (from D19 onward)."""
    boundary = 19 + 2 * (len(composites) - 1)
    entities = (
        *_OFFSET_SUPPORT,
        *((102, 0, composite) for composite in composites),
        (142, 0, f"142,0,9,0,{boundary},1;"),
        (144, 0, f"144,9,1,0,{boundary + 2};"),
    )
    global_record = (
        _OFFSET_GLOBAL.replace("1.0,6,1HM,", "1.0,2,2HMM,")
        if millimeters
        else _OFFSET_GLOBAL
    )
    return _independent_iges(
        "Independent IGES5.3 offset surface source", global_record, entities
    )


@pytest.mark.parametrize("name", SOLIDS)
def test_native_solid_retains_topology_and_pcurves(name: str, tmp_path: Path) -> None:
    model = native_solid(name)
    path = tmp_path / f"{name}.igs"
    write_iges(model, path)
    result = read_iges(path, POLICY, trusted_root=tmp_path)
    assert result.model.topology.num_solids == model.topology.num_solids
    assert result.model.topology.num_faces == model.topology.num_faces
    assert geometry_of(result.model).occurrences == geometry_of(model).occurrences
    assert result.coverage.pcurves_honored == len(geometry_of(model).pcurves)
    assert result.coverage.pcurves_fitted == 0
    assert float(solid_volumes(result.model)[0]) == pytest.approx(
        EXACT_VOLUMES[name], rel=_VOLUME_RELATIVE_TOLERANCE
    )
    assert not result.report.losses


def test_cylinder_seam_pcurves_retain_trimmed_knot_domain_and_edge_use_sense() -> None:
    """Retain the quadric loop pattern: trimmed spline plus oppositely used seam."""
    model = native_solid("cylinder")
    writer = _iges._IgesWriter(model, CadExportPolicy())
    writer.encode()
    changed: set[int] = set()
    for entity in writer.entities:
        if entity.entity_type != 126 or entity.parameters[:2] != ["1", "1"]:
            continue
        values = entity.parameters
        points = np.asarray([float(value) for value in values[12:18]]).reshape(2, 3)
        if points[0, 0] != points[1, 0] or points[0, 1] == points[1, 1]:
            continue
        lower, upper = float(values[18]), float(values[19])
        # As in the earned quadric artifact, the IGES 126 active interval
        # selects only the last portion of a larger, perfectly valid spline.
        # Prepend exact linear continuation without changing that interval.
        extended = np.concatenate(((2.0 * points[0] - points[1])[None], points))
        entity.parameters = [
            "2", *values[1:6],
            *writer.reals((2.0 * lower - upper, 2.0 * lower - upper, lower, upper, upper)),
            "1", "1", "1",
            *writer.reals(extended),
            *values[18:],
        ]
        changed.add(entity.number)
    assert changed
    senses = set()
    for entity in writer.entities:
        if entity.entity_type == 508:
            for position in range(1, len(entity.parameters), 7):
                if int(entity.parameters[position + 6]) in changed:
                    senses.add(entity.parameters[position + 3])
    assert senses == {"0", "1"}
    result = decode_iges_bytes(writer._text(2.0), POLICY)
    assert result.model.topology.num_faces == model.topology.num_faces
    assert result.coverage.pcurves_honored == len(geometry_of(model).pcurves)
    assert result.coverage.pcurves_fitted == 0
    assert float(solid_volumes(result.model)[0]) == pytest.approx(
        EXACT_VOLUMES["cylinder"], rel=_VOLUME_RELATIVE_TOLERANCE
    )


@pytest.mark.parametrize(
    "edge", (0, 2), ids=("first-rational-edge", "opposite-rational-edge")
)
def test_rational_surface_and_curves_survive_native_roundtrip(
    edge: int, tmp_path: Path
) -> None:
    model = native_rational_spline_face()
    path = tmp_path / "rational.igs"
    write_iges(model, path)
    restored = read_iges(path, POLICY, trusted_root=tmp_path)
    patch = restored.model.patches[0]
    original = model.patches[0]
    if not isinstance(patch, BSplineSurfacePatch) or not isinstance(
        original, BSplineSurfacePatch
    ):
        raise TypeError(
            "fixture.surface must retain its spline family; got another carrier"
        )
    np.testing.assert_array_equal(patch.control_points, original.control_points)
    np.testing.assert_array_equal(patch.weights, original.weights)
    np.testing.assert_array_equal(patch.u_knots, original.u_knots)
    np.testing.assert_array_equal(patch.v_knots, original.v_knots)
    assert restored.coverage.pcurves_honored == 4
    restored_geometry, source_geometry = geometry_of(restored.model), geometry_of(model)
    curve = restored_geometry.curves[restored_geometry.edge_curves[edge]]
    source = source_geometry.curves[source_geometry.edge_curves[edge]]
    if not isinstance(curve, BSplineCurve) or not isinstance(source, BSplineCurve):
        raise TypeError(
            "fixture.boundary must retain its spline family; got another carrier"
        )
    np.testing.assert_array_equal(curve.weights, source.weights)
    np.testing.assert_array_equal(curve.control_points, source.control_points)


def test_native_assembly_retains_occurrence_paths_and_placements(tmp_path: Path) -> None:
    model = native_assembly()
    path = tmp_path / "assembly.igs"
    write_iges(model, path)
    result = read_iges(path, POLICY, trusted_root=tmp_path)
    assert geometry_of(result.model).occurrences == geometry_of(model).occurrences
    assert result.model.topology.num_solids == 2
    np.testing.assert_allclose(
        np.asarray(prepare_brep_query(result.model).measures.solid_volumes),
        np.asarray(prepare_brep_query(model).measures.solid_volumes),
        rtol=_VOLUME_RELATIVE_TOLERANCE,
    )


def test_repeated_definition_retains_independent_nested_occurrences(
    tmp_path: Path,
) -> None:
    model = native_repeated_assembly()
    path = tmp_path / "repeated.igs"
    write_iges(model, path)
    result = read_iges(path, POLICY, trusted_root=tmp_path)
    assert result.model.topology.num_solids == 1
    assert geometry_of(result.model).occurrences == geometry_of(model).occurrences
    assert result.model.topology.num_faces == 6
    assert {
        record.path: (record.member_paths, record.child_paths)
        for record in geometry_of(result.model).assembly_containers
    } == {
        record.path: (record.member_paths, record.child_paths)
        for record in geometry_of(model).assembly_containers
    }


def _replace_iges_section(data: bytes, section: str, text: Sequence[str]) -> bytes:
    sections = _iges._lines(data, "test")
    sections[section] = [
        f"{value:72s}{section}{index + 1:7d}"
        for index, value in enumerate(text)
    ]
    termination = "".join(
        f"{letter}{len(sections[letter]):7d}" for letter in "SGDP"
    )
    sections["T"] = [f"{termination:72s}T{1:7d}"]
    return (
        "\n".join(line for letter in "SGDPT" for line in sections[letter]) + "\n"
    ).encode("ascii")


def _with_bounded_descriptor(data: bytes, descriptor: str) -> bytes:
    return _replace_iges_section(
        data, "S",
        [
            "Independent descriptive metadata",
            f"PhydraxBoundedApproximation:{len(descriptor)}",
            *(descriptor[index:index + 72] for index in range(0, len(descriptor), 72)),
        ],
    )


@pytest.fixture(scope="module")
def bounded_iges_export(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[BRepModel, CadExportResult, Path]:
    model = native_intersection_cap()
    path = tmp_path_factory.mktemp("bounded-iges") / "approximated.igs"
    fit = CadCurveFitPolicy(
        tolerance=1.0e-7, maximum_control_points=512, check_samples=1025
    )
    result = write_iges(
        model, path, policy=CadExportPolicy(intersection_approximation=fit)
    )
    return model, result, path


def test_explicit_branch_approximation_emits_coupled_geometry(
    tmp_path: Path,
    bounded_iges_export: tuple[BRepModel, CadExportResult, Path],
) -> None:
    model, result, path = bounded_iges_export
    exact = tmp_path / "exact.igs"
    with pytest.raises(CadInterchangeError) as refused:
        write_iges(model, exact)
    assert refused.value.refusal.reason == "inexact-export"
    assert not exact.exists()
    fit = CadCurveFitPolicy(
        tolerance=1.0e-7, maximum_control_points=512, check_samples=1025
    )
    assert result.report.status == AdapterStatus.DECLARED_LOSS
    approximation = result.approximations[0]
    loss = result.report.losses[0]
    assert loss.path == f"intersection-curve {approximation.branch_id}"
    assert approximation.approximation_id in loss.rationale
    assert f"{approximation.continuous_evidence.distance_bound:.17g}" in loss.rationale
    assert f"{approximation.continuous_evidence.parameter_bound:.17g}" in loss.rationale
    assert (
        f"{approximation.continuous_evidence.first_correspondence_bound:.17g}"
        in loss.rationale
    )
    assert (
        f"{approximation.continuous_evidence.second_correspondence_bound:.17g}"
        in loss.rationale
    )
    assert len(result.report.waivers) == 1
    assert result.report.waivers[0].loss_id == loss.loss_id
    source_branch = geometry_of(model).curves[0]
    if not isinstance(source_branch, IntersectionCurve):
        raise TypeError(
            "fixture.cap_curve must be a coupled intersection; got another carrier"
        )
    assert result.approximations[0].branch_id == source_branch.branch_id
    imported = read_iges(path, POLICY, trusted_root=path.parent)
    restored = imported.model
    assert restored.model_id != model.model_id
    assert imported.source_digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result.report.target_profile is not None
    qualifiers = dict(result.report.target_profile.qualifiers)
    assert qualifiers["artifact-sha256"] == result.receipt.content_sha256
    assert qualifiers["representation"] == "bounded-coupled-approximation"
    descriptor = _iges._decode(path.read_bytes(), POLICY).bounded_descriptor
    assert descriptor == qualifiers["bounded-approximation-descriptor"]
    assert descriptor is not None
    gap = decode_bounded_approximation_descriptor(descriptor)
    assert gap > 0.0
    assert imported.report.status == AdapterStatus.DECLARED_LOSS
    assert imported.report.source_profile is not None
    assert (
        dict(imported.report.source_profile.qualifiers)["bounded-representation-id"]
        == qualifiers["bounded-representation-id"]
    )
    assert imported.source_digest in imported.report.losses[0].rationale
    assert qualifiers["bounded-representation-id"] in imported.report.losses[0].rationale
    restored_geometry = geometry_of(restored)
    edge = restored_geometry.curves[restored_geometry.edge_curves[0]]
    if not isinstance(edge, BSplineCurve):
        raise TypeError("iges.approximated_edge must be a spline; got another carrier")
    interval = np.asarray(restored_geometry.edge_ranges)[0]
    parameters = jnp.linspace(float(interval[0]), float(interval[1]), 257)
    points = np.asarray(edge.evaluate(parameters))
    # Radial distance is 1-Lipschitz. Two fit tolerances retain the pre-refactor
    # sampled acceptance envelope for fitting plus IGES decimal serialization.
    sample_tolerance = 2.0 * fit.tolerance
    np.testing.assert_allclose(
        np.linalg.norm(points[:, :2], axis=1),
        np.sqrt(1.0 - 0.3**2),
        atol=sample_tolerance,
    )
    np.testing.assert_allclose(points[:, 2], 0.3, atol=sample_tolerance)


def test_bounded_representation_gap_scales_to_target_length_unit(
    bounded_iges_export: tuple[BRepModel, CadExportResult, Path],
) -> None:
    _, _, path = bounded_iges_export
    data = path.read_bytes()
    descriptor = _iges._decode(data, POLICY).bounded_descriptor
    assert descriptor is not None
    source_gap = decode_bounded_approximation_descriptor(descriptor)
    target_gap = decode_bounded_approximation_descriptor(
        descriptor, length_factor=1000.0
    )
    assert target_gap >= 1000.0 * source_gap
    assert target_gap == pytest.approx(
        1000.0 * source_gap, rel=4.0 * np.finfo(np.float64).eps
    )
    imported = decode_iges_bytes(
        data, CadImportPolicy(SpatialCoordinateContract(MILLIMETER), LIMITS)
    )
    assert imported.source_length_unit_meters == 1.0
    assert imported.model.coordinate_contract.length_unit == MILLIMETER
    assert imported.report.status == AdapterStatus.DECLARED_LOSS
    assert imported.report.valid
    bounded_loss = next(
        loss for loss in imported.report.losses
        if loss.path == "bounded-coupled-approximation"
    )
    assert not bounded_loss.changes_interpretation


@pytest.mark.parametrize(
    "descriptor",
    (
        '{"representation":"bounded-coupled-approximation"}',
        '{"representation":"bounded-coupled-approximation","gap_bound":1.0}',
    ),
    ids=("marker-only", "forged-gap-only"),
)
def test_incomplete_bounded_metadata_never_grants_gap_allowance(
    descriptor: str, tmp_path: Path,
) -> None:
    path = tmp_path / "exact.igs"
    write_iges(native_solid("box"), path)
    data = _with_bounded_descriptor(path.read_bytes(), descriptor)
    with pytest.raises(CadInterchangeError):
        decode_iges_bytes(data, POLICY)


@pytest.mark.parametrize("change", ("law", "stored-carrier", "truncated"))
def test_bounded_descriptor_and_actual_carrier_tampering_refuse(
    change: str,
    bounded_iges_export: tuple[BRepModel, CadExportResult, Path],
) -> None:
    _, _, path = bounded_iges_export
    data = path.read_bytes()
    decoded = _iges._decode(data, POLICY)
    assert decoded.bounded_descriptor is not None
    if change == "law":
        descriptor = json.loads(decoded.bounded_descriptor)
        descriptor["gap_bound"] = 1.0e6
        data = _with_bounded_descriptor(
            data, json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
        )
    elif change == "truncated":
        sections = _iges._lines(data, "test")
        data = _replace_iges_section(
            data, "S", [line[:72] for line in sections["S"][:-1]]
        )
    else:
        # Corrupt only an interior 3D control point: the endpoints, valid
        # descriptor and declared bound remain untouched. Admission must prove
        # the stored carrier, not trust this descriptor's provenance.
        edge_list = next(e for e in decoded.entries.values() if e.entity_type == 504)
        curve_number = int(_iges._string(edge_list.parameters[1]))
        curve = decoded.entries[curve_number]
        assert curve.entity_type == 126
        count = int(_iges._string(curve.parameters[0])) + 1
        degree = int(_iges._string(curve.parameters[1]))
        assert count > 2
        sections = _iges._lines(data, "test")
        first = sections["D"][curve_number - 1]
        second = sections["D"][curve_number]
        pointer = int(first[8:16]) - 1
        records = int(second[24:32])
        text = "".join(line[:64] for line in sections["P"][pointer:pointer + records])
        tokens = text.split(",")
        coordinate = 1 + 6 + count + degree + 1 + count + 3 * (count // 2) + 2
        token = tokens[coordinate]
        tokens[coordinate] = "9" + " " * (len(token) - 1)
        changed = ",".join(tokens)
        for index in range(records):
            old = sections["P"][pointer + index]
            sections["P"][pointer + index] = changed[index * 64:(index + 1) * 64] + old[64:]
        data = ("\n".join(
            line for letter in "SGDPT" for line in sections[letter]
        ) + "\n").encode("ascii")
    with pytest.raises(CadInterchangeError):
        decode_iges_bytes(data, POLICY)


def test_global_resolution_is_not_a_bounded_gap_allowance() -> None:
    # A handwritten ordinary source: all boundary curves are 0.01 m above
    # their plane, while still forming an exactly closed spatial loop.
    data = _offset_face(["102,4,11,13,15,17;"]).replace(b"0.75", b"0.76")
    sections = _iges._lines(data, "test")
    _, _, values = _iges._global(sections["G"], LIMITS.max_attributes)
    assert _iges._real(values[18], "G19") == 1.0e-6
    # G19 is a coordinate coincidence/discernibility scalar (§2.2.4.3.19),
    # not uncertainty on an edge/surface pair. Even a resolution larger than
    # the actual 0.01 m gap must not weaken the original source policy.
    global_text = "".join(line[:72] for line in sections["G"])
    global_text = global_text.replace(",0.000001,", ",1.0,", 1).rstrip()
    data = _replace_iges_section(
        data, "G",
        [global_text[index:index + 72] for index in range(0, len(global_text), 72)],
    )
    data = _replace_iges_section(data, "S", ["Ordinary independent IGES source"])
    assert _iges._decode(data, POLICY).bounded_descriptor is None
    assert POLICY.relative_geometric_tolerance == 1.0e-8
    with pytest.raises(CadInterchangeError):
        decode_iges_bytes(data, POLICY)


def test_declared_iges_units_scale_geometry_and_preserve_pcurves(tmp_path: Path) -> None:
    model = brep_box(
        (0, 0, 0), (1, 2, 3), coordinate_contract=SpatialCoordinateContract(MILLIMETER)
    )
    path = tmp_path / "millimeters.igs"
    write_iges(model, path)
    result = read_iges(path, POLICY, trusted_root=tmp_path)
    assert result.source_length_unit_meters == 1.0e-3
    assert result.report.source_profile is not None
    qualifiers = dict(result.report.source_profile.qualifiers)
    assert qualifiers["minimum-coordinate-discernibility-source-units"] == "1.0E-07"
    assert float(
        qualifiers["minimum-coordinate-discernibility-target-units"]
    ) == pytest.approx(1.0e-10)
    assert _iges._decode(path.read_bytes(), POLICY).bounded_descriptor is None
    assert POLICY.relative_geometric_tolerance == 1.0e-8
    np.testing.assert_allclose(
        np.max(np.asarray(geometry_of(result.model).vertex_points), axis=0),
        [1.0e-3, 2.0e-3, 3.0e-3],
        rtol=0.0,
        atol=_MILLIMETER_COORDINATE_ABSOLUTE_TOLERANCE,
    )
    assert result.coverage.pcurves_honored == len(geometry_of(model).pcurves)
    assert float(solid_volumes(result.model)[0]) == pytest.approx(
        6.0e-9, rel=_VOLUME_RELATIVE_TOLERANCE
    )


def test_empty_native_model_has_no_synthetic_geometry_or_occurrences(
    tmp_path: Path,
) -> None:
    model = native_empty_model()
    path = tmp_path / "empty.igs"
    write_iges(model, path)
    result = read_iges(path, POLICY, trusted_root=tmp_path)
    assert geometry_of(result.model).geometry_id == geometry_of(model).geometry_id
    assert result.model.topology.num_solids == 0
    assert result.model.topology.num_faces == 0
    assert geometry_of(result.model).occurrences == ()
    assert np.asarray(geometry_of(result.model).vertex_points).shape == (0, 3)
    assert not result.report.losses


def test_native_writer_is_deterministic(tmp_path: Path) -> None:
    first = write_iges(native_assembly(), tmp_path / "a.igs")
    second = write_iges(native_assembly(), tmp_path / "b.igs")
    assert first.receipt.content_sha256 == second.receipt.content_sha256
    assert (
        Path(first.receipt.destination).read_bytes()
        == Path(second.receipt.destination).read_bytes()
    )


@pytest.mark.parametrize(
    "limits",
    (
        replace(LIMITS, max_bytes=128),
        replace(LIMITS, max_nodes=1),
        replace(LIMITS, max_depth=1),
    ),
    ids=("byte-limit", "entity-limit", "reference-depth-limit"),
)
def test_decoder_enforces_byte_entity_and_depth_limits(
    limits: ResourceLimits, tmp_path: Path
) -> None:
    path = tmp_path / "box.igs"
    write_iges(native_solid("box"), path)
    data = path.read_bytes()
    with pytest.raises(CadInterchangeError) as refused:
        decode_iges_bytes(data, CadImportPolicy(CONTRACT, limits))
    assert refused.value.refusal.reason == "limit"


def test_declared_trim_refuses_stale_topology(tmp_path: Path) -> None:
    source = native_solid("box")
    domains = list(source.trim_domains)
    domains[0] = TrimDomain(
        PolygonTrimLoop([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
    )
    model = BRepModel(
        patches=source.patches,
        parameter_bounds=source.parameter_bounds,
        orientation=source.orientation,
        trim_domains=tuple(domains),
        topology=source.topology,
        coordinate_contract=source.coordinate_contract,
        mesh_vertices=source.mesh_vertices,
        mesh_faces=source.mesh_faces,
        triangle_face_ids=source.triangle_face_ids,
        triangle_parameters=source.triangle_parameters,
        physical_tags=source.physical_tags,
        report=source.report,
        geometry=source.geometry,
        tessellation_deviation_bounds=source.tessellation_deviation_bounds,
        tessellation_normal_bounds=source.tessellation_normal_bounds,
        mesh_vertex_source_dimensions=source.mesh_vertex_source_dimensions,
        mesh_vertex_source_indices=source.mesh_vertex_source_indices,
        mesh_vertex_parameters=source.mesh_vertex_parameters,
    )
    path = tmp_path / "stale.igs"
    with pytest.raises(CadInterchangeError) as refused:
        write_iges(model, path)
    assert refused.value.refusal.reason == "inexact-export"
    assert "face:0" in refused.value.refusal.chain
    assert not path.exists()


@pytest.mark.parametrize(
    "policy",
    (CadExportPolicy(maximum_entities=2), CadExportPolicy(maximum_bytes=128)),
    ids=("entity-limit", "byte-limit"),
)
def test_export_budget_failure_leaves_no_destination(
    policy: CadExportPolicy, tmp_path: Path
) -> None:
    path = tmp_path / "too-large.igs"
    with pytest.raises(CadInterchangeError) as refused:
        write_iges(native_solid("box"), path, policy=policy)
    assert refused.value.refusal.reason == "limit"
    assert not path.exists()


def test_unsupported_required_geometry_has_dependency_chain(tmp_path: Path) -> None:
    path = tmp_path / "box.igs"
    write_iges(native_solid("box"), path)
    lines = path.read_text().splitlines()
    target = next(
        i for i, line in enumerate(lines) if line[72] == "D" and int(line[:8]) == 190
    )
    pointer = int(lines[target][8:16])
    number = int(lines[target][73:80])
    lines[target] = f"{114:8d}" + lines[target][8:]
    lines[target + 1] = f"{114:8d}" + lines[target + 1][8:]
    parameter = next(
        i for i, line in enumerate(lines) if line[72] == "P" and int(line[73:]) == pointer
    )
    lines[parameter] = "114" + lines[parameter][3:]
    with pytest.raises(CadInterchangeError) as refused:
        decode_iges_bytes(("\n".join(lines) + "\n").encode("ascii"), POLICY)
    assert refused.value.refusal.reason == "unsupported-entity"
    assert refused.value.refusal.entity == f"D{number} E114"
    assert len(refused.value.refusal.chain) >= 2


@pytest.mark.parametrize("kind", ("box", "cylinder", "sphere", "torus"))
def test_independent_ocp_fixture_decodes_without_ocp_algorithms(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import (
        BRepPrimAPI_MakeBox,
        BRepPrimAPI_MakeCylinder,
        BRepPrimAPI_MakeSphere,
        BRepPrimAPI_MakeTorus,
    )
    from OCP.IGESControl import IGESControl_Controller, IGESControl_Writer

    # Keep OCCT's native millimeters for both model-space and parameter-space
    # lengths; importing converts the source to the SI contract.
    factories = {
        "box": lambda: BRepPrimAPI_MakeBox(1000.0, 2000.0, 3000.0).Shape(),
        "cylinder": lambda: BRepPrimAPI_MakeCylinder(1000.0, 2000.0).Shape(),
        "sphere": lambda: BRepPrimAPI_MakeSphere(1500.0).Shape(),
        "torus": lambda: BRepPrimAPI_MakeTorus(3000.0, 1000.0).Shape(),
    }
    IGESControl_Controller.Init_s()
    writer = IGESControl_Writer("MM", 1)
    assert writer.AddShape(factories[kind]())
    writer.ComputeModel()
    path = tmp_path / "reference.igs"
    assert writer.Write(str(path))
    data = path.read_bytes()
    original_import = builtins.__import__

    def import_without_ocp(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> object:
        if name == "OCP" or name.startswith("OCP."):
            raise AssertionError(
                "iges.decoder.dependencies must exclude OCP; got an OCP import"
            )
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_ocp)
    result = decode_iges_bytes(data, POLICY)
    assert result.model.topology.num_solids == 1
    assert result.coverage.pcurves_fitted == 0
    assert float(solid_volumes(result.model)[0]) == pytest.approx(
        EXACT_VOLUMES[kind], rel=_VOLUME_RELATIVE_TOLERANCE
    )


def test_revolution_export_uses_standard_parameter_space_for_independent_ocp(
    tmp_path: Path,
) -> None:
    # IGES 120 parameterizes (generatrix t, angle theta in radians); OCCT's
    # independent reader rebuilds faces from those p-curves, so a revolved
    # rational ellipse keeps its Pappus volume and area only if they are
    # written in that space with a consistent normal.
    pytest.importorskip("OCP")
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.IGESControl import IGESControl_Reader

    radius, a, b = 3.0, 1.0, 0.5
    weight = float(np.sqrt(0.5))
    meridian = phx.geometry.BSplineCurve(
        [
            [radius + a, 0.0],
            [radius + a, b],
            [radius, b],
            [radius - a, b],
            [radius - a, 0.0],
            [radius - a, -b],
            [radius, -b],
            [radius + a, -b],
            [radius + a, 0.0],
        ],
        [1.0, weight, 1.0, weight, 1.0, weight, 1.0, weight, 1.0],
        [0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0],
        2,
    )
    model = phx.geometry.brep_revolution(
        phx.geometry.PlanarProfile(
            phx.geometry.ProfilePlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            phx.geometry.ProfileLoop(((radius + a, 0.0),), (meridian,)),
        ),
        (0.0, 0.0),
        (0.0, 1.0),
        coordinate_contract=CONTRACT,
        tessellation=phx.geometry.BRepTessellationPolicy(realize=False),
    )
    path = tmp_path / "revolution.igs"
    assert write_iges(model, path).report.valid
    reader = IGESControl_Reader()
    assert reader.ReadFile(str(path)) == IFSelect_RetDone
    reader.TransferRoots()
    shape = reader.OneShape()
    assert BRepCheck_Analyzer(shape).IsValid()
    angles = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
    perimeter = (
        float(np.mean(np.hypot(a * np.sin(angles), b * np.cos(angles)))) * 2 * np.pi
    )
    # OCCT reports millimeters.
    volume, area = GProp_GProps(), GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, volume, 1.0e-10)
    BRepGProp.SurfaceProperties_s(shape, area, 1.0e-10)
    assert volume.Mass() * 1.0e-9 == pytest.approx(
        2.0 * np.pi * radius * np.pi * a * b, rel=_OCP_MEASURE_RELATIVE_TOLERANCE
    )
    assert area.Mass() * 1.0e-6 == pytest.approx(
        2.0 * np.pi * radius * perimeter, rel=_OCP_MEASURE_RELATIVE_TOLERANCE
    )


@pytest.mark.parametrize("millimeters", (False, True))
def test_independent_offset_surface_source_retains_tree_without_fitted_pcurves(
    millimeters: bool,
    tmp_path: Path,
) -> None:
    """IGES5.3 section4.30 source; no native writer or external kernel."""
    factor = 1.0e-3 if millimeters else 1.0
    imported = decode_iges_bytes(
        _offset_face(("102,4,11,13,15,17;",), millimeters=millimeters), POLICY
    )
    assert imported.report.valid
    assert imported.coverage.pcurves_fitted == 0
    (patch,) = imported.model.patches
    assert isinstance(patch, OffsetSurface)
    assert isinstance(patch.base, PlanePatch)
    assert float(patch.distance) == 0.25 * factor
    np.testing.assert_allclose(
        patch.evaluate(jnp.asarray([[0.2, 0.7], [1.7, 0.2]], dtype=jnp.float64) * factor),
        np.asarray([[0.2, 0.7, 0.75], [1.7, 0.2, 0.75]]) * factor,
        rtol=0.0,
        atol=_OFFSET_COORDINATE_ABSOLUTE_TOLERANCE,
    )
    assert (
        imported.model.topology.num_faces == 1 and imported.model.topology.num_edges == 4
    )
    area = prepare_brep_query(imported.model).measures.face_areas[0]
    assert float(area) == pytest.approx(
        2.0 * factor**2, rel=_OFFSET_AREA_RELATIVE_TOLERANCE
    )
    receipt = save_brep_archive(imported.model, tmp_path / "independent-offset.phx")
    restored = load_brep_archive(receipt.path)
    assert restored.model_id == imported.model.model_id == receipt.model_id
    assert restored.source_id == imported.model.source_id
    assert isinstance(restored.patches[0], OffsetSurface)
    np.testing.assert_array_equal(
        restored.patches[0].evaluate(jnp.asarray([0.3, 0.4], dtype=jnp.float64) * factor),
        patch.evaluate(jnp.asarray([0.3, 0.4], dtype=jnp.float64) * factor),
    )


def test_nested_composite_boundary_keeps_flat_segment_order() -> None:
    flat = decode_iges_bytes(_offset_face(("102,4,11,13,15,17;",)), POLICY)
    nested = decode_iges_bytes(
        _offset_face(("102,2,11,13;", "102,2,15,17;", "102,2,19,21;")), POLICY
    )
    assert nested.report.valid
    assert nested.model.topology.num_faces == 1
    assert nested.model.topology.num_edges == 4
    np.testing.assert_array_equal(
        np.asarray(nested.model.require_geometry().vertex_points),
        np.asarray(flat.model.require_geometry().vertex_points),
    )
    area = prepare_brep_query(nested.model).measures.face_areas[0]
    assert float(area) == pytest.approx(2.0, rel=_OFFSET_AREA_RELATIVE_TOLERANCE)


@pytest.mark.parametrize(
    ("depth", "reason", "message"),
    (
        # Sixteen uses of one line stay sixteen segments, so the loop is
        # disconnected rather than a deduplicated open boundary.
        (4, "inconsistent-geometry", "disconnected"),
        # 2046 unfolded references exceed the 256-attribute import limit.
        (10, "limit", "unfolded_references"),
    ),
)
def test_shared_composite_references_are_admitted_before_expansion(
    depth: int, reason: str, message: str
) -> None:
    composites = ["102,2,11,11;"]
    for level in range(1, depth):
        previous = 19 + 2 * (level - 1)
        composites.append(f"102,2,{previous},{previous};")
    policy = CadImportPolicy(CONTRACT, replace(LIMITS, max_attributes=256))
    with pytest.raises(CadInterchangeError) as refused:
        decode_iges_bytes(_offset_face(composites), policy)
    assert refused.value.refusal.reason == reason
    assert message in refused.value.refusal.message


@pytest.mark.parametrize(
    "failure", ("nonbinary-property", "reversed-u", "outside-u", "outside-v")
)
def test_independent_spline_surface_refuses_invalid_declared_domain_with_provenance(
    failure: str,
) -> None:
    header = [1, 1, 1, 1, 0, 0, 1, 0, 0]
    domain = [0, 1, 0, 1]
    if failure == "nonbinary-property":
        header[7] = 2
    elif failure == "reversed-u":
        domain[:2] = [1, 0]
    elif failure == "outside-u":
        domain[1] = 2
    else:
        domain[2] = -1
    surface = (
        ",".join(
            map(
                str,
                [
                    128,
                    *header,
                    0,
                    0,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                    1,
                    0,
                    *domain,
                ],
            )
        )
        + ";"
    )
    global_record = (
        "1H,,1H;,6HPHYDRA,11Hspline.iges,6HPHYDRA,7Hfixture,"
        "32,38,6,308,15,6HPHYDRA,1.0,6,1HM,1,0.01,"
        "15H20261004.000000,0.000001,2.0,6HPHYDRA,6HPHYDRA,"
        "11,0,15H20261004.000000;"
    )
    globals_ = [
        f"{global_record[start : start + 72]:72s}G{index + 1:7d}"
        for index, start in enumerate(range(0, len(global_record), 72))
    ]
    directories, parameters = [], []
    for index, (kind, payload) in enumerate(((128, surface), (144, "144,1,0,0;"))):
        number = 2 * index + 1
        chunks = [payload[start : start + 64] for start in range(0, len(payload), 64)]
        first = (kind, len(parameters) + 1, 0, 0, 0, 0, 0, 0, 0)
        second = (kind, 0, 0, len(chunks), 0, 0, 0)
        directories.extend(
            (
                "".join(f"{value:8d}" for value in first) + f"D{number:7d}",
                "".join(f"{value:8d}" for value in second)
                + f"{'SOURCE':8s}{0:8d}D{number + 1:7d}",
            )
        )
        for chunk in chunks:
            parameters.append(f"{chunk:64s}{number:8d}P{len(parameters) + 1:7d}")
    termination = f"S{1:7d}G{len(globals_):7d}D{len(directories):7d}P{len(parameters):7d}"
    data = (
        "\n".join(
            [
                f"{'Independent IGES5.3 declared spline domain':72s}S{1:7d}",
                *globals_,
                *directories,
                *parameters,
                f"{termination:72s}T{1:7d}",
            ]
        )
        + "\n"
    ).encode("ascii")
    with pytest.raises(CadInterchangeError) as refused:
        decode_iges_bytes(data, POLICY)
    assert refused.value.refusal.reason == "malformed"
    assert len(refused.value.refusal.chain) == 2
    assert refused.value.refusal.chain[-1] == refused.value.refusal.entity
    manifest = refused.value.resource_manifest
    assert manifest is not None
    assert manifest.size_bytes == len(data)


def test_file_and_byte_decoders_preserve_geometry_and_report(tmp_path: Path) -> None:
    path = tmp_path / "dispatch.igs"
    write_iges(native_repeated_assembly(), path)
    from_file = read_iges(path, POLICY, trusted_root=tmp_path)
    from_bytes = decode_iges_bytes(path.read_bytes(), POLICY, source_path=str(path))
    assert from_file.model.model_id == from_bytes.model.model_id
    assert from_file.report.report_id == from_bytes.report.report_id
    assert from_file.coverage == from_bytes.coverage
    assert from_file.provenance == from_bytes.provenance
    assert from_file.source_digest == from_bytes.source_digest
    assert (
        from_file.resource_manifest.size_bytes == from_bytes.resource_manifest.size_bytes
    )


def test_export_report_preserves_unit_and_policy_evidence(tmp_path: Path) -> None:
    model = native_solid("box")
    result = write_iges(model, tmp_path / "report.igs")
    assert result.report.target_id == result.receipt.receipt_id
    assert result.report.source_id == model.model_id
    assert result.report.stage == "adapter"
    assert result.report.coordinate_mapping == (
        f"IGES unit of {model.coordinate_contract.length_unit.symbol}",
    )
    assert len(result.report.assumptions) == 1
    assert result.report.target_profile.qualifiers == (("profile", "iges-5.3"),)
    assert result.report.status == AdapterStatus.LOSSLESS
    assert not result.report.losses
    assert not result.report.waivers
