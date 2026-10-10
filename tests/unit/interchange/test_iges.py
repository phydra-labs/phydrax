"""Native IGES exact topology/carriers, bounded decoding and independent fixtures."""

from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._external_resource import ResourceLimits
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._atlas import PolygonTrimLoop, TrimDomain
from phydrax.geometry.brep import brep_box, BRepGeometry, BRepModel, prepare_brep_query
from phydrax.geometry.brep._intersection_curve import IntersectionCurve
from phydrax.geometry.brep._patches import (
    BSplineCurve,
    BSplineSurfacePatch,
    OffsetSurface,
    PlanePatch,
)
from phydrax.interchange import (
    AdapterStatus,
    CadCurveFitPolicy,
    CadExportPolicy,
    CadImportPolicy,
    CadInterchangeError,
    load_brep_archive,
    save_brep_archive,
)
from phydrax.interchange._iges import decode_iges_bytes, read_iges, write_iges
from phydrax.units import MILLIMETER
from tests._support.cad_models import (
    CONTRACT,
    EXACT_VOLUMES,
    native_assembly,
    native_empty_model,
    native_intersection_cap,
    native_rational_spline_face,
    native_repeated_assembly,
    native_solid,
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


def _volume(model: BRepModel) -> float:
    return float(prepare_brep_query(model).measures.solid_volumes[0])


def _geometry(model: BRepModel) -> BRepGeometry:
    geometry = model.geometry
    if geometry is None:
        raise TypeError("Native CAD qualification requires exact geometry.")
    return geometry


@pytest.mark.parametrize("name", SOLIDS)
def test_native_solid_retains_topology_and_pcurves(name: str, tmp_path: Path) -> None:
    model = native_solid(name)
    path = tmp_path / f"{name}.igs"
    write_iges(model, path)
    result = read_iges(path, POLICY, trusted_root=tmp_path)
    assert result.model.topology.num_solids == model.topology.num_solids
    assert result.model.topology.num_faces == model.topology.num_faces
    assert _geometry(result.model).occurrences == _geometry(model).occurrences
    assert result.coverage.pcurves_honored == len(_geometry(model).pcurves)
    assert result.coverage.pcurves_fitted == 0
    assert _volume(result.model) == pytest.approx(EXACT_VOLUMES[name], rel=2.0e-6)
    assert not result.report.losses


def test_rational_surface_and_curves_survive_native_roundtrip(tmp_path: Path) -> None:
    model = native_rational_spline_face()
    path = tmp_path / "rational.igs"
    write_iges(model, path)
    restored = read_iges(path, POLICY, trusted_root=tmp_path)
    patch = restored.model.patches[0]
    original = model.patches[0]
    if not isinstance(patch, BSplineSurfacePatch) or not isinstance(
        original, BSplineSurfacePatch
    ):
        raise TypeError("The rational surface fixture must retain its spline family.")
    np.testing.assert_array_equal(patch.control_points, original.control_points)
    np.testing.assert_array_equal(patch.weights, original.weights)
    np.testing.assert_array_equal(patch.u_knots, original.u_knots)
    np.testing.assert_array_equal(patch.v_knots, original.v_knots)
    assert restored.coverage.pcurves_honored == 4
    for edge in (0, 2):
        restored_geometry, source_geometry = _geometry(restored.model), _geometry(model)
        curve = restored_geometry.curves[restored_geometry.edge_curves[edge]]
        source = source_geometry.curves[source_geometry.edge_curves[edge]]
        if not isinstance(curve, BSplineCurve) or not isinstance(source, BSplineCurve):
            raise TypeError(
                "The rational boundary fixture must retain its spline family."
            )
        np.testing.assert_array_equal(curve.weights, source.weights)
        np.testing.assert_array_equal(curve.control_points, source.control_points)


def test_native_assembly_retains_occurrence_paths_and_placements(tmp_path: Path) -> None:
    model = native_assembly()
    path = tmp_path / "assembly.igs"
    write_iges(model, path)
    result = read_iges(path, POLICY, trusted_root=tmp_path)
    assert _geometry(result.model).occurrences == _geometry(model).occurrences
    assert result.model.topology.num_solids == 2
    np.testing.assert_allclose(
        np.asarray(prepare_brep_query(result.model).measures.solid_volumes),
        np.asarray(prepare_brep_query(model).measures.solid_volumes),
        rtol=2.0e-6,
    )


def test_repeated_definition_retains_independent_nested_occurrences(
    tmp_path: Path,
) -> None:
    model = native_repeated_assembly()
    path = tmp_path / "repeated.igs"
    write_iges(model, path)
    result = read_iges(path, POLICY, trusted_root=tmp_path)
    assert result.model.topology.num_solids == 1
    assert _geometry(result.model).occurrences == _geometry(model).occurrences
    assert result.model.topology.num_faces == 6
    assert {
        record.path: (record.member_paths, record.child_paths)
        for record in _geometry(result.model).assembly_containers
    } == {
        record.path: (record.member_paths, record.child_paths)
        for record in _geometry(model).assembly_containers
    }


def test_explicit_branch_approximation_emits_the_actual_coupled_geometry(
    tmp_path: Path,
) -> None:
    model = native_intersection_cap()
    exact = tmp_path / "exact.igs"
    with pytest.raises(CadInterchangeError) as refused:
        write_iges(model, exact)
    assert refused.value.refusal.reason == "inexact-export"
    assert not exact.exists()
    fit = CadCurveFitPolicy(
        tolerance=1.0e-7, maximum_control_points=512, check_samples=1025
    )
    path = tmp_path / "approximated.igs"
    result = write_iges(
        model, path, policy=CadExportPolicy(intersection_approximation=fit)
    )
    assert result.report.status == AdapterStatus.DECLARED_LOSS
    source_branch = _geometry(model).curves[0]
    if not isinstance(source_branch, IntersectionCurve):
        raise TypeError("The cap fixture must retain its coupled intersection source.")
    assert result.approximations[0].branch_id == source_branch.branch_id
    restored = read_iges(path, POLICY, trusted_root=tmp_path).model
    restored_geometry = _geometry(restored)
    edge = restored_geometry.curves[restored_geometry.edge_curves[0]]
    if not isinstance(edge, BSplineCurve):
        raise TypeError("Explicit branch approximation must emit a spline carrier.")
    interval = np.asarray(restored_geometry.edge_ranges)[0]
    parameters = jnp.linspace(float(interval[0]), float(interval[1]), 257)
    points = np.asarray(edge.evaluate(parameters))
    np.testing.assert_allclose(
        np.linalg.norm(points[:, :2], axis=1), np.sqrt(1.0 - 0.3**2), atol=2.0e-7
    )
    np.testing.assert_allclose(points[:, 2], 0.3, atol=2.0e-7)


def test_declared_iges_units_scale_geometry_and_preserve_pcurves(tmp_path: Path) -> None:
    model = brep_box(
        (0, 0, 0), (1, 2, 3), coordinate_contract=SpatialCoordinateContract(MILLIMETER)
    )
    path = tmp_path / "millimeters.igs"
    write_iges(model, path)
    result = read_iges(path, POLICY, trusted_root=tmp_path)
    assert result.source_length_unit_meters == 1.0e-3
    np.testing.assert_allclose(
        np.max(np.asarray(_geometry(result.model).vertex_points), axis=0),
        [1.0e-3, 2.0e-3, 3.0e-3],
        rtol=0.0,
        atol=1.0e-18,
    )
    assert result.coverage.pcurves_honored == len(_geometry(model).pcurves)
    assert _volume(result.model) == pytest.approx(6.0e-9, rel=2.0e-6)


def test_empty_native_model_has_no_synthetic_geometry_or_occurrences(
    tmp_path: Path,
) -> None:
    model = native_empty_model()
    path = tmp_path / "empty.igs"
    write_iges(model, path)
    result = read_iges(path, POLICY, trusted_root=tmp_path)
    assert _geometry(result.model).geometry_id == _geometry(model).geometry_id
    assert result.model.topology.num_solids == 0
    assert result.model.topology.num_faces == 0
    assert _geometry(result.model).occurrences == ()
    assert np.asarray(_geometry(result.model).vertex_points).shape == (0, 3)
    assert not result.report.losses


def test_native_writer_is_deterministic(tmp_path: Path) -> None:
    first = write_iges(native_assembly(), tmp_path / "a.igs")
    second = write_iges(native_assembly(), tmp_path / "b.igs")
    assert first.receipt.content_sha256 == second.receipt.content_sha256
    assert (
        Path(first.receipt.destination).read_bytes()
        == Path(second.receipt.destination).read_bytes()
    )


def test_decoder_enforces_byte_entity_and_depth_limits(tmp_path: Path) -> None:
    path = tmp_path / "box.igs"
    write_iges(native_solid("box"), path)
    data = path.read_bytes()
    for limits in (
        replace(LIMITS, max_bytes=128),
        replace(LIMITS, max_nodes=1),
        replace(LIMITS, max_depth=1),
    ):
        with pytest.raises(CadInterchangeError) as refused:
            decode_iges_bytes(data, CadImportPolicy(CONTRACT, limits))
        assert refused.value.refusal.reason == "limit"


def test_authoritative_trim_is_never_replaced_by_stale_topology(tmp_path: Path) -> None:
    source = native_solid("box")
    fields = {
        name: getattr(source, name)
        for name in (
            "patches",
            "parameter_bounds",
            "orientation",
            "trim_domains",
            "topology",
            "coordinate_contract",
            "mesh_vertices",
            "mesh_faces",
            "triangle_face_ids",
            "triangle_parameters",
            "physical_tags",
            "report",
            "geometry",
            "tessellation_deviation_bounds",
            "tessellation_normal_bounds",
            "mesh_vertex_source_dimensions",
            "mesh_vertex_source_indices",
            "mesh_vertex_parameters",
        )
    }
    domains = list(source.trim_domains)
    domains[0] = TrimDomain(
        PolygonTrimLoop([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
    )
    fields["trim_domains"] = tuple(domains)
    model = BRepModel(**fields)
    path = tmp_path / "stale.igs"
    with pytest.raises(CadInterchangeError) as refused:
        write_iges(model, path)
    assert refused.value.refusal.reason == "inexact-export"
    assert "face:0" in refused.value.refusal.chain
    assert not path.exists()


def test_export_budget_failure_leaves_no_destination(tmp_path: Path) -> None:
    path = tmp_path / "too-large.igs"
    with pytest.raises(CadInterchangeError) as refused:
        write_iges(native_solid("box"), path, policy=CadExportPolicy(maximum_entities=2))
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
    # lengths; importing converts the complete source to the SI contract.
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
                "Native IGES decoding invoked the independent fixture producer."
            )
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_ocp)
    result = decode_iges_bytes(data, POLICY)
    assert result.model.topology.num_solids == 1
    assert result.coverage.pcurves_fitted == 0
    assert _volume(result.model) == pytest.approx(EXACT_VOLUMES[kind], rel=2.0e-6)


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
        2.0 * np.pi * radius * np.pi * a * b, rel=1.0e-7
    )
    assert area.Mass() * 1.0e-6 == pytest.approx(
        2.0 * np.pi * radius * perimeter, rel=1.0e-7
    )


@pytest.mark.parametrize("millimeters", (False, True))
def test_independent_offset_surface_source_retains_tree_without_fitted_pcurves(
    millimeters: bool,
    tmp_path: Path,
) -> None:
    """IGES5.3 section4.30 source; no native writer or external kernel."""
    entities = (
        (116, 0, "116,0,0,0.5;"),
        (123, 0, "123,0,0,1;"),
        (123, 0, "123,1,0,0;"),
        (190, 1, "190,1,3,5;"),
        (140, 0, "140,0,0,1,0.25,7;"),
        (110, 0, "110,0,0,0.75,2,0,0.75;"),
        (110, 0, "110,2,0,0.75,2,1,0.75;"),
        (110, 0, "110,2,1,0.75,0,1,0.75;"),
        (110, 0, "110,0,1,0.75,0,0,0.75;"),
        (102, 0, "102,4,11,13,15,17;"),
        (142, 0, "142,0,9,0,19,1;"),
        (144, 0, "144,9,1,0,21;"),
    )
    global_record = (
        "1H,,1H;,6HPHYDRA,11Hoffset.iges,6HPHYDRA,7Hfixture,"
        "32,38,6,308,15,6HPHYDRA,1.0,6,1HM,1,0.01,"
        "15H20261004.000000,0.000001,2.0,6HPHYDRA,6HPHYDRA,"
        "11,0,15H20261004.000000;"
    )
    if millimeters:
        global_record = global_record.replace("1.0,6,1HM,", "1.0,2,2HMM,")
    factor = 1.0e-3 if millimeters else 1.0
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
        f"{'Independent IGES5.3 offset surface source':72s}S{1:7d}",
        *global_lines,
        *directory_lines,
        *parameter_lines,
        f"{termination:72s}T{1:7d}",
    ]
    imported = decode_iges_bytes(("\n".join(records) + "\n").encode("ascii"), POLICY)
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
        atol=1e-14,
    )
    assert (
        imported.model.topology.num_faces == 1 and imported.model.topology.num_edges == 4
    )
    area = prepare_brep_query(imported.model).measures.face_areas[0]
    assert float(area) == pytest.approx(2.0 * factor**2, rel=1e-10)
    receipt = save_brep_archive(imported.model, tmp_path / "independent-offset.phx")
    restored = load_brep_archive(receipt.path)
    assert restored.model_id == imported.model.model_id == receipt.model_id
    assert restored.source_id == imported.model.source_id
    assert isinstance(restored.patches[0], OffsetSurface)
    np.testing.assert_array_equal(
        restored.patches[0].evaluate(jnp.asarray([0.3, 0.4], dtype=jnp.float64) * factor),
        patch.evaluate(jnp.asarray([0.3, 0.4], dtype=jnp.float64) * factor),
    )


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
