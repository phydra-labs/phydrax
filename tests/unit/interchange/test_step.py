"""Native STEP Part 21 interchange contracts.

Independent references: closed-form solid volumes, and Open CASCADE (optional
comparison provider) reading and writing the same solids, compared through
topology counts and OCCT `GProp` volumes.
"""

import builtins
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Literal

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._external_resource import ResourceLimits
from phydrax.geometry._atlas import PolygonTrimLoop, TrimDomain
from phydrax.geometry.brep._model import BRepAssemblyContainer
from phydrax.units import MILLIMETER
from tests._support.cad_models import (
    CONTRACT,
    EXACT_VOLUMES,
    native_assembly,
    native_empty_model,
    native_rational_spline_face,
    native_repeated_assembly,
    native_sector_face,
    native_solid,
    QUARTER_TURN,
    solid_volumes,
    SOLIDS,
)


interchange = phx.interchange
LIMITS = ResourceLimits(
    max_bytes=1 << 24,
    max_depth=64,
    max_nodes=200_000,
    max_attributes=4_000_000,
    max_losses=0,
)
POLICY = interchange.CadImportPolicy(CONTRACT, LIMITS)
# Open CASCADE writes STEP in millimetres and reports shapes read from STEP in millimetres.
OCCT_POLICY = interchange.CadImportPolicy(
    phx._physical.SpatialCoordinateContract(MILLIMETER), LIMITS
)


def _write(model: phx.geometry.brep.BRepModel, path: Path) -> bytes:
    result = interchange.write_step(model, path)
    assert result.report.valid
    return path.read_bytes()


def _read(
    data: bytes, policy: interchange.CadImportPolicy = POLICY
) -> interchange.CadImportResult:
    return interchange.decode_step_bytes(data, policy)


def _refusal(
    data: bytes, policy: interchange.CadImportPolicy = POLICY
) -> interchange.CadRefusal:
    with pytest.raises(interchange.CadInterchangeError) as caught:
        _read(data, policy)
    return caught.value.refusal


def _counts(geometry: phx.geometry.brep.BRepGeometry) -> tuple[int, ...]:
    topology = geometry.topology()
    return (
        topology.num_solids,
        topology.num_faces,
        topology.num_edges,
        topology.num_vertices,
        geometry.euler_characteristic(),
    )


@pytest.mark.parametrize("schema", ["AP203", "AP214", "AP242"])
def test_empty_shape_round_trip_has_no_invented_topology(
    schema: Literal["AP203", "AP214", "AP242"], tmp_path: Path
) -> None:
    native = native_empty_model()
    path = tmp_path / "empty.step"
    interchange.write_step(native, path, schema=schema)
    restored = _read(path.read_bytes()).model
    assert restored.geometry is not None and native.geometry is not None
    assert restored.geometry.geometry_id == native.geometry.geometry_id
    assert restored.geometry.vertex_points.shape == (0, 3)
    assert restored.geometry.edge_ranges.shape == (0, 2)
    assert restored.geometry.occurrences == ()
    assert restored.patches == ()
    assert restored.topology.num_solids == 0
    assert restored.mesh_vertices.shape == (0, 3)
    assert restored.mesh_faces.shape == (0, 3)


@pytest.mark.parametrize("name", SOLIDS)
def test_native_solid_round_trips_to_canonical_fixed_point(
    name: str, tmp_path: Path
) -> None:
    native = native_solid(name)
    first = _read(_write(native, tmp_path / "first.step"))
    second_bytes = _write(first.model, tmp_path / "second.step")
    second = _read(second_bytes)
    assert native.geometry is not None and first.model.geometry is not None
    assert second.model.geometry is not None
    # One write/read reaches the documented canonical form; it is then stable.
    assert second.model.geometry.geometry_id == first.model.geometry.geometry_id
    assert _write(second.model, tmp_path / "third.step") == second_bytes
    assert _counts(first.model.geometry) == _counts(native.geometry)
    np.testing.assert_allclose(
        np.sort(np.asarray(first.model.geometry.vertex_points), axis=0),
        np.sort(np.asarray(native.geometry.vertex_points), axis=0),
        rtol=0.0,
        atol=1e-15,
    )
    assert solid_volumes(first.model)[0] == pytest.approx(EXACT_VOLUMES[name], rel=1e-10)
    assert first.report.status == interchange.AdapterStatus.LOSSLESS
    assert first.coverage.pcurves_fitted == 0


def test_axis_aligned_box_round_trip_preserves_exact_geometry_identity(
    tmp_path: Path,
) -> None:
    native = native_solid("box")
    model = _read(_write(native, tmp_path / "box.step")).model
    assert native.geometry is not None and model.geometry is not None
    assert model.geometry.geometry_id == native.geometry.geometry_id


def test_writer_is_deterministic(tmp_path: Path) -> None:
    model = native_solid("plate")
    assert _write(model, tmp_path / "a.step") == _write(model, tmp_path / "b.step")


def test_two_solid_assembly_round_trips_occurrences(tmp_path: Path) -> None:
    native = native_assembly()
    result = _read(_write(native, tmp_path / "assembly.step"))
    geometry = result.model.geometry
    assert geometry is not None and native.geometry is not None
    assert result.model.topology.num_solids == 2
    assert [occurrence.path for occurrence in geometry.occurrences] == [
        ("base",),
        ("post",),
    ]
    placed = geometry.occurrences[1]
    np.testing.assert_array_equal(np.asarray(placed.rotation), np.asarray(QUARTER_TURN))
    assert placed.translation == (3.0, 0.0, 0.0)
    np.testing.assert_allclose(
        solid_volumes(result.model),
        [EXACT_VOLUMES["box"], EXACT_VOLUMES["cylinder"]],
        rtol=1e-10,
    )


def test_repeated_nested_component_keeps_one_definition(tmp_path: Path) -> None:
    native = native_repeated_assembly()
    result = _read(_write(native, tmp_path / "repeated.step"))
    assert result.model.geometry is not None and native.geometry is not None
    assert result.model.topology.num_solids == 1
    assert result.model.geometry.geometry_id == native.geometry.geometry_id
    assert result.model.geometry.occurrences == native.geometry.occurrences
    assert result.model.geometry.occurrences[1].path == ("group", "moved")
    assert result.model.geometry.occurrences[1].translation == (4.0, 2.0, 1.0)
    np.testing.assert_array_equal(
        result.model.geometry.occurrences[1].rotation, QUARTER_TURN
    )


def test_independent_repeated_instances_do_not_infer_a_container_from_paths(
    tmp_path: Path,
) -> None:
    native = native_repeated_assembly(declared=False)
    result = _read(_write(native, tmp_path / "independent.step"))
    assert result.model.geometry is not None and native.geometry is not None
    assert result.model.geometry.geometry_id == native.geometry.geometry_id
    assert result.model.geometry.occurrences == native.geometry.occurrences
    assert result.model.geometry.assembly_containers == ()


def test_assembly_root_without_direct_shape_link_keeps_occurrences(
    tmp_path: Path,
) -> None:
    native = native_repeated_assembly()
    data = _write(native, tmp_path / "linked.step").decode()
    product = re.search(
        r"#(\d+)=PRODUCT\('phydrax:container:\[\"model\"\]','model',", data
    )
    assert product is not None
    formation = re.search(
        rf"#(\d+)=PRODUCT_DEFINITION_FORMATION\('','',#{product.group(1)}\);", data
    )
    assert formation is not None
    definition = re.search(
        rf"#(\d+)=PRODUCT_DEFINITION\('design','',#{formation.group(1)},#\d+\);", data
    )
    assert definition is not None
    shape = re.search(
        rf"#(\d+)=PRODUCT_DEFINITION_SHAPE\('','',#{definition.group(1)}\);", data
    )
    assert shape is not None
    data = re.sub(
        rf"#\d+=SHAPE_DEFINITION_REPRESENTATION\(#{shape.group(1)},#\d+\);\n",
        "",
        data,
        count=1,
    )
    restored = _read(data.encode()).model
    assert restored.geometry is not None and native.geometry is not None
    assert restored.geometry.occurrences == native.geometry.occurrences
    assert restored.geometry.geometry_id == native.geometry.geometry_id


def test_cyclic_product_structure_is_refused_even_when_entity_graph_is_acyclic(
    tmp_path: Path,
) -> None:
    data = _write(native_repeated_assembly(), tmp_path / "assembly.step").decode()
    usage = re.search(
        r"NEXT_ASSEMBLY_USAGE_OCCURRENCE\('[^']*','[^']*','',#(\d+),#(\d+),\$\)", data
    )
    assert usage is not None
    reverse = (
        f"#90001=NEXT_ASSEMBLY_USAGE_OCCURRENCE('cycle','cycle','',"
        f"#{usage.group(2)},#{usage.group(1)},$);\n"
    )
    data = data.replace("ENDSEC;\nEND-ISO", reverse + "ENDSEC;\nEND-ISO", 1)
    refusal = _refusal(data.encode())
    assert refusal.reason == "cyclic-reference"
    assert any("PRODUCT_DEFINITION" in label for label in refusal.chain)


def test_units_are_converted_exactly_into_the_target_contract(tmp_path: Path) -> None:
    millimetre = phx._physical.SpatialCoordinateContract(MILLIMETER)
    native = phx.geometry.brep.brep_box(
        (0.0, 0.0, 0.0), (1.0, 2.0, 3.0), coordinate_contract=millimetre
    )
    data = _write(native, tmp_path / "mm.step")
    same = _read(data, interchange.CadImportPolicy(millimetre, LIMITS))
    assert same.model.geometry is not None and native.geometry is not None
    assert same.model.geometry.geometry_id == native.geometry.geometry_id
    metres = _read(data)
    assert metres.source_length_unit_meters == pytest.approx(1e-3, rel=0, abs=0)
    assert metres.model.geometry is not None
    np.testing.assert_array_equal(
        np.max(np.asarray(metres.model.geometry.vertex_points), axis=0),
        [0.001, 0.002, 0.003],
    )
    assert solid_volumes(metres.model)[0] == pytest.approx(6e-9, rel=1e-10)


@pytest.mark.parametrize("schema", ["AP203", "AP214", "AP242"])
def test_native_rational_spline_round_trip_without_optional_kernel(
    schema: Literal["AP203", "AP214", "AP242"], tmp_path: Path
) -> None:
    native = native_rational_spline_face()
    path = tmp_path / "spline.step"
    interchange.write_step(native, path, schema=schema)
    result = _read(path.read_bytes())
    assert result.schema == schema
    assert result.coverage.pcurves_honored == 4
    assert result.coverage.pcurves_fitted == 0
    assert result.model.geometry is not None and native.geometry is not None
    assert result.model.geometry.geometry_id == native.geometry.geometry_id
    patch = result.model.patches[0]
    expected = native.patches[0]
    for field in ("control_points", "weights", "u_knots", "v_knots"):
        np.testing.assert_array_equal(getattr(patch, field), getattr(expected, field))
    area = float(
        phx.geometry.brep.prepare_brep_query(result.model).measures.face_areas[0]
    )
    assert area == pytest.approx(np.pi, rel=1e-9)


def test_trimmed_long_arc_preserves_range_pcurve_and_area(tmp_path: Path) -> None:
    native = native_sector_face()
    result = _read(_write(native, tmp_path / "sector.step"))
    assert result.model.geometry is not None
    assert result.model.geometry.edge_ranges[0, 1] == 1.5 * np.pi
    assert result.coverage.pcurves_honored == 3
    area = float(
        phx.geometry.brep.prepare_brep_query(result.model).measures.face_areas[0]
    )
    assert area == pytest.approx(0.75 * np.pi, rel=1e-9)


def test_cartesian_trim_selectors_retain_the_long_arc(tmp_path: Path) -> None:
    data = _write(native_sector_face(), tmp_path / "sector.step").decode()
    vertices = re.findall(r"VERTEX_POINT\('',(#\d+)\)", data)
    match = re.search(
        r"(TRIMMED_CURVE\('',#\d+,)\(PARAMETER_VALUE\([^)]+\)\),"
        r"\(PARAMETER_VALUE\([^)]+\)\),\.T\.,\.PARAMETER\.\)",
        data,
    )
    assert match is not None
    data = (
        data[: match.start()]
        + (f"{match.group(1)}({vertices[0]}),({vertices[1]}),.T.,.CARTESIAN.)")
        + data[match.end() :]
    )
    restored = _read(data.encode()).model
    assert restored.geometry is not None
    np.testing.assert_array_equal(restored.geometry.edge_ranges[0], [0.0, 1.5 * np.pi])
    area = float(phx.geometry.brep.prepare_brep_query(restored).measures.face_areas[0])
    assert area == pytest.approx(0.75 * np.pi, rel=1e-9)


def test_reversed_trim_preserves_oriented_edge_incidence(tmp_path: Path) -> None:
    data = _write(native_sector_face(), tmp_path / "sector.step").decode()
    match = re.search(
        r"(#\d+=TRIMMED_CURVE\('',#\d+,)\(PARAMETER_VALUE\(([^)]+)\)\),"
        r"\(PARAMETER_VALUE\(([^)]+)\)\),\.T\.,\.PARAMETER\.\)",
        data,
    )
    assert match is not None
    identifier = match.group(1).split("=")[0]
    data = (
        data[: match.start()]
        + (
            f"{match.group(1)}(PARAMETER_VALUE({match.group(3)})),"
            f"(PARAMETER_VALUE({match.group(2)})),.F.,.PARAMETER.)"
        )
        + data[match.end() :]
    )
    # EDGE_CURVE orientation is relative to the trimmed carrier, not its basis.
    carrier = re.search(rf"#(\d+)=SURFACE_CURVE\('',{identifier},", data)
    assert carrier is not None
    data = re.sub(
        rf"(EDGE_CURVE\('',#\d+,#\d+,#{carrier.group(1)},)\.T\.",
        r"\1.F.",
        data,
        count=1,
    )
    result = _read(data.encode())
    assert result.model.geometry is not None
    assert result.model.geometry.coedge_senses == (1, 1, 1)
    np.testing.assert_array_equal(
        result.model.geometry.edge_ranges[0], [0.0, 1.5 * np.pi]
    )


def test_trim_endpoint_conflicting_with_vertex_is_refused(tmp_path: Path) -> None:
    data = _write(native_sector_face(), tmp_path / "sector.step")
    changed = data.replace(b"PARAMETER_VALUE(0.0)", b"PARAMETER_VALUE(0.25)", 1)
    assert changed != data
    data = changed
    refusal = _refusal(data)
    assert refusal.reason == "inconsistent-geometry"
    assert "EDGE_CURVE" in refusal.entity


def test_hostile_spline_multiplicity_is_refused_before_expansion(tmp_path: Path) -> None:
    data = _write(native_rational_spline_face(), tmp_path / "spline.step").decode()
    data = data.replace(
        "B_SPLINE_CURVE_WITH_KNOTS((3,3)", "B_SPLINE_CURVE_WITH_KNOTS((1000000000,3)", 1
    )
    refusal = _refusal(data.encode())
    assert refusal.reason == "malformed"
    assert "B_SPLINE" in refusal.entity


def test_nested_assembly_occurrences_retain_shared_solid_and_placement(
    tmp_path: Path,
) -> None:
    source = native_assembly()
    assert source.geometry is not None
    geometry = source.geometry
    repeated = phx.geometry.brep.BRepGeometry(
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
        vertex_roots=geometry.vertex_roots,
        edge_endpoint_roots=geometry.edge_endpoint_roots,
        coedge_endpoint_roots=geometry.coedge_endpoint_roots,
        occurrences=(
            phx.geometry.brep.BRepOccurrence(("group", "base"), 0),
            phx.geometry.brep.BRepOccurrence(
                ("group", "post"),
                1,
                np.asarray(QUARTER_TURN),
                np.asarray((3.0, 0.0, 0.0)),
            ),
            phx.geometry.brep.BRepOccurrence(
                ("copy",), 1, np.asarray(QUARTER_TURN), np.asarray((7.0, 2.0, 0.0))
            ),
        ),
        assembly_containers=(
            BRepAssemblyContainer(("model",), (("copy",),), (("group",),)),
            BRepAssemblyContainer(("group",), (("group", "base"), ("group", "post"))),
        ),
    )
    model = phx.geometry.brep.assemble_brep_model(
        repeated,
        source.patches,
        source.parameter_bounds,
        source.orientation,
        source.physical_tags,
        coordinate_contract=source.coordinate_contract,
        source_id=source.source_id,
        source_format="native",
        source_digest=source.source_digest,
        import_policy_id=source.report.import_policy_id,
    )
    restored = _read(_write(model, tmp_path / "nested.step")).model
    assert restored.geometry is not None
    assert restored.geometry.occurrences == repeated.occurrences
    assert restored.topology.num_solids == 2
    policy = interchange.CadImportPolicy(CONTRACT, LIMITS, maximum_occurrences=2)
    assert _refusal(_write(model, tmp_path / "limited.step"), policy).reason == "limit"


# ------------------------------------------------------------- refusals


@pytest.fixture(scope="module")
def cylinder_step(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("step") / "cylinder.step"
    return _write(native_solid("cylinder"), path).decode()


def test_truncated_file_is_refused(cylinder_step: str) -> None:
    refusal = _refusal(cylinder_step[: len(cylinder_step) // 2].encode())
    assert refusal.reason == "malformed"


def test_dangling_reference_is_refused_with_chain(cylinder_step: str) -> None:
    text = cylinder_step.replace("VERTEX_POINT('',#", "VERTEX_POINT('',#9999", 1)
    refusal = _refusal(text.encode())
    assert refusal.reason == "dangling-reference"
    assert refusal.chain and "VERTEX_POINT" in refusal.chain[-1]


def test_cyclic_reference_is_refused(cylinder_step: str) -> None:
    extra = "#90001=CARTESIAN_POINT('',(#90002));\n#90002=CARTESIAN_POINT('',(#90001));\nENDSEC;"
    text = cylinder_step.replace("ENDSEC;\nEND-ISO", extra + "\nEND-ISO", 1)
    assert _refusal(text.encode()).reason == "cyclic-reference"


def test_unsupported_entity_is_refused_with_dependency_chain(cylinder_step: str) -> None:
    text = cylinder_step.replace("CYLINDRICAL_SURFACE(", "SURFACE_REPLICA(", 1)
    refusal = _refusal(text.encode())
    assert refusal.reason == "unsupported-entity"
    assert "SURFACE_REPLICA" in refusal.entity
    assert any("MANIFOLD_SOLID_BREP" in label for label in refusal.chain)
    assert any("ADVANCED_FACE" in label for label in refusal.chain)


def test_missing_length_unit_is_refused(cylinder_step: str) -> None:
    text = cylinder_step.replace(
        "( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.) )",
        "( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() )",
    )
    assert _refusal(text.encode()).reason == "units"


def test_contradictory_length_units_are_refused(cylinder_step: str) -> None:
    text = cylinder_step.replace(
        "GLOBAL_UNIT_ASSIGNED_CONTEXT((", "GLOBAL_UNIT_ASSIGNED_CONTEXT((#90001,", 1
    )
    text = text.replace(
        "ENDSEC;\nEND-ISO",
        "#90001=( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );\nENDSEC;\nEND-ISO",
        1,
    )
    assert _refusal(text.encode()).reason == "units"


def test_unknown_schema_is_refused(cylinder_step: str) -> None:
    text = cylinder_step.replace("AUTOMOTIVE_DESIGN", "SHIP_STRUCTURES", 1)
    assert _refusal(text.encode()).reason == "unsupported-entity"


@pytest.mark.parametrize(
    ("limits", "reason"),
    [
        pytest.param(
            ResourceLimits(1024, 64, 200_000, 4_000_000, 0), "limit", id="bytes"
        ),
        pytest.param(
            ResourceLimits(1 << 24, 64, 50, 4_000_000, 0), "limit", id="instances"
        ),
        pytest.param(
            ResourceLimits(1 << 24, 64, 200_000, 100, 0), "limit", id="parameters"
        ),
        pytest.param(
            ResourceLimits(1 << 24, 4, 200_000, 4_000_000, 0), "limit", id="depth"
        ),
    ],
)
def test_over_limit_files_are_refused(
    cylinder_step: str, limits: ResourceLimits, reason: str
) -> None:
    policy = interchange.CadImportPolicy(CONTRACT, limits)
    refusal = _refusal(cylinder_step.encode(), policy)
    assert refusal.reason == reason


def test_authoritative_trim_mismatch_is_not_silently_dropped(tmp_path: Path) -> None:
    source = native_solid("box")
    changed = TrimDomain(PolygonTrimLoop([[0, 0], [0.5, 0], [0.5, 1], [0, 1]]))
    model = phx.geometry.brep.BRepModel(
        patches=source.patches,
        parameter_bounds=source.parameter_bounds,
        orientation=source.orientation,
        trim_domains=(changed, *source.trim_domains[1:]),
        topology=source.topology,
        coordinate_contract=source.coordinate_contract,
        mesh_vertices=source.mesh_vertices,
        mesh_faces=source.mesh_faces,
        triangle_face_ids=source.triangle_face_ids,
        triangle_parameters=source.triangle_parameters,
        tessellation_deviation_bounds=source.tessellation_deviation_bounds,
        tessellation_normal_bounds=source.tessellation_normal_bounds,
        mesh_vertex_source_dimensions=source.mesh_vertex_source_dimensions,
        mesh_vertex_source_indices=source.mesh_vertex_source_indices,
        mesh_vertex_parameters=source.mesh_vertex_parameters,
        mesh_chart_restriction_vertices=source.mesh_chart_restriction_vertices,
        mesh_chart_restriction_edges=source.mesh_chart_restriction_edges,
        mesh_chart_restriction_endpoint_parameters=(
            source.mesh_chart_restriction_endpoint_parameters
        ),
        mesh_chart_restriction_parameters=source.mesh_chart_restriction_parameters,
        coedge_deviation_bounds=source.coedge_deviation_bounds,
        triangle_occurrence_ids=source.triangle_occurrence_ids,
        vertex_occurrence_ids=source.vertex_occurrence_ids,
        physical_tags=source.physical_tags,
        report=source.report,
        geometry=source.geometry,
    )
    destination = tmp_path / "mismatched.step"
    with pytest.raises(interchange.CadInterchangeError) as caught:
        interchange.write_step(model, destination)
    assert caught.value.refusal.reason == "inexact-export"
    assert "face:0" in (*caught.value.refusal.chain, caught.value.refusal.entity)
    assert not destination.exists()


def test_model_without_exact_topology_is_not_exported(tmp_path: Path) -> None:
    model = native_solid("box")
    stripped = phx.geometry.brep.BRepModel(
        patches=model.patches,
        parameter_bounds=model.parameter_bounds,
        orientation=model.orientation,
        trim_domains=model.trim_domains,
        topology=model.topology,
        coordinate_contract=model.coordinate_contract,
        mesh_vertices=model.mesh_vertices,
        mesh_faces=model.mesh_faces,
        triangle_face_ids=model.triangle_face_ids,
        triangle_parameters=model.triangle_parameters,
        physical_tags=model.physical_tags,
        report=model.report,
    )
    with pytest.raises(interchange.CadInterchangeError) as caught:
        interchange.write_step(stripped, tmp_path / "x.step")
    assert caught.value.refusal.reason == "inexact-export"
    assert not (tmp_path / "x.step").exists()


# ---------------------------------------------------- OCCT comparison


@pytest.fixture(scope="module")
def occt() -> object:
    return pytest.importorskip("OCP")


def _occt_shapes(name: str) -> object:
    from OCP.BRepPrimAPI import (
        BRepPrimAPI_MakeBox,
        BRepPrimAPI_MakeCone,
        BRepPrimAPI_MakeCylinder,
        BRepPrimAPI_MakeSphere,
        BRepPrimAPI_MakeTorus,
    )

    match name:
        case "box":
            return BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Shape()
        case "cylinder":
            return BRepPrimAPI_MakeCylinder(1.0, 2.0).Shape()
        case "sphere":
            return BRepPrimAPI_MakeSphere(1.5).Shape()
        case "torus":
            return BRepPrimAPI_MakeTorus(3.0, 1.0).Shape()
        case "cone":
            return BRepPrimAPI_MakeCone(1.0, 0.0, 2.0).Shape()
        case "plate":
            from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
            from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

            plate = BRepPrimAPI_MakeBox(4.0, 3.0, 0.5).Shape()
            bore = BRepPrimAPI_MakeCylinder(
                gp_Ax2(gp_Pnt(2.0, 1.5, -0.1), gp_Dir(0, 0, 1)), 0.5, 0.7
            ).Shape()
            return BRepAlgoAPI_Cut(plate, bore).Shape()
        case "ring":
            from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut

            return BRepAlgoAPI_Cut(
                BRepPrimAPI_MakeCylinder(3.0, 1.0).Shape(),
                BRepPrimAPI_MakeCylinder(2.0, 1.0).Shape(),
            ).Shape()
        case "assembly":
            from OCP.BRep import BRep_Builder
            from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
            from OCP.TopLoc import TopLoc_Location
            from OCP.TopoDS import TopoDS_Compound

            trsf = gp_Trsf()
            trsf.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), np.pi / 2)
            shift = gp_Trsf()
            shift.SetTranslation(gp_Vec(3.0, 0.0, 0.0))
            compound = TopoDS_Compound()
            builder = BRep_Builder()
            builder.MakeCompound(compound)
            builder.Add(compound, BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Shape())
            cylinder = BRepPrimAPI_MakeCylinder(1.0, 2.0).Shape()
            builder.Add(compound, cylinder.Moved(TopLoc_Location(shift.Multiplied(trsf))))
            return compound
        case "spline":
            from OCP.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert

            return BRepBuilderAPI_NurbsConvert(
                BRepPrimAPI_MakeCylinder(1.0, 2.0).Shape(), True
            ).Shape()
        case _:
            raise ValueError(name)


def _occt_write(
    shape: object, path: Path, *, pcurves: bool = True, assembly: bool = False
) -> bytes:
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Interface import Interface_Static
    from OCP.STEPControl import (
        STEPControl_AsIs,
        STEPControl_Controller,
        STEPControl_Writer,
    )

    STEPControl_Controller.Init_s()
    assert Interface_Static.SetIVal_s("write.surfacecurve.mode", 1 if pcurves else 0)
    assert Interface_Static.SetIVal_s("write.step.assembly", 1 if assembly else 0)
    writer = STEPControl_Writer()
    assert writer.Transfer(shape, STEPControl_AsIs) == IFSelect_RetDone
    assert writer.Write(str(path)) == IFSelect_RetDone
    Interface_Static.SetIVal_s("write.surfacecurve.mode", 1)
    Interface_Static.SetIVal_s("write.step.assembly", 0)
    return path.read_bytes()


def _occt_read(path: Path) -> object:
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    assert reader.ReadFile(str(path)) == IFSelect_RetDone
    assert reader.TransferRoots() >= 1
    return reader.OneShape()


def _occt_facts(shape: object) -> tuple[int, int, float, bool]:
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    def count(kind: object) -> int:
        explorer = TopExp_Explorer(shape, kind)
        total = 0
        while explorer.More():
            total += 1
            explorer.Next()
        return total

    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties)
    return (
        count(TopAbs_SOLID),
        count(TopAbs_FACE),
        properties.Mass(),
        BRepCheck_Analyzer(shape).IsValid(),
    )


def _occt_edge_vertex_counts(shape: object) -> tuple[int, int]:
    from OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
    from OCP.TopExp import TopExp

    edges = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()
    vertices = IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher()
    TopExp.MapShapes_s(shape, TopAbs_EDGE, edges)
    TopExp.MapShapes_s(shape, TopAbs_VERTEX, vertices)
    return edges.Extent(), vertices.Extent()


@pytest.mark.parametrize("name", [*SOLIDS, "assembly"])
def test_occt_written_step_matches_native_reading(
    occt: object, name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / f"{name}.step"
    data = _occt_write(_occt_shapes(name), path, assembly=name == "assembly")
    shape = _occt_read(path)
    solids, faces, volume, _ = _occt_facts(shape)
    edges, vertices = _occt_edge_vertex_counts(shape)
    original_import = builtins.__import__

    def native_only_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "OCP" or name.startswith("OCP."):
            raise AssertionError("Native STEP decoding must not import OCP algorithms.")
        return original_import(name, globals, locals, fromlist, level)

    with monkeypatch.context() as guard:
        guard.setattr(builtins, "__import__", native_only_import)
        result = _read(data, OCCT_POLICY)
    assert result.model.topology.num_solids == solids
    assert result.model.topology.num_faces == faces
    geometry = result.model.geometry
    assert geometry is not None
    assert (geometry.topology().num_edges, geometry.topology().num_vertices) == (
        edges,
        vertices,
    )
    placed = solid_volumes(result.model)[
        [occurrence.solid for occurrence in geometry.occurrences]
    ]
    assert float(np.sum(placed)) == pytest.approx(volume, rel=1e-9)
    assert result.schema == "AP214"


@pytest.mark.parametrize("name", [*SOLIDS, "assembly"])
def test_native_step_is_read_by_occt(occt: object, name: str, tmp_path: Path) -> None:
    model = native_assembly() if name == "assembly" else native_solid(name)
    path = tmp_path / f"{name}.step"
    _write(model, path)
    solids, faces, volume, valid = _occt_facts(_occt_read(path))
    assert valid
    assert (solids, faces) == (model.topology.num_solids, model.topology.num_faces)
    assert volume == pytest.approx(1e9 * float(np.sum(solid_volumes(model))), rel=1e-9)


def test_occt_step_without_pcurves_recovers_exact_pcurves(
    occt: object, tmp_path: Path
) -> None:
    data = _occt_write(_occt_shapes("cylinder"), tmp_path / "bare.step", pcurves=False)
    result = _read(data, OCCT_POLICY)
    assert result.coverage.pcurves_honored == 0
    assert result.coverage.pcurves_exact > 0 and result.coverage.pcurves_fitted == 0
    assert solid_volumes(result.model)[0] == pytest.approx(
        EXACT_VOLUMES["cylinder"], rel=1e-9
    )


def _rational_spline_face() -> object:
    """A clamped rational biquadratic patch (a quarter-cylinder strip, weights 1/sqrt 2)."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.collections import Array1_double, Array1_int, Array2_double, Array2_gp_Pnt
    from OCP.Geom import Geom_BSplineSurface
    from OCP.gp import gp_Pnt

    poles = Array2_gp_Pnt(1, 3, 1, 2)
    weights = Array2_double(1, 3, 1, 2)
    for row, (x, y, w) in enumerate(
        ((1.0, 0.0, 1.0), (1.0, 1.0, 0.5**0.5), (0.0, 1.0, 1.0)), 1
    ):
        for column, z in enumerate((0.0, 2.0), 1):
            poles.SetValue(row, column, gp_Pnt(x, y, z))
            weights.SetValue(row, column, w)
    knots = Array1_double(1, 2)
    knots.SetValue(1, 0.0)
    knots.SetValue(2, 1.0)
    u_mults, v_mults = Array1_int(1, 2), Array1_int(1, 2)
    for index in (1, 2):
        u_mults.SetValue(index, 3)
        v_mults.SetValue(index, 2)
    surface = Geom_BSplineSurface(poles, weights, knots, knots, u_mults, v_mults, 2, 1)
    return BRepBuilderAPI_MakeFace(surface, 1e-9).Face()


def test_rational_spline_face_round_trips_exactly(occt: object, tmp_path: Path) -> None:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    path = tmp_path / "spline.step"
    data = _occt_write(_rational_spline_face(), path)
    properties = GProp_GProps()
    BRepGProp.SurfaceProperties_s(_occt_read(path), properties)
    first = _read(data, OCCT_POLICY).model
    (patch,) = first.patches
    assert isinstance(patch, phx.geometry.brep.BSplineSurfacePatch)
    assert np.any(np.asarray(patch.weights) != 1.0)
    second = _read(_write(first, tmp_path / "again.step"), OCCT_POLICY).model
    assert first.geometry is not None and second.geometry is not None
    assert second.geometry.geometry_id == first.geometry.geometry_id
    (again,) = second.patches
    assert isinstance(again, phx.geometry.brep.BSplineSurfacePatch)
    for field in ("control_points", "weights", "u_knots", "v_knots"):
        np.testing.assert_array_equal(
            np.asarray(getattr(patch, field)), np.asarray(getattr(again, field))
        )
    area = float(phx.geometry.brep.prepare_brep_query(first).measures.face_areas[0])
    assert area == pytest.approx(properties.Mass(), rel=1e-9)
    assert area == pytest.approx(np.pi / 2 * 2.0, rel=1e-9)


def test_periodic_spline_solid_retains_exact_source_and_topology(
    occt: object, tmp_path: Path
) -> None:
    data = _occt_write(_occt_shapes("spline"), tmp_path / "periodic.step")
    imported = _read(data, OCCT_POLICY)
    assert imported.report.valid
    assert imported.model.topology.num_solids == 1
    assert imported.model.topology.num_faces == 3
    assert imported.coverage.pcurves_fitted == 0
    assert any(
        isinstance(patch, phx.geometry.BSplineSurfacePatch)
        for patch in imported.model.patches
    )
    assert solid_volumes(imported.model)[0] == pytest.approx(2.0 * np.pi, rel=1e-8)
    restored = interchange.load_brep_archive(
        interchange.save_brep_archive(imported.model, tmp_path / "periodic.phx").path
    )
    assert restored.model_id == imported.model.model_id
    assert restored.source_id == imported.model.source_id
    assert solid_volumes(restored)[0] == solid_volumes(imported.model)[0]


@pytest.mark.parametrize("millimeters", (False, True))
def test_independent_offset_surface_source_retains_tree_without_fitted_pcurves(
    millimeters: bool,
    tmp_path: Path,
) -> None:
    """Hand-authored Part21 source, independent of the native writer.

    The offset entity follows the STEP geometry schema: basis surface,
    signed length distance, and LOGICAL self-intersection declaration.
    """
    data = b"""ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('Offset surface source contract'),'2;1');
FILE_NAME('offset-plane.step','2026-10-04T00:00:00',('PHYDRA'),('PHYDRA'),'independent fixture','independent fixture','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.5));
#2=DIRECTION('',(0.,0.,1.));
#3=DIRECTION('',(1.,0.,0.));
#4=AXIS2_PLACEMENT_3D('',#1,#2,#3);
#5=PLANE('',#4);
#6=OFFSET_SURFACE('',#5,0.25,.U.);
#7=CARTESIAN_POINT('',(0.,0.,0.75));
#8=CARTESIAN_POINT('',(2.,0.,0.75));
#9=CARTESIAN_POINT('',(2.,1.,0.75));
#10=CARTESIAN_POINT('',(0.,1.,0.75));
#11=VERTEX_POINT('',#7);
#12=VERTEX_POINT('',#8);
#13=VERTEX_POINT('',#9);
#14=VERTEX_POINT('',#10);
#15=DIRECTION('',(1.,0.,0.));
#16=DIRECTION('',(0.,1.,0.));
#17=DIRECTION('',(-1.,0.,0.));
#18=DIRECTION('',(0.,-1.,0.));
#19=VECTOR('',#15,2.);
#20=VECTOR('',#16,1.);
#21=VECTOR('',#17,2.);
#22=VECTOR('',#18,1.);
#23=LINE('',#7,#19);
#24=LINE('',#8,#20);
#25=LINE('',#9,#21);
#26=LINE('',#10,#22);
#27=EDGE_CURVE('',#11,#12,#23,.T.);
#28=EDGE_CURVE('',#12,#13,#24,.T.);
#29=EDGE_CURVE('',#13,#14,#25,.T.);
#30=EDGE_CURVE('',#14,#11,#26,.T.);
#31=ORIENTED_EDGE('',*,*,#27,.T.);
#32=ORIENTED_EDGE('',*,*,#28,.T.);
#33=ORIENTED_EDGE('',*,*,#29,.T.);
#34=ORIENTED_EDGE('',*,*,#30,.T.);
#35=EDGE_LOOP('',(#31,#32,#33,#34));
#36=FACE_OUTER_BOUND('',#35,.T.);
#37=ADVANCED_FACE('',(#36),#6,.T.);
#38=OPEN_SHELL('',(#37));
#39=SHELL_BASED_SURFACE_MODEL('',(#38));
#40=MANIFOLD_SURFACE_SHAPE_REPRESENTATION('',(#39),#44);
#41=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.));
#42=(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.));
#43=(NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT());
#44=(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNIT_ASSIGNED_CONTEXT((#41,#42,#43)) REPRESENTATION_CONTEXT('',''));
ENDSEC;
END-ISO-10303-21;
"""
    if millimeters:
        data = data.replace(b"SI_UNIT($,.METRE.)", b"SI_UNIT(.MILLI.,.METRE.)")
    factor = 1.0e-3 if millimeters else 1.0
    imported = _read(data)
    assert imported.report.valid
    assert imported.coverage.pcurves_fitted == 0
    (patch,) = imported.model.patches
    assert isinstance(patch, phx.geometry.OffsetSurface)
    assert isinstance(patch.base, phx.geometry.PlanePatch)
    assert float(patch.distance) == 0.25 * factor
    np.testing.assert_allclose(
        patch.evaluate(jnp.asarray([[0.2, 0.7], [1.7, 0.2]], dtype=jnp.float64) * factor),
        np.asarray([[0.2, 0.7, 0.75], [1.7, 0.2, 0.75]]) * factor,
        rtol=0.0,
        atol=1e-14,
    )
    assert imported.model.topology.num_faces == 1
    assert imported.model.topology.num_edges == 4
    area = phx.geometry.prepare_brep_query(imported.model).measures.face_areas[0]
    assert float(area) == pytest.approx(2.0 * factor**2, rel=1e-10)
    receipt = interchange.save_brep_archive(
        imported.model, tmp_path / "independent-offset.phx"
    )
    restored = interchange.load_brep_archive(receipt.path)
    assert restored.model_id == imported.model.model_id == receipt.model_id
    assert restored.source_id == imported.model.source_id
    assert isinstance(restored.patches[0], phx.geometry.OffsetSurface)
    np.testing.assert_array_equal(
        restored.patches[0].evaluate(jnp.asarray([0.3, 0.4], dtype=jnp.float64) * factor),
        patch.evaluate(jnp.asarray([0.3, 0.4], dtype=jnp.float64) * factor),
    )
