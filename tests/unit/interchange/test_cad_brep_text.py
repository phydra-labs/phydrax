#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native OCCT BRep text interchange: exact round trips, OCCT references, refusals.

OCCT (via OCP) is an independent reference only: it writes the external files
the native reader must decode and reads the files the native writer emits.
Volumes are compared with the native exact-geometry measure and OCCT's
``BRepGProp``; tolerances reflect quadrature error, not interchange error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax._external_resource import ResourceLimits
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._atlas import TrimDomain
from phydrax.geometry.brep import (
    assemble_brep_model,
    brep_box,
    brep_cylinder,
    brep_extrusion,
    brep_revolution,
    brep_sphere,
    brep_torus,
    BRepGeometry,
    BRepModel,
    BRepOccurrence,
    BRepTessellationPolicy,
    HyperbolaCurve,
    OffsetCurve,
    ParabolaCurve,
    PlanarProfile,
    prepare_brep_query,
    ProfileLoop,
    ProfilePlane,
)
from phydrax.geometry.brep._intersection_curve import IntersectionCurve
from phydrax.geometry.brep._model import BRepAssemblyContainer
from phydrax.geometry.brep._patches import (
    BSplineCurve,
    BSplineSurfacePatch,
    CylinderPatch,
    LineCurve,
    OffsetSurface,
    PlanePatch,
)
from phydrax.interchange import (
    AdapterStatus,
    CadCurveFitPolicy,
    CadExportPolicy,
    CadImportPolicy,
    CadInterchangeError,
    decode_brep_text_bytes,
    load_brep_archive,
    read_brep_text,
    save_brep_archive,
    write_brep_text,
)
from phydrax.interchange._cad import CadStage, recover_pcurve, StagedCoedge
from phydrax.interchange._cad_carriers import RigidPlacement
from phydrax.units import METER, MILLIMETER, UnitDefinition
from tests._support.cad_models import native_intersection_cap, native_offset_plane


_CONTRACT = SpatialCoordinateContract(METER)
_LIMITS = ResourceLimits(
    max_bytes=1 << 22,
    max_depth=32,
    max_nodes=20_000,
    max_attributes=2_000_000,
    max_losses=8,
)
_POLICY = CadImportPolicy(_CONTRACT, _LIMITS)
_OCCT_SPLINE_POLICY = CadImportPolicy(
    _CONTRACT,
    _LIMITS,
    tessellation=BRepTessellationPolicy(realize=False),
    relative_geometric_tolerance=2.0e-6,
)


_T = TypeVar("_T")


def _require_type(value: object, expected: type[_T], /) -> _T:
    if not isinstance(value, expected):
        raise TypeError(f"Expected {expected.__name__}, got {type(value).__name__}.")
    return value


def _geometry(model: BRepModel, /) -> BRepGeometry:
    geometry = model.geometry
    if geometry is None:
        raise ValueError("This CAD fixture requires exact native geometry.")
    return geometry


def _plate() -> BRepModel:
    profile = PlanarProfile(
        ProfilePlane((-2.0, -2.0, 0.0)),
        ProfileLoop.polygon(((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0))),
        (ProfileLoop.circle((2.0, 2.0), 0.5),),
    )
    return brep_extrusion(profile, (0.0, 0.0, 0.5), coordinate_contract=_CONTRACT)


def _ring() -> BRepModel:
    profile = PlanarProfile(
        ProfilePlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ProfileLoop.polygon(((1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0))),
    )
    return brep_revolution(profile, (0.0, 0.0), (0.0, 1.0), coordinate_contract=_CONTRACT)


_NATIVE = {
    "box": lambda: brep_box(
        (0.0, 0.0, 0.0), (1.0, 2.0, 3.0), coordinate_contract=_CONTRACT
    ),
    "cylinder": lambda: brep_cylinder(0.5, 2.0, coordinate_contract=_CONTRACT),
    "sphere": lambda: brep_sphere(0.75, coordinate_contract=_CONTRACT),
    "torus": lambda: brep_torus(2.0, 0.5, coordinate_contract=_CONTRACT),
    "plate-with-hole": _plate,
    "revolved-ring": _ring,
}
# Exact volumes of the native solids and of the OCCT-built references.
_VOLUMES = {
    "box": 6.0,
    "cylinder": np.pi * 0.25 * 2.0,
    "sphere": 4.0 / 3.0 * np.pi * 0.75**3,
    "torus": 2.0 * np.pi**2 * 2.0 * 0.25,
    "plate-with-hole": 8.0 - np.pi * 0.25 * 0.5,
    "revolved-ring": np.pi * 3.0,
}


def _volume(model: BRepModel) -> float:
    return float(prepare_brep_query(model).measures.solid_volumes[0])


def _counts(model: BRepModel) -> tuple[int, int, int, int]:
    topology = model.topology
    return (
        topology.num_solids,
        topology.num_faces,
        topology.num_edges - sum(_geometry(model).degenerate_edges),
        topology.num_vertices,
    )


@pytest.mark.parametrize("name", sorted(_NATIVE))
def test_native_models_round_trip_with_exact_identity(name: str, tmp_path: Path) -> None:
    model = _NATIVE[name]()
    write_brep_text(model, tmp_path / "model.brep")
    result = read_brep_text(
        tmp_path / "model.brep", _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert _geometry(result.model).geometry_id == _geometry(model).geometry_id
    assert np.array_equal(result.model.orientation, model.orientation)
    assert result.coverage.pcurves_honored == len(_geometry(model).pcurves)
    assert result.coverage.pcurves_fitted == 0
    assert result.report.valid
    assert _volume(result.model) == pytest.approx(_VOLUMES[name], rel=1.0e-6)


def test_writer_output_is_deterministic(tmp_path: Path) -> None:
    model = _NATIVE["torus"]()
    first = write_brep_text(model, tmp_path / "a.brep")
    second = write_brep_text(model, tmp_path / "b.brep")
    assert (tmp_path / "a.brep").read_bytes() == (tmp_path / "b.brep").read_bytes()
    assert first.receipt.content_sha256 == second.receipt.content_sha256


def test_versions_share_one_model(tmp_path: Path) -> None:
    model = _NATIVE["cylinder"]()
    identities = set()
    for version in ("V1", "V2", "V3"):
        path = tmp_path / f"{version}.brep"
        write_brep_text(model, path, version=version)
        result = read_brep_text(
            path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
        )
        assert result.schema == f"cascade-topology-{version}"
        identities.add(_geometry(result.model).geometry_id)
    assert identities == {_geometry(model).geometry_id}


def test_source_length_unit_is_converted_exactly(tmp_path: Path) -> None:
    model = _NATIVE["box"]()
    write_brep_text(model, tmp_path / "box.brep")
    result = read_brep_text(
        tmp_path / "box.brep",
        _POLICY,
        trusted_root=tmp_path,
        source_length_unit=MILLIMETER,
    )
    assert result.source_length_unit_meters == pytest.approx(1.0e-3)
    upper = np.max(np.asarray(_geometry(result.model).vertex_points), axis=0)
    assert np.allclose(upper, (1.0e-3, 2.0e-3, 3.0e-3), rtol=0.0, atol=1.0e-15)
    assert _volume(result.model) == pytest.approx(6.0e-9, rel=1.0e-6)
    # Unit changes preserve supplied p-curves rather than refitting their shape.
    for coedge, pcurve in enumerate(_geometry(result.model).pcurves):
        edge = _geometry(result.model).coedge_edges[coedge]
        first, last = np.asarray(_geometry(result.model).edge_ranges[edge])
        parameters = np.linspace(float(first), float(last), 7)
        face = next(
            face
            for face, loops in enumerate(_geometry(result.model).face_loops)
            if any(coedge in loop for loop in loops)
        )
        on_surface = np.asarray(
            result.model.patches[face].evaluate(pcurve.evaluate(jnp.asarray(parameters)))
        )
        carrier = _geometry(result.model).curves[
            _geometry(result.model).edge_curves[edge]
        ]
        np.testing.assert_allclose(
            np.asarray(on_surface),
            np.asarray(carrier.evaluate(jnp.asarray(parameters))),
            atol=1.0e-12,
        )


def test_placed_occurrences_round_trip(tmp_path: Path) -> None:
    ring = _NATIVE["revolved-ring"]()
    geometry = _geometry(ring)
    rotation = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    placed = BRepOccurrence(
        ("solid0", "instance1"),
        0,
        np.asarray(rotation),
        np.asarray((5.0, 0.0, 0.0)),
    )
    first = BRepOccurrence(("solid0", "instance0"), 0)
    rebuilt = BRepGeometry(
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
        occurrences=(first, placed),
        assembly_containers=(
            BRepAssemblyContainer(("model",), (first.path, placed.path)),
        ),
        vertex_roots=geometry.vertex_roots,
        edge_endpoint_roots=geometry.edge_endpoint_roots,
        coedge_endpoint_roots=geometry.coedge_endpoint_roots,
    )
    model = assemble_brep_model(
        rebuilt,
        ring.patches,
        ring.parameter_bounds,
        ring.orientation,
        ring.physical_tags,
        coordinate_contract=_CONTRACT,
        source_id="placed-ring",
        source_format="native",
        source_digest=ring.source_digest,
        import_policy_id=ring.import_policy_id,
    )
    write_brep_text(model, tmp_path / "assembly.brep")
    result = read_brep_text(
        tmp_path / "assembly.brep",
        _POLICY,
        trusted_root=tmp_path,
        source_length_unit=METER,
    )
    assert _geometry(result.model).occurrences == (first, placed)
    assert _geometry(result.model).assembly_containers == (
        BRepAssemblyContainer(("model",), (first.path, placed.path)),
    )
    assert _geometry(result.model).geometry_id == rebuilt.geometry_id


# ------------------------------------------------------------ malformed input


def _valid_text(tmp_path: Path) -> str:
    write_brep_text(_NATIVE["box"](), tmp_path / "box.brep")
    return (tmp_path / "box.brep").read_text()


def _decode(text: str) -> None:
    decode_brep_text_bytes(text.encode("ascii"), _POLICY, source_length_unit=METER)


def test_unknown_header_is_refused() -> None:
    with pytest.raises(CadInterchangeError) as refused:
        _decode("DBRep_DrawableShape\n\nCASCADE Topology V9, (c) Nobody\n")
    assert refused.value.refusal.reason == "malformed"


def test_truncated_file_is_refused(tmp_path: Path) -> None:
    text = _valid_text(tmp_path)
    with pytest.raises(CadInterchangeError) as refused:
        _decode(text[: len(text) // 2])
    assert refused.value.refusal.reason == "malformed"


def test_dangling_shape_reference_is_refused(tmp_path: Path) -> None:
    text = _valid_text(tmp_path)
    head, _, _ = text.rpartition("+1 0")
    with pytest.raises(CadInterchangeError) as refused:
        _decode(head + "+999 0\n")
    assert refused.value.refusal.reason == "dangling-reference"


def test_forward_shape_reference_is_refused_as_cycle() -> None:
    text = (
        "DBRep_DrawableShape\n\nCASCADE Topology V3, (c) Open Cascade\n"
        "Locations 0\nCurve2ds 0\nCurves 0\nPolygon3D 0\nPolygonOnTriangulations 0\n"
        "Surfaces 0\nTriangulations 0\nTShapes 1\nCo\n\n1100000\n+1 0 *\n\n+1 0\n"
    )
    with pytest.raises(CadInterchangeError) as refused:
        _decode(text)
    assert refused.value.refusal.reason == "cyclic-reference"


def test_entity_limit_is_enforced(tmp_path: Path) -> None:
    text = _valid_text(tmp_path)
    tight = CadImportPolicy(
        _CONTRACT,
        ResourceLimits(
            max_bytes=1 << 22,
            max_depth=32,
            max_nodes=20,
            max_attributes=2_000_000,
            max_losses=8,
        ),
    )
    with pytest.raises(CadInterchangeError) as refused:
        decode_brep_text_bytes(text.encode("ascii"), tight, source_length_unit=METER)
    assert refused.value.refusal.reason == "limit"


def test_nested_offset_operation_refuses_unknown_basis_record(tmp_path: Path) -> None:
    text = _valid_text(tmp_path)
    head, marker, tail = text.partition("Surfaces 6\n")
    _, _, rest = tail.partition("\n")
    with pytest.raises(CadInterchangeError) as refused:
        _decode(head + marker + "11 0.25\n6 0 0 1\n10\n" + rest)
    assert refused.value.refusal.reason == "malformed"
    assert "Unknown curve type 10" in refused.value.refusal.message


def test_unreferenced_exact_parabola_is_admitted(tmp_path: Path) -> None:
    text = _valid_text(tmp_path)
    text = text.replace("Curves 12\n", "Curves 13\n", 1).replace(
        "Polygon3D 0", "4 0 0 0 0 0 1 1 0 0 0 1 0 2.5\nPolygon3D 0", 1
    )
    result = decode_brep_text_bytes(
        text.encode("ascii"), _POLICY, source_length_unit=METER
    )
    assert result.model.topology.num_faces == 6


# ------------------------------------------------------- OCCT as a reference


def _occt_shapes() -> dict[str, object]:
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
    from OCP.BRepPrimAPI import (
        BRepPrimAPI_MakeBox,
        BRepPrimAPI_MakeCylinder,
        BRepPrimAPI_MakeRevol,
        BRepPrimAPI_MakeSphere,
        BRepPrimAPI_MakeTorus,
    )
    from OCP.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt

    plate = BRepPrimAPI_MakeBox(gp_Pnt(-2, -2, 0), 4.0, 4.0, 0.5).Shape()
    hole = BRepPrimAPI_MakeCylinder(
        gp_Ax2(gp_Pnt(0, 0, -1), gp_Dir(0, 0, 1)), 0.5, 3.0
    ).Shape()
    rectangle = BRepBuilderAPI_MakePolygon(
        gp_Pnt(1, 0, 0), gp_Pnt(2, 0, 0), gp_Pnt(2, 0, 1), gp_Pnt(1, 0, 1), True
    ).Wire()
    return {
        "box": BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1.0, 2.0, 3.0).Shape(),
        "cylinder": BRepPrimAPI_MakeCylinder(0.5, 2.0).Shape(),
        "sphere": BRepPrimAPI_MakeSphere(0.75).Shape(),
        "torus": BRepPrimAPI_MakeTorus(2.0, 0.5).Shape(),
        "plate-with-hole": BRepAlgoAPI_Cut(plate, hole).Shape(),
        "revolved-ring": BRepPrimAPI_MakeRevol(
            BRepBuilderAPI_MakeFace(rectangle).Face(),
            gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)),
        ).Shape(),
    }


def _occt_counts(shape: object) -> tuple[int, int, int, int]:
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    counts = []
    for kind in (TopAbs_SOLID, TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX):
        explorer = TopExp_Explorer(shape, kind)
        shapes = []
        while explorer.More():
            current = explorer.Current()
            if not any(current.IsSame(previous) for previous in shapes):
                shapes.append(current)
            explorer.Next()
        counts.append(
            sum(
                kind != TopAbs_EDGE or not BRep_Tool.Degenerated_s(TopoDS.Edge(current))
                for current in shapes
            )
        )
    return counts[0], counts[1], counts[2], counts[3]


def _occt_volume(shape: object) -> float:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties)
    return properties.Mass()


@pytest.mark.parametrize("name", sorted(_NATIVE))
def test_occt_written_files_match_occt_topology_and_volume(
    name: str, tmp_path: Path
) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools

    shape = _occt_shapes()[name]
    # Derived triangulations are written too; the reader must drop them explicitly.
    BRepMesh_IncrementalMesh(shape, 0.2)
    BRepTools.Write_s(shape, str(tmp_path / "occt.brep"))
    result = read_brep_text(
        tmp_path / "occt.brep", _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert _counts(result.model) == _occt_counts(shape)
    assert _volume(result.model) == pytest.approx(_occt_volume(shape), rel=1.0e-6)
    assert result.coverage.pcurves_fitted == 0
    assert "Triangulations" in {loss.path for loss in result.report.losses}


@pytest.mark.parametrize("name", sorted(_NATIVE))
def test_occt_reads_native_files_as_valid_solids(name: str, tmp_path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRep import BRep_Builder
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepTools import BRepTools
    from OCP.TopoDS import TopoDS_Shape

    model = _NATIVE[name]()
    write_brep_text(model, tmp_path / "native.brep")
    shape = TopoDS_Shape()
    assert BRepTools.Read_s(shape, str(tmp_path / "native.brep"), BRep_Builder())
    assert BRepCheck_Analyzer(shape).IsValid()
    assert _occt_counts(shape) == _counts(model)
    assert _occt_volume(shape) == pytest.approx(_VOLUMES[name], rel=1.0e-6)


def test_occt_located_assembly_becomes_occurrences(tmp_path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRep import BRep_Builder
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.BRepTools import BRepTools
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS_Compound

    unit = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 1.0, 2.0, 3.0).Shape()
    turn = gp_Trsf()
    turn.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 0.5)
    shift = gp_Trsf()
    shift.SetTranslation(gp_Vec(0, 0, 7))
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    builder.Add(compound, unit)
    builder.Add(compound, unit.Moved(TopLoc_Location(shift * turn)))
    BRepTools.Write_s(compound, str(tmp_path / "assembly.brep"))
    result = read_brep_text(
        tmp_path / "assembly.brep",
        _POLICY,
        trusted_root=tmp_path,
        source_length_unit=METER,
    )
    occurrences = _geometry(result.model).occurrences
    assert [occurrence.path for occurrence in occurrences] == [
        ("solid0", "instance0"),
        ("solid0", "instance1"),
    ]
    assert result.model.topology.num_solids == 1
    expected = np.asarray(
        (
            (np.cos(0.5), -np.sin(0.5), 0.0),
            (np.sin(0.5), np.cos(0.5), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    assert np.allclose(occurrences[1].rotation, expected, atol=1.0e-14)
    assert np.allclose(occurrences[1].translation, (0.0, 0.0, 7.0), atol=1.0e-14)
    assert _geometry(result.model).assembly_containers == (
        BRepAssemblyContainer(
            ("model",), tuple(occurrence.path for occurrence in occurrences)
        ),
    )


def _rational_sheet(*, free_edge: bool = False) -> BRepModel:
    controls = np.asarray(
        (((0.0, 0.0, 0.0), (0.0, 2.0, 0.0)), ((3.0, 0.0, 0.0), (3.0, 2.0, 1.0)))
    )
    weights = np.asarray(((1.0, 2.0), (3.0, 1.0)))
    knots = np.asarray((0.0, 0.0, 1.0, 1.0))
    patch = BSplineSurfacePatch(controls, weights, knots, knots, 1, 1)
    stage = CadStage(3.0, None)
    vertices = [
        stage.vertex(controls[i, j], f"corner:{i}:{j}")
        for i, j in ((0, 0), (1, 0), (1, 1), (0, 1))
    ]
    boundaries = (
        (controls[:, 0], weights[:, 0], 0, 1, LineCurve((0.0, 0.0), (1.0, 0.0)), 1),
        (controls[1, :], weights[1, :], 1, 2, LineCurve((1.0, 0.0), (0.0, 1.0)), 1),
        (controls[:, 1], weights[:, 1], 3, 2, LineCurve((0.0, 1.0), (1.0, 0.0)), -1),
        (controls[0, :], weights[0, :], 0, 3, LineCurve((0.0, 0.0), (0.0, 1.0)), -1),
    )
    loop = []
    for index, (points, row_weights, start, end, pcurve, sense) in enumerate(boundaries):
        edge = stage.edge(
            BSplineCurve(points, row_weights, knots, 1),
            vertices[start],
            vertices[end],
            f"edge:{index}",
            parameter_range=(0.0, 1.0),
        )
        loop.append(StagedCoedge(edge, sense, pcurve, f"edge:{index}"))
    stage.face(patch, [loop], 1, "bspline", "rational-sheet", outer_known=True)
    if free_edge:
        start = stage.vertex(np.asarray((5.0, 0.0, 0.0)), "free-start")
        end = stage.vertex(np.asarray((5.0, 2.0, 0.0)), "free-end")
        stage.edge(
            LineCurve((5.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            start,
            end,
            "free-edge",
            parameter_range=(0.0, 2.0),
        )
    return stage.publish(
        coordinate_contract=_CONTRACT,
        source_id="rational-sheet",
        source_format="native",
        source_digest="0" * 64,
        import_policy_id=_POLICY.policy_id,
        tessellation=None,
        default_occurrences=True,
    )


@pytest.mark.parametrize("source_unit", (METER, MILLIMETER))
def test_rational_spline_sheet_preserves_carriers_and_pcurves(
    source_unit: UnitDefinition, tmp_path: Path
) -> None:
    model = _rational_sheet()
    path = tmp_path / "rational.brep"
    write_brep_text(model, path)
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=source_unit
    )
    factor = 1.0 if source_unit is METER else 1.0e-3
    uv = np.asarray(((0.2, 0.7), (0.9, 0.1), (0.5, 0.5)))
    np.testing.assert_allclose(
        np.asarray(result.model.patches[0].evaluate(jnp.asarray(uv))),
        np.asarray(factor * model.patches[0].evaluate(jnp.asarray(uv))),
        atol=1.0e-14,
    )
    np.testing.assert_array_equal(
        np.asarray(_require_type(result.model.patches[0], BSplineSurfacePatch).weights),
        np.asarray(_require_type(model.patches[0], BSplineSurfacePatch).weights),
    )
    assert result.model.topology.num_faces == 1 and result.model.topology.num_edges == 4
    for actual, expected in zip(
        _geometry(result.model).pcurves, _geometry(model).pcurves, strict=True
    ):
        np.testing.assert_array_equal(
            np.asarray(actual.evaluate(jnp.asarray((0.0, 0.3, 1.0)))),
            np.asarray(expected.evaluate(jnp.asarray((0.0, 0.3, 1.0)))),
        )


def test_inconsistent_supplied_pcurve_is_not_silently_recovered(tmp_path: Path) -> None:
    text = _valid_text(tmp_path)
    head, marker, tail = text.partition("Curve2ds ")
    count, separator, rest = tail.partition("\n")
    first, _, remaining = rest.partition("\n")
    fields = first.split()
    coordinate = 6 if fields[0] == "7" else 1
    fields[coordinate] = str(float(fields[coordinate]) + 10.0)
    damaged = head + marker + count + separator + " ".join(fields) + "\n" + remaining
    with pytest.raises(CadInterchangeError) as refused:
        _decode(damaged)
    assert refused.value.refusal.reason == "inconsistent-geometry"
    assert refused.value.refusal.chain


def test_sample_invisible_curve_is_not_promoted_to_exact_pcurve() -> None:
    patch = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0)
    parameters = np.asarray((0.0, 0.010, 0.011, 0.012, 1.0))
    controls = np.stack(
        (np.asarray((1.0, 1.0, 2.0, 1.0, 1.0)), np.zeros(5), parameters), axis=1
    )
    curve = BSplineCurve(controls, np.ones(5), np.r_[0.0, parameters, 1.0], 1)
    with pytest.raises(ValueError):
        recover_pcurve(patch, curve, 0.0, 1.0, 1.0, None)


def test_distinct_nearby_locations_never_merge() -> None:
    first = RigidPlacement.identity()
    second = RigidPlacement(np.eye(3), np.asarray((1.0e-13, 0.0, 0.0)))
    assert first.key() != second.key()
    assert first.compose(second).translation[0] == 1.0e-13


def test_topological_reference_depth_is_bounded(tmp_path: Path) -> None:
    text = _valid_text(tmp_path)
    tight = CadImportPolicy(
        _CONTRACT,
        ResourceLimits(
            max_bytes=1 << 22,
            max_depth=3,
            max_nodes=20_000,
            max_attributes=2_000_000,
            max_losses=8,
        ),
    )
    with pytest.raises(CadInterchangeError) as refused:
        decode_brep_text_bytes(text.encode("ascii"), tight, source_length_unit=METER)
    assert refused.value.refusal.reason == "limit"


def test_expanded_knot_multiplicities_are_bounded(tmp_path: Path) -> None:
    path = tmp_path / "rational.brep"
    write_brep_text(_rational_sheet(), path)
    text = path.read_text().replace("0.0 2 1.0 2", "0.0 2000000000 1.0 2", 1)
    with pytest.raises(CadInterchangeError) as refused:
        _decode(text)
    assert refused.value.refusal.reason == "limit"


def test_occt_produced_rational_surface_has_exact_native_pcurves(tmp_path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepTools import BRepTools
    from OCP.collections import (
        Array1_double,
        Array1_int,
        Array2_double,
        Array2_gp_Pnt,
    )
    from OCP.Geom import Geom_BSplineSurface
    from OCP.gp import gp_Pnt

    native = _rational_sheet()
    patch = _require_type(native.patches[0], BSplineSurfacePatch)
    poles = Array2_gp_Pnt(1, 2, 1, 2)
    weights = Array2_double(1, 2, 1, 2)
    knots = Array1_double(1, 2)
    multiplicities = Array1_int(1, 2)
    for i in range(2):
        knots.SetValue(i + 1, float(i))
        multiplicities.SetValue(i + 1, 2)
        for j in range(2):
            poles.SetValue(
                i + 1, j + 1, gp_Pnt(*map(float, np.asarray(patch.control_points)[i, j]))
            )
            weights.SetValue(i + 1, j + 1, float(patch.weights[i, j]))
    surface = Geom_BSplineSurface(
        poles, weights, knots, knots, multiplicities, multiplicities, 1, 1
    )
    face = BRepBuilderAPI_MakeFace(surface, 1.0e-7).Face()
    path = tmp_path / "occt-rational.brep"
    BRepTools.Write_s(face, str(path))
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    uv = np.asarray(((0.1, 0.4), (0.8, 0.7), (0.5, 0.5)))
    np.testing.assert_allclose(
        np.asarray(result.model.patches[0].evaluate(jnp.asarray(uv))),
        np.asarray(patch.evaluate(jnp.asarray(uv))),
        atol=1.0e-13,
    )
    assert result.model.topology.num_edges == 4
    for coedge, pcurve in enumerate(_geometry(result.model).pcurves):
        edge = _geometry(result.model).coedge_edges[coedge]
        first, last = np.asarray(_geometry(result.model).edge_ranges[edge])
        parameters = np.linspace(float(first), float(last), 11)
        carrier = _geometry(result.model).curves[
            _geometry(result.model).edge_curves[edge]
        ]
        np.testing.assert_allclose(
            np.asarray(
                result.model.patches[0].evaluate(pcurve.evaluate(jnp.asarray(parameters)))
            ),
            np.asarray(carrier.evaluate(jnp.asarray(parameters))),
            atol=1.0e-12,
        )


@pytest.mark.parametrize("name", ("sphere", "torus", "revolved-ring"))
def test_unit_conversion_keeps_seams_and_pole_edges(name: str, tmp_path: Path) -> None:
    model = _NATIVE[name]()
    path = tmp_path / f"{name}.brep"
    write_brep_text(model, path)
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=MILLIMETER
    )
    assert _geometry(result.model).edge_curves == _geometry(model).edge_curves
    assert _geometry(result.model).edge_vertices == _geometry(model).edge_vertices
    assert _geometry(result.model).face_loops == _geometry(model).face_loops
    np.testing.assert_allclose(
        np.asarray(_geometry(result.model).vertex_points),
        np.asarray(1.0e-3 * np.asarray(_geometry(model).vertex_points)),
        atol=1.0e-14,
    )
    assert _volume(result.model) == pytest.approx(_VOLUMES[name] * 1.0e-9, rel=1.0e-6)


def test_rectangular_surface_trim_is_not_discarded(tmp_path: Path) -> None:
    path = tmp_path / "sheet.brep"
    write_brep_text(_rational_sheet(), path)
    text = path.read_text()
    admitted = text.replace("Surfaces 1\n", "Surfaces 1\n10 0 1 0 1\n", 1)
    result = decode_brep_text_bytes(
        admitted.encode("ascii"), _POLICY, source_length_unit=METER
    )
    assert result.model.topology.num_faces == 1
    clipped = text.replace("Surfaces 1\n", "Surfaces 1\n10 0 0.5 0 1\n", 1)
    with pytest.raises(CadInterchangeError) as refused:
        _decode(clipped)
    assert refused.value.refusal.reason == "inconsistent-geometry"
    assert refused.value.refusal.chain[-1].endswith("Fa")


def test_explicit_pcurve_approximation_reports_observed_error() -> None:
    controls = np.asarray(
        (((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), ((1.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    )
    knots = np.asarray((0.0, 0.0, 1.0, 1.0))
    patch = BSplineSurfacePatch(controls, np.ones((2, 2)), knots, knots, 1, 1)
    curve = BSplineCurve.bezier(
        np.asarray(((0, 0, 0), (0.5, 0.5, 0), (1, 1, 1))),
        np.ones(3),
    )
    with pytest.raises(ValueError):
        recover_pcurve(patch, curve, 0.0, 1.0, 1.0, None)
    fitted = recover_pcurve(
        patch, curve, 0.0, 1.0, 1.0, CadCurveFitPolicy(tolerance=1.0e-7)
    )
    assert fitted.kind == "fitted" and fitted.error <= 1.0e-7
    parameters = np.asarray((0.013, 0.273, 0.817))
    np.testing.assert_allclose(
        np.asarray(patch.evaluate(fitted.curve.evaluate(jnp.asarray(parameters)))),
        np.asarray(curve.evaluate(jnp.asarray(parameters))),
        atol=1.0e-7,
    )


def test_shared_face_shell_incidences_preserve_opposite_senses(tmp_path: Path) -> None:
    from phydrax.geometry.brep._partition import (
        BRepPartitionOperand,
        BRepPartitionPlan,
        BRepPartitionPolicy,
        BRepPartitionRole,
        partition_brep,
    )

    operands = tuple(
        BRepPartitionOperand(
            name,
            brep_box(lower, upper, coordinate_contract=_CONTRACT),
            BRepPartitionRole.REGION,
        )
        for name, lower, upper in (
            ("left", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ("right", (1.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
        )
    )
    plan = BRepPartitionPlan(
        _CONTRACT,
        operands,
        BRepPartitionPolicy(("left", "right")),
    )
    model = partition_brep(plan, destination=tmp_path / "shared.phx").model
    geometry = _geometry(model)
    shared = [
        face for face, solids in enumerate(model.topology.face_solids) if len(solids) == 2
    ]
    assert len(shared) == 1
    face = shared[0]
    signs = tuple(
        geometry.shell_orientations[shell][geometry.shell_faces[shell].index(face)]
        for shell, faces in enumerate(geometry.shell_faces)
        if face in faces
    )
    assert signs == (1, -1) or signs == (-1, 1)
    path = tmp_path / "shared.brep"
    write_brep_text(model, path)
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert _geometry(result.model).geometry_id == geometry.geometry_id


@pytest.mark.parametrize(
    "policy", (CadExportPolicy(maximum_bytes=16), CadExportPolicy(maximum_entities=12))
)
def test_export_limits_refuse_before_publication(
    policy: CadExportPolicy, tmp_path: Path
) -> None:
    path = tmp_path / "limited.brep"
    with pytest.raises(CadInterchangeError) as refused:
        write_brep_text(_rational_sheet(), path, policy=policy)
    assert refused.value.refusal.reason == "limit"
    assert not path.exists()


def test_exact_empty_compound_round_trip_has_no_fake_occurrences(tmp_path: Path) -> None:
    stage = CadStage(1.0, None)
    model = stage.publish(
        coordinate_contract=_CONTRACT,
        source_id="empty",
        source_format="native",
        source_digest="0" * 64,
        import_policy_id=_POLICY.policy_id,
        tessellation=None,
        default_occurrences=True,
    )
    path = tmp_path / "empty.brep"
    write_brep_text(model, path)
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert _geometry(result.model).geometry_id == _geometry(model).geometry_id
    assert _counts(result.model) == (0, 0, 0, 0)
    assert _geometry(result.model).occurrences == ()
    assert np.asarray(result.model.parameter_bounds).shape == (0, 2, 2)


def test_export_honors_authoritative_trim_closure(tmp_path: Path) -> None:
    model = _rational_sheet()
    exact = eqx.tree_at(
        lambda value: value.trim_domains,
        model,
        (TrimDomain(np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)))),),
    )
    path = tmp_path / "polygon.brep"
    write_brep_text(exact, path)
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert _geometry(result.model).geometry_id == _geometry(model).geometry_id
    different = eqx.tree_at(
        lambda value: value.trim_domains,
        model,
        (TrimDomain(np.asarray(((0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)))),),
    )
    with pytest.raises(CadInterchangeError) as refused:
        write_brep_text(different, tmp_path / "different.brep")
    assert refused.value.refusal.reason == "inexact-export"
    assert refused.value.refusal.chain == ("face:0", "loop:0")


def test_free_edge_and_face_round_trip_without_dropping_geometry(tmp_path: Path) -> None:
    model = _rational_sheet(free_edge=True)
    path = tmp_path / "face-and-edge.brep"
    write_brep_text(model, path)
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert _geometry(result.model).geometry_id == _geometry(model).geometry_id
    assert result.model.topology.num_faces == 1 and result.model.topology.num_edges == 5
    free = [
        index for index, faces in enumerate(result.model.topology.edge_faces) if not faces
    ]
    assert free == [4]
    edge = _geometry(result.model).curves[_geometry(result.model).edge_curves[free[0]]]
    np.testing.assert_allclose(
        np.asarray(edge.evaluate(jnp.asarray((0.0, 0.3, 2.0)))),
        np.asarray(((5, 0, 0), (5, 0.3, 0), (5, 2, 0))),
    )


def test_occt_planar_face_compound_preserves_unattached_edge(tmp_path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace
    from OCP.BRepTools import BRepTools
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
    from OCP.TopoDS import TopoDS_Compound

    face = BRepBuilderAPI_MakeFace(
        gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 0, 2, 0, 1
    ).Face()
    edge = BRepBuilderAPI_MakeEdge(gp_Pnt(5, 0, 0), gp_Pnt(5, 3, 0)).Edge()
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    builder.Add(compound, face)
    builder.Add(compound, edge)
    path = tmp_path / "occt-face-and-edge.brep"
    BRepTools.Write_s(compound, str(path))
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert result.model.topology.num_faces == 1 and result.model.topology.num_edges == 5
    free = [
        index for index, faces in enumerate(result.model.topology.edge_faces) if not faces
    ]
    assert len(free) == 1
    endpoints = np.asarray(_geometry(result.model).vertex_points)[
        list(_geometry(result.model).edge_vertices[free[0]])
    ]
    np.testing.assert_allclose(endpoints, np.asarray(((5, 0, 0), (5, 3, 0))))
    exported = tmp_path / "round-trip.brep"
    write_brep_text(result.model, exported)
    restored = read_brep_text(
        exported, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert _geometry(restored.model).geometry_id == _geometry(result.model).geometry_id


@pytest.mark.parametrize("periodic_axes", ((True, False), (False, True), (True, True)))
def test_occt_periodic_rational_surface_unrolls_exactly(
    periodic_axes: tuple[bool, bool], tmp_path: Path
) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepTools import BRepTools
    from OCP.collections import (
        Array1_double,
        Array1_int,
        Array2_double,
        Array2_gp_Pnt,
    )
    from OCP.Geom import Geom_BSplineSurface
    from OCP.gp import gp_Pnt

    counts = tuple(6 if periodic else 2 for periodic in periodic_axes)
    degrees = tuple(3 if periodic else 1 for periodic in periodic_axes)
    knots, mults = [], []
    for count, periodic in zip(counts, periodic_axes, strict=True):
        size = count + 1 if periodic else 2
        values = Array1_double(1, size)
        multiplicities = Array1_int(1, size)
        for index in range(size):
            values.SetValue(index + 1, float(index))
            multiplicities.SetValue(index + 1, 1 if periodic else 2)
        knots.append(values)
        mults.append(multiplicities)
    poles = Array2_gp_Pnt(1, counts[0], 1, counts[1])
    weights = Array2_double(1, counts[0], 1, counts[1])
    for i in range(counts[0]):
        u = 2 * np.pi * i / counts[0] if periodic_axes[0] else 0.5 * i
        for j in range(counts[1]):
            v = 2 * np.pi * j / counts[1] if periodic_axes[1] else 0.5 * j
            point = ((3 + np.cos(v)) * np.cos(u), (3 + np.cos(v)) * np.sin(u), np.sin(v))
            poles.SetValue(i + 1, j + 1, gp_Pnt(*point))
            weights.SetValue(i + 1, j + 1, 1.0 + 0.1 * np.sin(u) + 0.05 * np.cos(v))
    surface = Geom_BSplineSurface(
        poles,
        weights,
        knots[0],
        knots[1],
        mults[0],
        mults[1],
        degrees[0],
        degrees[1],
        periodic_axes[0],
        periodic_axes[1],
    )
    face = BRepBuilderAPI_MakeFace(surface, 1.0e-7).Face()
    path = tmp_path / "periodic-surface.brep"
    BRepTools.Write_s(face, str(path))
    result = read_brep_text(
        path, _OCCT_SPLINE_POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    uv = np.asarray(
        [
            (u, v)
            for u in np.linspace(0, counts[0] if periodic_axes[0] else 1, 7)
            for v in np.linspace(0, counts[1] if periodic_axes[1] else 1, 7)
        ]
    )
    expected = []
    for u, v in uv:
        point = surface.Value(float(u), float(v))
        expected.append((point.X(), point.Y(), point.Z()))
    np.testing.assert_allclose(
        np.asarray(result.model.patches[0].evaluate(jnp.asarray(uv))),
        np.asarray(expected),
        atol=1.0e-11,
    )
    assert result.model.topology.num_faces == 1
    assert result.coverage.pcurves_fitted == 0
    exported = tmp_path / "unrolled.brep"
    write_brep_text(result.model, exported)
    restored = read_brep_text(
        exported, _OCCT_SPLINE_POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert _geometry(restored.model).geometry_id == _geometry(result.model).geometry_id


def test_occt_periodic_rational_curve_keeps_original_pole_origin(tmp_path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCP.BRepTools import BRepTools
    from OCP.collections import Array1_double, Array1_gp_Pnt, Array1_int
    from OCP.Geom import Geom_BSplineCurve
    from OCP.gp import gp_Pnt

    poles = Array1_gp_Pnt(1, 6)
    weights = Array1_double(1, 6)
    knots = Array1_double(1, 7)
    mults = Array1_int(1, 7)
    for index in range(6):
        angle = index * 2 * np.pi / 6
        poles.SetValue(
            index + 1, gp_Pnt(np.cos(angle), np.sin(angle), 0.2 * np.sin(2 * angle))
        )
        weights.SetValue(index + 1, 1.0 + 0.1 * index)
    for index in range(7):
        knots.SetValue(index + 1, float(index))
        mults.SetValue(index + 1, 1)
    curve = Geom_BSplineCurve(poles, weights, knots, mults, 3, True)
    edge = BRepBuilderAPI_MakeEdge(curve).Edge()
    path = tmp_path / "periodic-edge.brep"
    BRepTools.Write_s(edge, str(path))
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    parameters = np.asarray((0.0, 0.01, 0.8, 2.3, 5.99, 6.0))
    expected = []
    for parameter in parameters:
        point = curve.Value(float(parameter))
        expected.append((point.X(), point.Y(), point.Z()))
    native = _geometry(result.model).curves[_geometry(result.model).edge_curves[0]]
    np.testing.assert_allclose(
        np.asarray(native.evaluate(jnp.asarray(parameters))),
        np.asarray(expected),
        atol=1.0e-12,
    )
    assert result.model.topology.edge_faces == ((),)


def test_occt_surface_array_fixture_imports_native_exact_surface(tmp_path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepTools import BRepTools
    from OCP.collections import Array2_gp_Pnt
    from OCP.GeomAbs import GeomAbs_C2
    from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
    from OCP.gp import gp_Pnt

    points = Array2_gp_Pnt(1, 5, 1, 5)
    for i, v in enumerate(np.linspace(0, 1, 5)):
        for j, u in enumerate(np.linspace(0, 1, 5)):
            points.SetValue(
                i + 1, j + 1, gp_Pnt(u, v, 0.15 * u * v + 0.05 * u**2 - 0.03 * v**2)
            )
    surface = GeomAPI_PointsToBSplineSurface(points, 2, 3, GeomAbs_C2, 1.0e-6).Surface()
    path = tmp_path / "surface-array.brep"
    BRepTools.Write_s(BRepBuilderAPI_MakeFace(surface, 1.0e-6).Face(), str(path))
    result = read_brep_text(
        path, _OCCT_SPLINE_POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    _require_type(result.model.patches[0], BSplineSurfacePatch)
    bounds = np.asarray(result.model.parameter_bounds[0])
    uv = np.asarray(
        [
            (u, v)
            for u in np.linspace(float(bounds[0, 0]), float(bounds[1, 0]), 5)
            for v in np.linspace(float(bounds[0, 1]), float(bounds[1, 1]), 5)
        ]
    )
    expected = []
    for u, v in uv:
        point = surface.Value(float(u), float(v))
        expected.append((point.X(), point.Y(), point.Z()))
    np.testing.assert_allclose(
        np.asarray(result.model.patches[0].evaluate(jnp.asarray(uv))),
        np.asarray(expected),
        atol=1.0e-12,
    )


def test_occt_three_circle_loft_preserves_spline_solid(tmp_path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    from OCP.BRepTools import BRepTools
    from OCP.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt

    loft = BRepOffsetAPI_ThruSections(True, False, 1.0e-7)
    for center, radius in (
        ((0, 0, 0), 1.0),
        ((0.15, -0.05, 0.8), 1.25),
        ((-0.1, 0.1, 1.7), 0.9),
    ):
        circle = gp_Circ(gp_Ax2(gp_Pnt(*center), gp_Dir(0, 0, 1)), radius)
        loft.AddWire(
            BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(circle).Edge()).Wire()
        )
    loft.Build()
    shape = loft.Shape()
    path = tmp_path / "circle-loft.brep"
    BRepTools.Write_s(shape, str(path))
    result = read_brep_text(
        path, _OCCT_SPLINE_POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert result.model.mesh_vertices.shape == (0, 3)
    assert result.model.mesh_faces.shape == (0, 3)
    assert any(isinstance(patch, BSplineSurfacePatch) for patch in result.model.patches)
    assert _counts(result.model) == _occt_counts(shape)
    assert _volume(result.model) == pytest.approx(_occt_volume(shape), rel=2.0e-5)


def _periodic_circle_text(*, last_multiplicity: int = 2) -> str:
    weight = float(np.sqrt(0.5))
    points = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
    poles = " ".join(
        f"{x} {y} 0 {1.0 if i % 2 == 0 else weight!r}" for i, (x, y) in enumerate(points)
    )
    return (
        "DBRep_DrawableShape\n\nCASCADE Topology V3, (c) Open Cascade\n"
        "Locations 0\nCurve2ds 0\nCurves 1\n"
        f"7 1 1 2 8 5 {poles} 0 2 1 2 2 2 3 2 4 {last_multiplicity}\n"
        "Polygon3D 0\nPolygonOnTriangulations 0\nSurfaces 0\nTriangulations 0\n"
        "TShapes 2\nVe\n1e-07\n1 0 0\n0 0\n0101101\n*\n"
        "Ed\n1e-07 1 1 0\n1 1 0 0 4\n0\n0101000\n+2 0 -2 0 *\n\n+1 0\n"
    )


def test_periodic_rational_circle_imports_without_occt_runtime(tmp_path: Path) -> None:
    result = decode_brep_text_bytes(
        _periodic_circle_text().encode("ascii"),
        _POLICY,
        source_length_unit=METER,
    )
    curve = _geometry(result.model).curves[0]
    root_half = np.sqrt(0.5)
    parameters = np.asarray((0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4))
    expected = np.asarray(
        (
            (1, 0, 0),
            (root_half, root_half, 0),
            (0, 1, 0),
            (-root_half, root_half, 0),
            (-1, 0, 0),
            (-root_half, -root_half, 0),
            (0, -1, 0),
            (root_half, -root_half, 0),
            (1, 0, 0),
        )
    )
    np.testing.assert_allclose(
        np.asarray(curve.evaluate(jnp.asarray(parameters))),
        np.asarray(expected),
        atol=1.0e-14,
    )
    assert result.model.topology.num_edges == 1 and result.model.topology.num_faces == 0
    assert result.model.topology.edge_faces == ((),)
    path = tmp_path / "finite-circle.brep"
    write_brep_text(result.model, path)
    restored = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert _geometry(restored.model).geometry_id == _geometry(result.model).geometry_id


def test_periodic_spline_rejects_mismatched_seam_multiplicities() -> None:
    with pytest.raises(CadInterchangeError) as refused:
        _decode(_periodic_circle_text(last_multiplicity=1))
    assert refused.value.refusal.reason == "malformed"


def test_occt_box_with_spherical_cavity_preserves_void_shell(tmp_path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere
    from OCP.BRepTools import BRepTools
    from OCP.gp import gp_Pnt

    box = BRepPrimAPI_MakeBox(gp_Pnt(-2, -2, -2), 4, 4, 4).Shape()
    sphere = BRepPrimAPI_MakeSphere(gp_Pnt(0, 0, 0), 1).Shape()
    shape = BRepAlgoAPI_Cut(box, sphere).Shape()
    path = tmp_path / "spherical-cavity.brep"
    BRepTools.Write_s(shape, str(path))
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert _counts(result.model) == _occt_counts(shape)
    assert len(_geometry(result.model).solid_shells[0]) == 2
    assert _volume(result.model) == pytest.approx(64 - 4 * np.pi / 3, rel=1.0e-6)


@pytest.mark.parametrize("source_unit", (METER, MILLIMETER))
def test_occt_split_boxes_preserve_shared_face_and_solid_incidence(
    source_unit: UnitDefinition, tmp_path: Path
) -> None:
    pytest.importorskip("OCP")
    from OCP.BOPAlgo import BOPAlgo_Splitter
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.BRepTools import BRepTools
    from OCP.gp import gp_Pnt

    splitter = BOPAlgo_Splitter()
    for x in (0.0, 1.0):
        splitter.AddArgument(BRepPrimAPI_MakeBox(gp_Pnt(x, 0, 0), 1, 1, 1).Shape())
    splitter.Perform()
    shape = splitter.Shape()
    path = tmp_path / "split-boxes.brep"
    BRepTools.Write_s(shape, str(path))
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=source_unit
    )
    assert result.model.topology.num_solids == 2
    assert _geometry(result.model).assembly_containers == (
        BRepAssemblyContainer(("model",), (("solid0",), ("solid1",))),
    )
    assert result.model.topology.num_faces == 11
    shared = set(_geometry(result.model).shell_faces[0]) & set(
        _geometry(result.model).shell_faces[1]
    )
    assert len(shared) == 1
    face = next(iter(shared))
    senses = []
    for faces, signs in zip(
        _geometry(result.model).shell_faces,
        _geometry(result.model).shell_orientations,
        strict=True,
    ):
        senses.append(signs[faces.index(face)])
    assert senses[0] == -senses[1]
    factor = 1.0 if source_unit is METER else 1.0e-3
    np.testing.assert_allclose(
        np.asarray(prepare_brep_query(result.model).measures.solid_volumes),
        np.asarray(np.asarray((factor**3, factor**3))),
        rtol=1.0e-6,
    )
    exported = tmp_path / "shared-round-trip.brep"
    write_brep_text(result.model, exported)
    restored = read_brep_text(
        exported, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert _geometry(restored.model).geometry_id == _geometry(result.model).geometry_id


def _with_assembly(
    occurrences: tuple[BRepOccurrence, ...],
    containers: tuple[BRepAssemblyContainer, ...],
) -> BRepModel:
    base = _NATIVE["box"]()
    geometry = _geometry(base)
    exact = BRepGeometry(
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
        occurrences=occurrences,
        assembly_containers=containers,
        vertex_roots=geometry.vertex_roots,
        edge_endpoint_roots=geometry.edge_endpoint_roots,
        coedge_endpoint_roots=geometry.coedge_endpoint_roots,
    )
    return assemble_brep_model(
        exact,
        base.patches,
        base.parameter_bounds,
        base.orientation,
        base.physical_tags,
        coordinate_contract=_CONTRACT,
        source_id="declared-assembly",
        source_format="native",
        source_digest=base.source_digest,
        import_policy_id=base.import_policy_id,
    )


def test_independent_instances_do_not_gain_inferred_container_from_paths_or_placement(
    tmp_path: Path,
) -> None:
    occurrences = (
        BRepOccurrence(("same-prefix", "first"), 0),
        BRepOccurrence(("same-prefix", "second"), 0),
    )
    model = _with_assembly(occurrences, ())
    path = tmp_path / "independent.brep"
    with pytest.raises(CadInterchangeError) as refused:
        write_brep_text(model, path)
    assert refused.value.refusal.reason == "inexact-export"
    assert not path.exists()
    assert _geometry(model).assembly_containers == ()


def test_arbitrary_occurrence_and_container_names_declare_normalization_loss(
    tmp_path: Path,
) -> None:
    occurrence = BRepOccurrence(
        ("experiment", "specimen-A"),
        0,
        translation=np.asarray((3.0, 0.0, 0.0)),
    )
    group = BRepAssemblyContainer(("scientific-group",), (occurrence.path,))
    model = _with_assembly((occurrence,), (group,))
    path = tmp_path / "named.brep"
    exported = write_brep_text(model, path)
    assert exported.report.status == AdapterStatus.DECLARED_LOSS
    assert {loss.path.split(":")[0] for loss in exported.report.losses} == {
        "occurrence-path",
        "container-path",
    }
    restored = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    ).model
    assert _geometry(restored).occurrences[0].translation == occurrence.translation
    assert _geometry(restored).assembly_containers == (
        BRepAssemblyContainer(("model",), (("solid0",),)),
    )
    assert _geometry(restored).geometry_id != _geometry(model).geometry_id


def test_occt_nested_compound_membership_is_explicit_and_round_trips(
    tmp_path: Path,
) -> None:
    pytest.importorskip("OCP")
    from OCP.BRep import BRep_Builder
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.BRepTools import BRepTools
    from OCP.gp import gp_Trsf, gp_Vec
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS_Compound

    unit = BRepPrimAPI_MakeBox(1, 1, 1).Shape()
    along = gp_Trsf()
    along.SetTranslation(gp_Vec(3, 0, 0))
    above = gp_Trsf()
    above.SetTranslation(gp_Vec(0, 0, 5))
    builder = BRep_Builder()
    child, root = TopoDS_Compound(), TopoDS_Compound()
    builder.MakeCompound(child)
    builder.MakeCompound(root)
    builder.Add(child, unit)
    builder.Add(child, unit.Moved(TopLoc_Location(along)))
    builder.Add(root, unit)
    builder.Add(root, child.Moved(TopLoc_Location(above)))
    path = tmp_path / "nested.brep"
    BRepTools.Write_s(root, str(path))
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    expected = (
        BRepAssemblyContainer(
            ("model",), (("solid0", "instance0"),), (("model", "container0"),)
        ),
        BRepAssemblyContainer(
            ("model", "container0"), (("solid0", "instance1"), ("solid0", "instance2"))
        ),
    )
    assert _geometry(result.model).assembly_containers == expected
    assert tuple(
        occurrence.translation for occurrence in _geometry(result.model).occurrences
    ) == (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 5.0),
        (3.0, 0.0, 5.0),
    )
    round_trip = tmp_path / "nested-round-trip.brep"
    write_brep_text(result.model, round_trip)
    restored = read_brep_text(
        round_trip, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert _geometry(restored.model).geometry_id == _geometry(result.model).geometry_id


def test_brep_substitutes_real_intersection_branch_only_under_explicit_policy(
    tmp_path: Path,
) -> None:
    model = native_intersection_cap()
    source_id = _geometry(model).geometry_id
    branch = _require_type(_geometry(model).curves[0], IntersectionCurve)
    with pytest.raises(CadInterchangeError) as exact:
        write_brep_text(model, tmp_path / "exact-cap.brep")
    assert exact.value.refusal.reason == "inexact-export"
    policy = CadExportPolicy(
        intersection_approximation=CadCurveFitPolicy(
            tolerance=1.0e-7,
            maximum_control_points=512,
            check_samples=1025,
        )
    )
    path = tmp_path / "sampled-cap.brep"
    exported = write_brep_text(model, path, policy=policy)
    assert exported.report.status == AdapterStatus.DECLARED_LOSS
    assert tuple(
        approximation.branch_id for approximation in exported.approximations
    ) == (branch.branch_id,)
    restored = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    ).model
    restored_curve = _require_type(_geometry(restored).curves[0], BSplineCurve)
    _require_type(_geometry(restored).pcurves[0], BSplineCurve)
    parameters = np.linspace(
        float(branch.parameter_interval[0]), float(branch.parameter_interval[1]), 129
    )
    points = np.asarray(restored_curve.evaluate(jnp.asarray(parameters)))
    np.testing.assert_allclose(np.asarray(points[:, 2]), np.asarray(0.3), atol=1.0e-12)
    np.testing.assert_allclose(
        np.asarray(np.linalg.norm(points[:, :2], axis=1)),
        np.asarray(np.sqrt(1.0 - 0.3**2)),
        atol=2.0e-7,
    )
    np.testing.assert_array_equal(
        np.asarray(_geometry(restored).edge_ranges),
        np.asarray(_geometry(model).edge_ranges),
    )
    assert _geometry(restored).coedge_senses == _geometry(model).coedge_senses
    assert float(prepare_brep_query(restored).measures.face_areas[0]) == pytest.approx(
        np.pi * (1.0 - 0.3**2),
        rel=2.0e-6,
    )
    assert _geometry(model).geometry_id == source_id


def test_scaled_oblique_plane_chart_exports_exact_pcurve_geometry(tmp_path: Path) -> None:
    patch = PlanePatch((0, 0, 0.5), (2, 0, 0), (1, 3, 0))
    uv = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)))
    points = np.asarray(patch.evaluate(jnp.asarray(uv)))
    stage = CadStage(3.0, None)
    vertices = [
        stage.vertex(point, f"corner:{index}") for index, point in enumerate(points)
    ]
    loop = []
    for index in range(4):
        following = (index + 1) % 4
        edge = stage.edge(
            LineCurve(points[index], points[following] - points[index]),
            vertices[index],
            vertices[following],
            f"edge:{index}",
            parameter_range=(0, 1),
        )
        loop.append(
            StagedCoedge(
                edge,
                1,
                LineCurve(uv[index], uv[following] - uv[index]),
                f"edge:{index}",
            )
        )
    stage.face(patch, [loop], 1, "plane", "oblique", outer_known=True)
    model = stage.publish(
        coordinate_contract=_CONTRACT,
        source_id="oblique",
        source_format="native",
        source_digest="0" * 64,
        import_policy_id=_POLICY.policy_id,
        tessellation=None,
        default_occurrences=True,
    )
    path = tmp_path / "oblique.brep"
    write_brep_text(model, path)
    restored = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    ).model
    roundoff_bound = (
        32.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(points))))
    )
    np.testing.assert_allclose(
        np.asarray(_geometry(restored).vertex_points),
        np.asarray(points),
        rtol=0.0,
        atol=roundoff_bound,
    )
    for coedge, pcurve in enumerate(_geometry(restored).pcurves):
        edge = _geometry(restored).coedge_edges[coedge]
        first, last = np.asarray(_geometry(restored).edge_ranges[edge])
        parameters = first + np.asarray((0.17, 0.63, 0.94)) * (last - first)
        carrier = _geometry(restored).curves[_geometry(restored).edge_curves[edge]]
        np.testing.assert_allclose(
            np.asarray(
                restored.patches[0].evaluate(pcurve.evaluate(jnp.asarray(parameters)))
            ),
            np.asarray(carrier.evaluate(jnp.asarray(parameters))),
            atol=1.0e-13,
        )
    assert float(prepare_brep_query(restored).measures.face_areas[0]) == pytest.approx(
        6.0, rel=1.0e-12
    )


def test_single_located_independent_root_keeps_placement_without_container(
    tmp_path: Path,
) -> None:
    occurrence = BRepOccurrence(("solid0",), 0, translation=np.asarray((4.0, 5.0, 6.0)))
    model = _with_assembly((occurrence,), ())
    path = tmp_path / "located-root.brep"
    write_brep_text(model, path)
    restored = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    ).model
    assert _geometry(restored).occurrences == (occurrence,)
    assert _geometry(restored).assembly_containers == ()
    assert _geometry(restored).geometry_id == _geometry(model).geometry_id


@pytest.mark.parametrize("source_unit", (METER, MILLIMETER))
def test_exact_offset_operation_tree_round_trips_with_units(
    source_unit: UnitDefinition, tmp_path: Path
) -> None:
    model = native_offset_plane(_POLICY)
    path = tmp_path / "offset.brep"
    write_brep_text(model, path)
    result = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=source_unit
    )
    patch = _require_type(result.model.patches[0], OffsetSurface)
    _require_type(patch.base, PlanePatch)
    factor = 1.0 if source_unit is METER else 1.0e-3
    assert float(patch.distance) == 0.25 * factor
    uv = factor * np.asarray(((0.2, 0.7), (1.7, 0.2)))
    np.testing.assert_allclose(
        np.asarray(patch.evaluate(jnp.asarray(uv))),
        np.asarray(factor * np.asarray(((0.2, 0.7, 0.75), (1.7, 0.2, 0.75)))),
        atol=1.0e-14,
    )
    assert result.model.topology.num_faces == 1 and result.model.topology.num_edges == 4
    assert float(
        prepare_brep_query(result.model).measures.face_areas[0]
    ) == pytest.approx(2 * factor**2, rel=1.0e-10)


def test_offset_placement_preserves_signed_normal_under_reflection() -> None:
    from phydrax.interchange._cad_carriers import place_surface

    patch = OffsetSurface(PlanePatch((0, 0, 0.5), (1, 0, 0), (0, 1, 0)), 0.25)
    matrix = np.asarray(((-1, 0, 0, 3), (0, 1, 0, 2), (0, 0, 1, 1)), dtype=np.float64)
    placement = RigidPlacement.from_matrix(matrix, "reflection", reflection=True)
    placed = place_surface(patch, placement, "offset-plane")
    uv = np.asarray(((0.2, 0.7), (1.7, 0.2)))
    expected = (
        np.asarray(patch.evaluate(jnp.asarray(uv))) @ placement.rotation.T
        + placement.translation
    )
    np.testing.assert_allclose(
        np.asarray(placed.evaluate(jnp.asarray(uv))), np.asarray(expected), atol=1.0e-14
    )


@pytest.mark.parametrize("kind", ("parabola", "hyperbola", "offset"))
@pytest.mark.parametrize("source_unit", (METER, MILLIMETER))
def test_independent_conic_and_offset_edge_retains_source_parameter_and_archive(
    kind: Literal["parabola", "hyperbola", "offset"],
    source_unit: UnitDefinition,
    tmp_path: Path,
) -> None:
    """Hand-authored BRep text; the coordinate oracle does not call a writer."""
    frame = "0 0 0 0 0 1 1 0 0 0 1 0"
    if kind == "hyperbola":
        record = f"5 {frame} 2 1"
    elif kind == "parabola":
        record = f"4 {frame} 0.5"
    else:
        record = f"9 0.25 0 0 1\n4 {frame} 0.5"

    def analytic(parameter: np.ndarray) -> np.ndarray:
        if kind == "hyperbola":
            return np.stack(
                (2.0 * np.cosh(parameter), np.sinh(parameter), np.zeros_like(parameter)),
                axis=-1,
            )
        x, y = 0.5 * parameter**2, parameter
        if kind == "offset":
            denominator = np.sqrt(1.0 + parameter**2)
            x, y = x + 0.25 / denominator, y - 0.25 * parameter / denominator
        return np.stack((x, y, np.zeros_like(parameter)), axis=-1)

    endpoints = analytic(np.asarray([-1.0, 1.0], dtype=np.float64))
    start = " ".join(repr(float(value)) for value in endpoints[0])
    end = " ".join(repr(float(value)) for value in endpoints[1])
    text = (
        "DBRep_DrawableShape\n\nCASCADE Topology V3, (c) Open Cascade\n"
        f"Locations 0\nCurve2ds 0\nCurves 1\n{record}\n"
        "Polygon3D 0\nPolygonOnTriangulations 0\nSurfaces 0\nTriangulations 0\n"
        f"TShapes 3\nVe\n1e-07\n{start}\n0 0\n0101101\n*\n"
        f"Ve\n1e-07\n{end}\n0 0\n0101101\n*\n"
        "Ed\n1e-07 1 1 0\n1 1 0 -1 1\n0\n0101000\n+3 0 -2 0 *\n\n+1 0\n"
    )
    imported = decode_brep_text_bytes(
        text.encode("ascii"), _POLICY, source_length_unit=source_unit
    )
    assert imported.report.valid
    assert imported.model.topology.num_edges == 1
    assert imported.model.topology.num_faces == 0
    assert imported.model.topology.edge_faces == ((),)
    geometry = _geometry(imported.model)
    curve = geometry.curves[geometry.edge_curves[0]]
    expected_type = {
        "parabola": ParabolaCurve,
        "hyperbola": HyperbolaCurve,
        "offset": OffsetCurve,
    }[kind]
    assert isinstance(curve, expected_type)
    factor = 1.0 if source_unit is METER else 1.0e-3
    parameter_factor = 1.0 if kind == "hyperbola" else factor
    parameters = np.asarray([-1.0, -0.3, 0.0, 0.6, 1.0], dtype=np.float64)
    expected = analytic(parameters) * factor
    values = curve.evaluate(jnp.asarray(parameters * parameter_factor))
    np.testing.assert_allclose(values, expected, rtol=1e-14, atol=1e-14)
    lower, upper = curve.derivative_bounds(-parameter_factor, parameter_factor)
    jets = jax.vmap(jax.jacfwd(curve.evaluate))(
        jnp.asarray(parameters * parameter_factor)
    )
    assert np.all(np.asarray(jets) >= lower) and np.all(np.asarray(jets) <= upper)
    path = tmp_path / "exact-edge.brep"
    write_brep_text(imported.model, path)
    reread = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert _geometry(reread.model).geometry_id == geometry.geometry_id
    receipt = save_brep_archive(imported.model, tmp_path / "exact-edge.phx")
    restored = load_brep_archive(receipt.path)
    assert restored.model_id == imported.model.model_id == receipt.model_id
    assert restored.source_id == imported.model.source_id
    np.testing.assert_array_equal(
        _geometry(restored)
        .curves[0]
        .evaluate(jnp.asarray(parameters * parameter_factor)),
        values,
    )


def test_exact_offset_edge_distance_jvp_vjp_retains_basis_authority() -> None:
    source = OffsetCurve(ParabolaCurve((0, 0), (1, 0), (0, 1), 0.5), 0.25)
    parameter = jnp.asarray(0.4, dtype=jnp.float64)

    def moved(distance: Array) -> Array:
        return eqx.tree_at(lambda curve: curve.distance, source, distance).evaluate(
            parameter
        )

    _, tangent = jax.jvp(
        moved, (source.distance,), (jnp.asarray(1.0, dtype=jnp.float64),)
    )
    expected = np.asarray([1.0, -0.4], dtype=np.float64) / np.sqrt(1.0 + 0.4**2)
    np.testing.assert_allclose(tangent, expected, rtol=1e-14, atol=1e-14)
    cotangent = jnp.asarray([0.3, -0.7], dtype=jnp.float64)
    _, pullback = jax.vjp(moved, source.distance)
    np.testing.assert_allclose(
        pullback(cotangent)[0], expected @ np.asarray(cotangent), rtol=1e-14, atol=1e-14
    )
    assert eqx.tree_equal(source.base, ParabolaCurve((0, 0), (1, 0), (0, 1), 0.5))


@pytest.mark.parametrize("kind", ("parabola", "hyperbola"))
def test_exact_conic_source_coefficients_remain_live_through_jit_jvp_and_vjp(
    kind: Literal["parabola", "hyperbola"],
) -> None:
    parameter = jnp.asarray(0.4, dtype=jnp.float64)
    source: ParabolaCurve | HyperbolaCurve
    if kind == "parabola":
        source = ParabolaCurve((0, 0, 0), (1, 0, 0), (0, 1, 0), 0.5)
        coefficients = source.focal_length[None]
        direction = jnp.asarray([0.3], dtype=jnp.float64)
        jacobian = np.asarray([[-0.16], [0.0], [0.0]], dtype=np.float64)
    else:
        source = HyperbolaCurve((0, 0, 0), (1, 0, 0), (0, 1, 0), 2.0, 1.0)
        coefficients = jnp.stack((source.first_radius, source.second_radius))
        direction = jnp.asarray([0.3, -0.2], dtype=jnp.float64)
        jacobian = np.asarray(
            [[np.cosh(0.4), 0.0], [0.0, np.sinh(0.4)], [0.0, 0.0]],
            dtype=np.float64,
        )

    def changed(values: Array) -> ParabolaCurve | HyperbolaCurve:
        if isinstance(source, ParabolaCurve):
            return eqx.tree_at(lambda curve: curve.focal_length, source, values[0])
        return eqx.tree_at(
            lambda curve: (curve.first_radius, curve.second_radius),
            source,
            (values[0], values[1]),
        )

    def point(values: Array) -> Array:
        return changed(values).evaluate(parameter)

    _, tangent = jax.jvp(point, (coefficients,), (direction,))
    np.testing.assert_allclose(
        tangent, jacobian @ np.asarray(direction), rtol=1e-14, atol=1e-14
    )
    cotangent = jnp.asarray([0.3, -0.7, 0.2], dtype=jnp.float64)
    _, pullback = jax.vjp(point, coefficients)
    np.testing.assert_allclose(
        pullback(cotangent)[0], jacobian.T @ np.asarray(cotangent), rtol=1e-14, atol=1e-14
    )

    def evaluate(curve: ParabolaCurve | HyperbolaCurve, value: Array) -> Array:
        return curve.evaluate(value)

    compiled = eqx.filter_jit(evaluate)
    shifted = coefficients + 1.0e-3 * direction
    moved = changed(shifted)
    if isinstance(moved, ParabolaCurve):
        expected = np.asarray(
            [0.4**2 / (4.0 * float(shifted[0])), 0.4, 0.0], dtype=np.float64
        )
    else:
        expected = np.asarray(
            [float(shifted[0]) * np.cosh(0.4), float(shifted[1]) * np.sinh(0.4), 0.0],
            dtype=np.float64,
        )
    np.testing.assert_allclose(
        compiled(moved, parameter), expected, rtol=1e-14, atol=1e-14
    )
    np.testing.assert_allclose(
        compiled(source, parameter), source.evaluate(parameter), rtol=1e-14, atol=1e-14
    )


@pytest.mark.parametrize("kind", ("parabola", "hyperbola", "offset"))
@pytest.mark.parametrize("reflection", (False, True), ids=("quarter-turn", "reflection"))
def test_exact_conic_and_offset_placement_preserves_source_parameter_jvp(
    kind: Literal["parabola", "hyperbola", "offset"],
    reflection: bool,
) -> None:
    from phydrax.interchange._cad_carriers import place_curve

    source: ParabolaCurve | HyperbolaCurve | OffsetCurve
    if kind == "parabola":
        source = ParabolaCurve((0, 0, 0), (1, 0, 0), (0, 1, 0), 0.5)
    elif kind == "hyperbola":
        source = HyperbolaCurve((0, 0, 0), (1, 0, 0), (0, 1, 0), 2.0, 1.0)
    else:
        source = OffsetCurve(
            ParabolaCurve((0, 0, 0), (1, 0, 0), (0, 1, 0), 0.5),
            0.25,
            (0, 0, 1),
        )
    matrix = np.asarray(
        ((-1, 0, 0, 3), (0, 1, 0, 2), (0, 0, 1, 1))
        if reflection
        else ((0, -1, 0, 3), (1, 0, 0, 2), (0, 0, 1, 1)),
        dtype=np.float64,
    )
    placement = RigidPlacement.from_matrix(
        matrix, "conic-placement", reflection=reflection
    )
    placed = place_curve(source, placement, "exact-source-edge")
    parameters = jnp.asarray([-1.0, -0.3, 0.0, 0.6, 1.0], dtype=jnp.float64)
    expected = (
        np.asarray(source.evaluate(parameters)) @ placement.rotation.T
        + placement.translation
    )
    np.testing.assert_allclose(
        placed.evaluate(parameters), expected, rtol=1e-14, atol=1e-14
    )
    parameter = jnp.asarray(0.4, dtype=jnp.float64)
    _, original_jet = jax.jvp(source.evaluate, (parameter,), (jnp.ones_like(parameter),))
    _, placed_jet = jax.jvp(placed.evaluate, (parameter,), (jnp.ones_like(parameter),))
    np.testing.assert_allclose(
        placed_jet,
        np.asarray(original_jet) @ placement.rotation.T,
        rtol=1e-14,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "foreign_vertex", (False, True), ids=("declared-endpoint", "interior-vertex")
)
def test_bounded_closed_source_edge_admits_only_its_declared_domain_endpoints(
    foreign_vertex: bool,
) -> None:
    from phydrax.interchange._cad import edge_range

    result = decode_brep_text_bytes(
        _periodic_circle_text().encode("ascii"),
        _POLICY,
        source_length_unit=METER,
    )
    curve = _require_type(_geometry(result.model).curves[0], BSplineCurve)
    vertex = np.asarray((0, 1, 0) if foreign_vertex else (1, 0, 0), dtype=np.float64)
    if foreign_vertex:
        with pytest.raises(ValueError):
            edge_range(curve, vertex, vertex, True, 1.0e-10)
    else:
        assert edge_range(curve, vertex, vertex, True, 1.0e-10) == (0.0, 4.0)
