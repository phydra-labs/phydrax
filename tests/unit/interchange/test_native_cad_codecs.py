"""Independent STEP, IGES and BRep-text fixtures for native CAD codecs."""

from __future__ import annotations

import builtins
import re
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from types import ModuleType

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
import phydrax.interchange._cad_brep_text as _brep_text_module
from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax._interpolation import TMesh2D
from phydrax._strict import StrictModule
from phydrax.geometry._atlas import TrimDomain
from phydrax.geometry.brep._assembly import BRepAssemblyGraph
from phydrax.geometry.brep._carrier_pieces import (
    carrier_payload as _carrier_payload,
)
from phydrax.geometry.brep._correspondence import exact_source_lift
from phydrax.geometry.brep._intersection import NativePeriodEndpoint
from phydrax.geometry.brep._intersection_curve import CurveTrimSegment, SurfaceRegion
from phydrax.geometry.brep._model import BRepModel
from phydrax.geometry.brep._patches import (
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    CylinderPatch,
    LineCurve,
    OffsetCurve,
    OffsetSurface,
    PlanePatch,
)
from phydrax.geometry.brep._source_correspondence import circle_correspondence
from phydrax.geometry.brep._tspline import TSplineSurfacePatch
from phydrax.interchange._cad import (
    CadCurveFitPolicy,
    CadExportPolicy,
    CadImportPolicy,
    CadInterchangeError,
    CadStage,
    recover_pcurve,
    StagedCoedge,
    validate_export_trims,
)
from phydrax.interchange._cad_brep_text import (
    decode_brep_text_bytes,
    read_brep_text,
    write_brep_text,
)
from phydrax.interchange._iges import (
    decode_iges_bytes,
    read_iges,
    write_iges,
)
from phydrax.interchange._report import AdapterReport
from phydrax.interchange._step import (
    decode_step_bytes,
    read_step,
    write_step,
)
from phydrax.units import METER


_COORDINATES = phx.SpatialCoordinateContract.si()
_POLICY = CadImportPolicy(_COORDINATES)

_STEP_SQUARE = b"""ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('independent planar shell fixture'),'2;1');
FILE_NAME('square.step','1970-01-01T00:00:00',(''),(''),'fixture','native-test','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1=(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#2)) GLOBAL_UNIT_ASSIGNED_CONTEXT((#3,#4,#5)) REPRESENTATION_CONTEXT('',''));
#2=UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-9),#3,'distance_accuracy_value','accuracy');
#3=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.));
#4=(PLANE_ANGLE_UNIT() NAMED_UNIT(*) SI_UNIT($,.RADIAN.));
#5=(NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT());
#6=DIRECTION('',(0.,0.,1.));
#7=DIRECTION('',(1.,0.,0.));
#8=AXIS2_PLACEMENT_3D('',#20,#6,#7);
#9=PLANE('',#8);
#20=CARTESIAN_POINT('',(0.,0.,0.));
#21=CARTESIAN_POINT('',(1.,0.,0.));
#22=CARTESIAN_POINT('',(1.,1.,0.));
#23=CARTESIAN_POINT('',(0.,1.,0.));
#30=DIRECTION('',(1.,0.,0.));
#31=DIRECTION('',(0.,1.,0.));
#32=DIRECTION('',(-1.,0.,0.));
#33=DIRECTION('',(0.,-1.,0.));
#34=VECTOR('',#30,1.);
#35=VECTOR('',#31,1.);
#36=VECTOR('',#32,1.);
#37=VECTOR('',#33,1.);
#38=LINE('',#20,#34);
#39=LINE('',#21,#35);
#40=LINE('',#22,#36);
#41=LINE('',#23,#37);
#42=VERTEX_POINT('',#20);
#43=VERTEX_POINT('',#21);
#44=VERTEX_POINT('',#22);
#45=VERTEX_POINT('',#23);
#46=EDGE_CURVE('',#42,#43,#38,.T.);
#47=EDGE_CURVE('',#43,#44,#39,.T.);
#48=EDGE_CURVE('',#44,#45,#40,.T.);
#49=EDGE_CURVE('',#45,#42,#41,.T.);
#60=ORIENTED_EDGE('',*,*,#46,.T.);
#61=ORIENTED_EDGE('',*,*,#47,.T.);
#62=ORIENTED_EDGE('',*,*,#48,.T.);
#63=ORIENTED_EDGE('',*,*,#49,.T.);
#70=EDGE_LOOP('',(#60,#61,#62,#63));
#71=FACE_OUTER_BOUND('',#70,.T.);
#72=ADVANCED_FACE('',(#71),#9,.T.);
#73=OPEN_SHELL('',(#72));
#74=SHELL_BASED_SURFACE_MODEL('',(#73));
#75=SHAPE_REPRESENTATION('',(#74),#1);
ENDSEC;
END-ISO-10303-21;
"""
_STEP_OFFSET_CURVES = _STEP_SQUARE.replace(
    b"#38=LINE('',#20,#34);",
    b"#90=CARTESIAN_POINT('',(0.,-1.,0.));\n"
    b"#91=LINE('',#90,#34);\n"
    b"#38=OFFSET_CURVE_3D('',#91,1.,.U.,#6);",
).replace(
    b"#46=EDGE_CURVE('',#42,#43,#38,.T.);",
    b"#93=CARTESIAN_POINT('',(0.,-1.));\n"
    b"#94=DIRECTION('',(1.,0.));\n"
    b"#95=VECTOR('',#94,1.);\n"
    b"#96=LINE('',#93,#95);\n"
    b"#97=OFFSET_CURVE_2D('',#96,1.,.U.);\n"
    b"#98=DEFINITIONAL_REPRESENTATION('',(#97),#1);\n"
    b"#99=PCURVE('',#9,#98);\n"
    b"#100=SURFACE_CURVE('',#38,(#99),.PCURVE_S1.);\n"
    b"#46=EDGE_CURVE('',#42,#43,#100,.T.);",
)


def _nested_step_carrier_fixture(kind: str, depth: int, /) -> bytes:
    match kind:
        case "curve":
            target = b"#38=LINE('',#20,#34);"
            base = "#1000=LINE('',#20,#34);"
            template = "#{identifier}=OFFSET_CURVE_3D('',#{basis},0.,.U.,#6);"
            root = 38
        case "surface":
            target = b"#9=PLANE('',#8);"
            base = "#1000=PLANE('',#8);"
            template = "#{identifier}=OFFSET_SURFACE('',#{basis},0.,.U.);"
            root = 9
        case _:
            raise ValueError(f"Unsupported test carrier kind: {kind!r}")
    declarations = [base]
    previous = 1000
    for identifier in range(1001, 1001 + depth):
        declarations.append(template.format(identifier=identifier, basis=previous))
        previous = identifier
    root_record = template.format(identifier=root, basis=previous).encode("ascii")
    replacement = ("\n".join(declarations) + "\n").encode("ascii") + root_record
    return _STEP_SQUARE.replace(target, replacement)


def _hollerith(value: str) -> str:
    return f"{len(value)}H{value}"


def _section(payload: str, name: str) -> list[str]:
    return [
        payload[offset : offset + 72].ljust(72) + f"{name}{sequence:7d}"
        for sequence, offset in enumerate(range(0, len(payload), 72), start=1)
    ] or [" " * 72 + f"{name}{1:7d}"]


def _global_records(values: tuple[str, ...]) -> list[str]:
    records: list[str] = []
    current = ""
    for index, value in enumerate(values):
        token = value + ("," if index + 1 < len(values) else ";")
        if current and len(current) + len(token) > 72:
            records.append(current.ljust(72))
            current = ""
        current += token
    records.append(current.ljust(72))
    return [line + f"G{sequence:7d}" for sequence, line in enumerate(records, start=1)]


def _parameter_chunks(entity_type: int, parameters: tuple[str, ...]) -> list[str]:
    values = (str(entity_type), *parameters)
    chunks: list[str] = []
    current = ""
    for index, value in enumerate(values):
        token = value + ("," if index + 1 < len(values) else ";")
        if current and len(current) + len(token) > 64:
            chunks.append(current)
            current = ""
        current += token
    chunks.append(current)
    return chunks


def _iges_bytes(
    entities: tuple[tuple[int, tuple[str, ...], int], ...],
    /,
    *,
    unit_flag: int = 6,
    unit_name: str = "M",
) -> bytes:
    start = _section("independent planar shell fixture", "S")
    global_values = (
        _hollerith(","),
        _hollerith(";"),
        _hollerith("Phydrax"),
        _hollerith("fixture.igs"),
        _hollerith("Phydrax"),
        _hollerith("native-test"),
        "32",
        "308",
        "15",
        "308",
        "15",
        _hollerith("Phydrax"),
        "1.0",
        str(unit_flag),
        _hollerith(unit_name),
        "1",
        "1.0",
        _hollerith("19700101.000000"),
        "1.0E-7",
        "1.0",
        _hollerith(""),
        _hollerith(""),
        "11",
        "0",
        _hollerith("19700101.000000"),
    )
    global_records = _global_records(global_values)
    directory_records: list[str] = []
    parameter_records: list[str] = []
    numbers = tuple(2 * index + 1 for index in range(len(entities)))
    for index, (entity_type, parameters, form) in enumerate(entities):
        number = numbers[index]
        chunks = _parameter_chunks(entity_type, parameters)
        pointer = len(parameter_records) + 1
        for chunk in chunks:
            sequence = len(parameter_records) + 1
            parameter_records.append(f"{chunk.ljust(64)} {number:7d}P{sequence:7d}")
        first = [entity_type, pointer, 0, 0, 0, 0, 0, 0]
        second = [entity_type, 0, 0, len(chunks), form]
        directory_records.append(
            "".join(f"{value:8d}" for value in first) + "00000000" + f"D{number:7d}"
        )
        directory_records.append(
            "".join(f"{value:8d}" for value in second)
            + " " * 16
            + " " * 8
            + f"{0:8d}"
            + f"D{number + 1:7d}"
        )
    terminate = (
        f"S{len(start):7d}G{len(global_records):7d}"
        f"D{len(directory_records):7d}P{len(parameter_records):7d}"
    ).ljust(72) + "T0000001"
    lines = [
        *start,
        *global_records,
        *directory_records,
        *parameter_records,
        terminate,
    ]
    return ("\n".join(lines) + "\n").encode("ascii")


def _iges_square(
    side: str = "1.",
    /,
    *,
    unit_flag: int = 6,
    unit_name: str = "M",
) -> bytes:
    zero = "0."
    vertices = (
        "4",
        zero,
        zero,
        zero,
        side,
        zero,
        zero,
        side,
        side,
        zero,
        zero,
        side,
        zero,
    )
    line_rows = (
        (zero, zero, zero, side, zero, zero),
        (side, zero, zero, side, side, zero),
        (side, side, zero, zero, side, zero),
        (zero, side, zero, zero, zero, zero),
    )
    line_numbers = (3, 5, 7, 9)
    edge_list_number, plane_number, loop_number, face_number = 11, 13, 15, 17
    edge_list = ["4"]
    for index, line_number in enumerate(line_numbers):
        edge_list.extend(
            (str(line_number), "1", str(index + 1), "1", str((index + 1) % 4 + 1))
        )
    loop = ["4"]
    for index in range(1, 5):
        loop.extend(("0", str(edge_list_number), str(index), "1", "0"))
    entities: tuple[tuple[int, tuple[str, ...], int], ...] = (
        (502, vertices, 0),
        *((110, row, 0) for row in line_rows),
        (504, tuple(edge_list), 0),
        (108, ("0.", "0.", "1.", "0."), 0),
        (508, tuple(loop), 0),
        (510, (str(plane_number), "1", "1", str(loop_number)), 0),
        (514, ("1", str(face_number), "1"), 2),
    )
    return _iges_bytes(entities, unit_flag=unit_flag, unit_name=unit_name)


def _iges_square_with_nested_composite(depth: int, /) -> bytes:
    zero = "0."
    vertices = (
        "4",
        zero,
        zero,
        zero,
        "1.",
        zero,
        zero,
        "1.",
        "1.",
        zero,
        zero,
        "1.",
        zero,
    )
    line_rows = (
        (zero, zero, zero, "1.", zero, zero),
        ("1.", zero, zero, "1.", "1.", zero),
        ("1.", "1.", zero, zero, "1.", zero),
        (zero, "1.", zero, zero, zero, zero),
    )
    entities: list[tuple[int, tuple[str, ...], int]] = [(502, vertices, 0)]
    entities.append((110, line_rows[0], 0))
    curve_pointer = 3
    for _ in range(depth):
        composite_pointer = 2 * len(entities) + 1
        entities.append((102, ("1", str(curve_pointer)), 0))
        curve_pointer = composite_pointer
    line_numbers = [curve_pointer]
    for row in line_rows[1:]:
        line_numbers.append(2 * len(entities) + 1)
        entities.append((110, row, 0))
    edge_list_number = 2 * len(entities) + 1
    edge_list = ["4"]
    for index, line_number in enumerate(line_numbers):
        edge_list.extend(
            (str(line_number), "1", str(index + 1), "1", str((index + 1) % 4 + 1))
        )
    entities.append((504, tuple(edge_list), 0))
    plane_number = 2 * len(entities) + 1
    entities.append((108, ("0.", "0.", "1.", "0."), 0))
    loop_number = 2 * len(entities) + 1
    loop = ["4"]
    for index in range(1, 5):
        loop.extend(("0", str(edge_list_number), str(index), "1", "0"))
    entities.append((508, tuple(loop), 0))
    face_number = 2 * len(entities) + 1
    entities.append((510, (str(plane_number), "1", "1", str(loop_number)), 0))
    entities.append((514, ("1", str(face_number), "1"), 2))
    return _iges_bytes(tuple(entities))


def _iges_nested_empty(
    leaf_name: str = "leaf", root_name: str = "root", root_uses: int = 1
) -> bytes:
    return _iges_bytes(
        (
            (308, ("0", _hollerith(leaf_name), "0"), 0),
            (
                308,
                (
                    "1",
                    _hollerith(root_name),
                    str(root_uses),
                    *("1" for _ in range(root_uses)),
                ),
                0,
            ),
        )
    )


def _brep_text_empty_compound() -> bytes:
    lines = (
        "DBRep_DrawableShape",
        "CASCADE Topology V3, (c) Open Cascade",
        "Locations 0",
        "Curve2ds 0",
        "Curves 0",
        "Polygon3D 0",
        "PolygonOnTriangulations 0",
        "Surfaces 0",
        "Triangulations 0",
        "TShapes 1",
        "Co",
        "0000000",
        "*",
        "+1 0",
    )
    return ("\n".join(lines) + "\n").encode("ascii")


def _shape_counts(model: BRepModel) -> tuple[int, int, int, int]:
    geometry = model.geometry
    assert geometry is not None
    return (
        len(geometry.vertex_points),
        len(geometry.edge_curves),
        len(geometry.face_loops),
        len(geometry.shell_faces),
    )


def _sorted_vertices(model: BRepModel) -> tuple[tuple[float, ...], ...]:
    geometry = model.geometry
    assert geometry is not None
    return tuple(
        sorted(tuple(float(value) for value in point) for point in geometry.vertex_points)
    )


def _planar_square_source() -> BRepModel:
    return decode_step_bytes(_STEP_SQUARE, _POLICY).model


def _native_bspline_model(*, ordinal_pole_ids: bool = False) -> BRepModel:
    source = _planar_square_source()
    geometry = source.geometry
    assert geometry is not None
    exact_controls = tuple(
        tuple(
            (
                Fraction(u, 2),
                Fraction(v, 2),
                Fraction(0),
            )
            for v in range(3)
        )
        for u in range(3)
    )
    knots = (Fraction(0), Fraction(0), Fraction(0), Fraction(1), Fraction(1), Fraction(1))
    patch = BSplineSurfacePatch(
        np.asarray(
            [
                [[float(value) for value in point] for point in row]
                for row in exact_controls
            ],
            dtype=np.float64,
        ),
        np.ones((3, 3), dtype=np.float64),
        np.asarray([float(value) for value in knots]),
        np.asarray([float(value) for value in knots]),
        2,
        2,
        exact_control_points=exact_controls,
        exact_weights=tuple(tuple(Fraction(1) for _ in range(3)) for _ in range(3)),
        exact_u_knots=knots,
        exact_v_knots=knots,
        control_point_ids=np.arange(9).reshape(3, 3) if ordinal_pole_ids else None,
    )
    return BRepModel(
        geometry=geometry,
        patches=(patch,),
        parameter_bounds=np.asarray(source.parameter_bounds),
        orientation=np.asarray(source.orientation),
        physical_tags=source.physical_tags,
        coordinate_contract=_COORDINATES,
        report=source.report,
        trim_domains=source.trim_domains,
    )


def _native_tspline_model(*, refined: bool = False) -> BRepModel:
    source = _planar_square_source()
    geometry = source.geometry
    assert geometry is not None
    u_breaks = (
        (Fraction(0), Fraction(1, 2), Fraction(1))
        if refined
        else (Fraction(0), Fraction(1))
    )
    mesh = TMesh2D.tensor_product(u_breaks, (Fraction(0), Fraction(1)))
    exact_controls = tuple(
        (
            *anchor.local_knots.greville_parameter,
            Fraction(0),
        )
        for anchor in mesh.anchors
    )
    patch = TSplineSurfacePatch(
        mesh,
        np.asarray([[float(value) for value in point] for point in exact_controls]),
        np.ones((mesh.coefficient_count,), dtype=np.float64),
        exact_control_points=exact_controls,
    )
    return BRepModel(
        geometry=geometry,
        patches=(patch,),
        parameter_bounds=np.asarray(source.parameter_bounds),
        orientation=np.asarray(source.orientation),
        physical_tags=source.physical_tags,
        coordinate_contract=_COORDINATES,
        report=source.report,
        trim_domains=source.trim_domains,
    )


type _ExactSurfacePoint = tuple[Fraction, ...]
type _ExactSurfacePoints = tuple[tuple[_ExactSurfacePoint, ...], ...]
type _ExactSurfaceWeights = tuple[tuple[Fraction, ...], ...]


def _tspline_exact_bezier_data(
    patch: TSplineSurfacePatch,
) -> tuple[
    _ExactSurfacePoints,
    _ExactSurfaceWeights,
    tuple[Fraction, ...],
    tuple[Fraction, ...],
]:
    assert len(patch.realization.extractions) == 1
    extraction = patch.realization.extraction_piece(
        patch.realization.extractions[0].piece_id
    )
    homogeneous_controls = patch.realization.exact_homogeneous_piece(
        extraction.piece_id, patch.exact_target_homogeneous
    )
    weights = tuple(value[3] for value in homogeneous_controls)
    points = tuple(
        tuple(
            tuple(
                homogeneous_controls[4 * u + v][axis] / weights[4 * u + v]
                for axis in range(3)
            )
            for v in range(4)
        )
        for u in range(4)
    )
    weights_2d = tuple(tuple(weights[4 * u + v] for v in range(4)) for u in range(4))
    u0, u1, v0, v1 = extraction.parameter_bounds
    return (
        points,
        weights_2d,
        (u0,) * 4 + (u1,) * 4,
        (v0,) * 4 + (v1,) * 4,
    )


def _assert_geometry_topology_equal(expected: BRepModel, actual: BRepModel) -> None:
    first, second = expected.geometry, actual.geometry
    assert first is not None and second is not None
    assert _shape_counts(expected) == _shape_counts(actual)
    for name in ("vertex_points", "edge_ranges"):
        assert np.array_equal(
            np.asarray(getattr(first, name)), np.asarray(getattr(second, name))
        )
    for name in (
        "edge_curves",
        "edge_vertices",
        "coedge_edges",
        "coedge_senses",
        "face_loops",
        "shell_faces",
        "shell_orientations",
        "solid_shells",
    ):
        assert getattr(first, name) == getattr(second, name)


def _assert_geometry_preserved(
    expected: BRepModel,
    actual: BRepModel,
    /,
    *,
    iges_line_pcurves: bool = False,
    allow_physical_tag_loss: bool = False,
) -> None:
    _assert_geometry_topology_equal(expected, actual)
    first, second = expected.geometry, actual.geometry
    assert first is not None and second is not None
    for name in ("curves", "pcurves"):
        expected_carriers = getattr(first, name)
        actual_carriers = getattr(second, name)
        if name == "pcurves" and iges_line_pcurves:
            ranges = np.asarray(first.edge_ranges)
            for coedge, (source_curve, restored_curve) in enumerate(
                zip(expected_carriers, actual_carriers, strict=True)
            ):
                if isinstance(source_curve, LineCurve) and isinstance(
                    restored_curve, BSplineCurve
                ):
                    edge = first.coedge_edges[coedge]
                    lower, upper = ranges[edge]
                    expected_points = np.asarray(
                        source_curve.evaluate(
                            jnp.asarray((lower, upper), dtype=jnp.float64)
                        )
                    )
                    assert restored_curve.degree == 1
                    assert np.array_equal(
                        np.asarray(restored_curve.control_points), expected_points
                    )
                    assert np.array_equal(
                        np.asarray(restored_curve.weights), np.ones((2,))
                    )
                    assert np.array_equal(
                        np.asarray(restored_curve.knots),
                        np.asarray((lower, lower, upper, upper)),
                    )
                else:
                    assert canonical_fingerprint(
                        _carrier_payload(source_curve)
                    ) == canonical_fingerprint(_carrier_payload(restored_curve))
                    assert array_tree_fingerprint(source_curve) == array_tree_fingerprint(
                        restored_curve
                    )
        else:
            assert tuple(
                canonical_fingerprint(_carrier_payload(value))
                for value in expected_carriers
            ) == tuple(
                canonical_fingerprint(_carrier_payload(value))
                for value in actual_carriers
            )
            assert tuple(
                array_tree_fingerprint(value) for value in expected_carriers
            ) == tuple(array_tree_fingerprint(value) for value in actual_carriers)
    assert tuple(
        canonical_fingerprint(_carrier_payload(value)) for value in expected.patches
    ) == tuple(canonical_fingerprint(_carrier_payload(value)) for value in actual.patches)
    assert tuple(array_tree_fingerprint(value) for value in expected.patches) == tuple(
        array_tree_fingerprint(value) for value in actual.patches
    )
    assert np.array_equal(
        np.asarray(expected.parameter_bounds), np.asarray(actual.parameter_bounds)
    )
    assert np.array_equal(
        np.asarray(expected.orientation), np.asarray(actual.orientation)
    )
    if not allow_physical_tag_loss:
        assert expected.physical_tags == actual.physical_tags


def _assert_source_identifier_loss(report: AdapterReport) -> None:
    assert any(
        loss.changes_interpretation and "source identifiers" in loss.rationale.lower()
        for loss in report.losses
    )


def _assert_shared_empty_definitions(graph: BRepAssemblyGraph) -> None:
    assert len(graph.definitions) == 2
    assert len(graph.roots) == 1
    root = graph.roots[0]
    assert root.definition_id is not None
    definitions = {value.definition_id: value for value in graph.definitions}
    root_definition = definitions[root.definition_id]
    assert len(root_definition.uses) == 2
    children = tuple(use.definition_id for use in root_definition.uses)
    assert children[0] is not None and children[0] == children[1]
    assert len({use.use_id for use in root_definition.uses}) == 2
    assert definitions[children[0]].uses == ()
    assert tuple(graph.iter_occurrences()) == ()


def _assert_shared_empty_definition_dag(graph: BRepAssemblyGraph) -> None:
    definitions = {value.definition_id: value for value in graph.definitions}
    assert len(definitions) >= 2
    assert len(graph.roots) == 1
    assert graph.roots[0].solid_index is None
    assert graph.roots[0].definition_id is not None
    references: dict[str, int] = {}
    for definition in graph.definitions:
        for use in definition.uses:
            assert use.solid_index is None
            if use.definition_id is not None:
                references[use.definition_id] = references.get(use.definition_id, 0) + 1
    assert any(count > 1 for count in references.values())
    pending = [graph.roots[0].definition_id]
    reachable: set[str] = set()
    while pending:
        definition_id = pending.pop()
        if definition_id in reachable:
            continue
        reachable.add(definition_id)
        pending.extend(
            use.definition_id
            for use in definitions[definition_id].uses
            if use.definition_id is not None
        )
    assert reachable == set(definitions)
    assert tuple(graph.iter_occurrences()) == ()


def _forbid_optional_occt(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = None,
        level: int = 0,
    ) -> ModuleType:
        if name == "OCP" or name.startswith("OCP."):
            raise AssertionError(
                "Native CAD codecs must not import the optional OCCT engine."
            )
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_independent_step_and_iges_fixtures_use_native_codecs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_optional_occt(monkeypatch)
    step = decode_step_bytes(_STEP_SQUARE, _POLICY)
    iges = decode_iges_bytes(_iges_square(), _POLICY)
    assert _shape_counts(step.model) == (4, 4, 1, 1)
    assert _shape_counts(iges.model) == (4, 4, 1, 1)
    assert step.report.valid and iges.report.valid
    assert step.model.report.source_format == "step"
    assert iges.model.report.source_format == "iges"
    # The fixture supplies no p-curves: each of the 4 coedges is recovered once.
    assert (
        step.coverage.pcurves_from_file,
        step.coverage.pcurves_recovered,
        step.coverage.pcurves_absent,
    ) == (0, 4, 0)
    for imported in (step, iges):
        geometry = imported.model.geometry
        assert geometry is not None
        coverage = imported.coverage
        assert coverage.pcurves_from_file + coverage.pcurves_recovered + (
            coverage.pcurves_absent
        ) == len(geometry.coedge_edges)
    renamed_source = _STEP_SQUARE.replace(b"square.step", b"renamed.step")
    renamed_step = decode_step_bytes(renamed_source, _POLICY)
    source_geometry = step.model.geometry
    renamed_geometry = renamed_step.model.geometry
    assert source_geometry is not None and renamed_geometry is not None
    assert source_geometry.assembly_graph is not None
    assert renamed_geometry.assembly_graph is not None
    assert (
        source_geometry.assembly_graph.identity_payload()
        == renamed_geometry.assembly_graph.identity_payload()
    )
    _assert_geometry_preserved(step.model, renamed_step.model)
    renumbered_step = decode_step_bytes(
        re.sub(
            rb"#([0-9]+)",
            lambda match: f"#{int(match.group(1)) + 1000}".encode(),
            _STEP_SQUARE,
        ),
        _POLICY,
    )
    _assert_geometry_preserved(step.model, renumbered_step.model)
    millimeters = decode_iges_bytes(
        _iges_square("1000.", unit_flag=2, unit_name="MM"), _POLICY
    )
    assert millimeters.source_length_unit_meters == pytest.approx(1.0e-3)
    assert max(
        value for point in _sorted_vertices(millimeters.model) for value in point
    ) == pytest.approx(1.0)


def test_source_curve_trim_domains_validate_native_coedge_carriers() -> None:
    model = decode_step_bytes(_STEP_SQUARE, _POLICY).model
    geometry = model.geometry
    assert geometry is not None
    ranges = geometry.edge_ranges
    domains = []
    for face_loops in geometry.face_loops:
        curve_loops = []
        for coedges in face_loops:
            segments = []
            for coedge in coedges:
                edge = geometry.coedge_edges[coedge]
                first, last = (float(value) for value in ranges[edge])
                first_root, last_root = geometry.coedge_endpoint_roots[coedge]
                segments.append(
                    CurveTrimSegment(
                        geometry.pcurves[coedge],
                        first,
                        last,
                        reversed=geometry.coedge_senses[coedge] < 0,
                        first_root=first_root,
                        last_root=last_root,
                    )
                )
            curve_loops.append(tuple(segments))
        domains.append(TrimDomain.from_curve_loops(tuple(curve_loops)))
    validate_export_trims(model, geometry, "native-test", trim_domains=tuple(domains))


def test_step_offset_curves_use_native_signed_distance_and_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_optional_occt(monkeypatch)
    imported = decode_step_bytes(_STEP_OFFSET_CURVES, _POLICY)
    geometry = imported.model.geometry
    assert geometry is not None
    edge_offsets = tuple(
        curve for curve in geometry.curves if isinstance(curve, OffsetCurve)
    )
    pcurve_offsets = tuple(
        curve for curve in geometry.pcurves if isinstance(curve, OffsetCurve)
    )
    assert len(edge_offsets) == len(pcurve_offsets) == 1
    assert edge_offsets[0].distance_exact == pcurve_offsets[0].distance_exact == -1
    expected = _shape_counts(imported.model)
    assert expected == (4, 4, 1, 1)

    output = tmp_path / "offset.step"
    exported = write_step(imported.model, output)
    payload = output.read_bytes()
    restored = read_step(output, _POLICY, trusted_root=tmp_path)
    restored_geometry = restored.model.geometry
    assert restored_geometry is not None
    restored_edge_offsets = tuple(
        curve for curve in restored_geometry.curves if isinstance(curve, OffsetCurve)
    )
    restored_pcurve_offsets = tuple(
        curve for curve in restored_geometry.pcurves if isinstance(curve, OffsetCurve)
    )
    assert b"OFFSET_CURVE_2D" in payload
    assert b"OFFSET_CURVE_3D" in payload
    assert exported.report.valid and restored.report.valid
    assert len(restored_edge_offsets) == len(restored_pcurve_offsets) == 1
    assert (
        restored_edge_offsets[0].distance_exact
        == restored_pcurve_offsets[0].distance_exact
        == -1
    )
    assert _shape_counts(restored.model) == expected
    assert _sorted_vertices(restored.model) == _sorted_vertices(imported.model)
    brep_path = tmp_path / "offset.brep"
    brep_export = write_brep_text(imported.model, brep_path)
    brep_restored = read_brep_text(
        brep_path,
        _POLICY,
        trusted_root=tmp_path,
        source_length_unit=METER,
    )
    brep_geometry = brep_restored.model.geometry
    assert brep_geometry is not None
    brep_edge_offsets = tuple(
        curve for curve in brep_geometry.curves if isinstance(curve, OffsetCurve)
    )
    brep_pcurve_offsets = tuple(
        curve for curve in brep_geometry.pcurves if isinstance(curve, OffsetCurve)
    )
    assert brep_export.report.valid and brep_restored.report.valid
    assert len(brep_edge_offsets) == len(brep_pcurve_offsets) == 1
    assert (
        brep_edge_offsets[0].distance_exact == brep_pcurve_offsets[0].distance_exact == -1
    )
    assert _shape_counts(brep_restored.model) == expected
    assert _sorted_vertices(brep_restored.model) == _sorted_vertices(imported.model)


# A planar annulus: the outer circle turns counterclockwise in the plane's
# parameters and the hole's circle (axis -z) clockwise, so one plane carries
# conic p-curves of both senses.
_STEP_ANNULUS = b"""ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('independent planar annulus fixture'),'2;1');
FILE_NAME('annulus.step','1970-01-01T00:00:00',(''),(''),'fixture','native-test','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1=(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#2)) GLOBAL_UNIT_ASSIGNED_CONTEXT((#3,#4,#5)) REPRESENTATION_CONTEXT('',''));
#2=UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-9),#3,'distance_accuracy_value','accuracy');
#3=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.));
#4=(PLANE_ANGLE_UNIT() NAMED_UNIT(*) SI_UNIT($,.RADIAN.));
#5=(NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT());
#6=DIRECTION('',(0.,0.,1.));
#7=DIRECTION('',(1.,0.,0.));
#10=DIRECTION('',(0.,0.,-1.));
#20=CARTESIAN_POINT('',(0.,0.,0.));
#21=CARTESIAN_POINT('',(2.,0.,0.));
#22=CARTESIAN_POINT('',(1.,0.,0.));
#8=AXIS2_PLACEMENT_3D('',#20,#6,#7);
#9=PLANE('',#8);
#11=AXIS2_PLACEMENT_3D('',#20,#10,#7);
#30=CIRCLE('',#8,2.);
#31=CIRCLE('',#11,1.);
#42=VERTEX_POINT('',#21);
#43=VERTEX_POINT('',#22);
#46=EDGE_CURVE('',#42,#42,#30,.T.);
#47=EDGE_CURVE('',#43,#43,#31,.T.);
#60=ORIENTED_EDGE('',*,*,#46,.T.);
#61=ORIENTED_EDGE('',*,*,#47,.T.);
#70=EDGE_LOOP('',(#60));
#71=EDGE_LOOP('',(#61));
#72=FACE_OUTER_BOUND('',#70,.T.);
#73=FACE_BOUND('',#71,.T.);
#74=ADVANCED_FACE('',(#72,#73),#9,.T.);
#75=OPEN_SHELL('',(#74));
#76=SHELL_BASED_SURFACE_MODEL('',(#75));
#77=SHAPE_REPRESENTATION('',(#76),#1);
ENDSEC;
END-ISO-10303-21;
"""


@pytest.mark.parametrize(
    "shared", (True, False), ids=("shared", "independent-coincident")
)
def test_independent_split_conic_loop_retains_native_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shared: bool
) -> None:
    _forbid_optional_occt(monkeypatch)
    # The two EDGE_CURVEs share a source circle and close through the SAME
    # topological vertices. The decimal pi split remains an ordinary trim.
    fixture = _STEP_ANNULUS.replace(
        b"#46=EDGE_CURVE('',#42,#42,#30,.T.);",
        b"#23=CARTESIAN_POINT('',(-2.,0.,0.));\n"
        b"#44=VERTEX_POINT('',#23);\n"
        b"#32=TRIMMED_CURVE('',#30,(PARAMETER_VALUE(0.)),"
        b"(PARAMETER_VALUE(3.141592653589793)),.T.,.PARAMETER.);\n"
        b"#33=TRIMMED_CURVE('',#30,(PARAMETER_VALUE(3.141592653589793)),"
        b"(PARAMETER_VALUE(6.283185307179586)),.T.,.PARAMETER.);\n"
        b"#46=EDGE_CURVE('',#42,#44,#32,.T.);\n"
        b"#48=EDGE_CURVE('',#44,#42,#33,.T.);\n"
        b"#62=ORIENTED_EDGE('',*,*,#48,.T.);",
    ).replace(b"#70=EDGE_LOOP('',(#60));", b"#70=EDGE_LOOP('',(#60,#62));")
    if not shared:
        fixture = fixture.replace(
            b"#33=TRIMMED_CURVE('',#30,",
            b"#34=CIRCLE('',#8,2.);\n#33=TRIMMED_CURVE('',#34,",
        )
    source = decode_step_bytes(fixture, _POLICY)
    path = tmp_path / "split-annulus.step"
    write_step(source.model, path)
    restored = read_step(path, _POLICY, trusted_root=tmp_path)
    for imported in (source, restored):
        geometry = imported.model.geometry
        assert geometry is not None
        loops = geometry.face_loops[0]
        outer = loops[0]
        assert len(outer) == 2
        first_edge, last_edge = (geometry.coedge_edges[coedge] for coedge in outer)
        np.testing.assert_array_equal(geometry.edge_ranges[first_edge], (0.0, np.pi))
        np.testing.assert_array_equal(
            geometry.edge_ranges[last_edge], (np.pi, 2.0 * np.pi)
        )
        first_root, split_left = geometry.coedge_endpoint_roots[outer[0]]
        split_right, last_root = geometry.coedge_endpoint_roots[outer[1]]
        assert (
            geometry.edge_curves[first_edge] == geometry.edge_curves[last_edge]
        ) is shared
        if not shared:
            assert not isinstance(first_root, NativePeriodEndpoint)
            assert not isinstance(last_root, NativePeriodEndpoint)
            continue
        assert isinstance(first_root, NativePeriodEndpoint) and first_root.turns == 0
        assert isinstance(last_root, NativePeriodEndpoint) and last_root.turns == 1
        assert split_left is None and split_right is None
        curve_loops = tuple(
            tuple(
                CurveTrimSegment(
                    geometry.pcurves[coedge],
                    *map(float, geometry.edge_ranges[geometry.coedge_edges[coedge]]),
                    reversed=geometry.coedge_senses[coedge] < 0,
                    first_root=geometry.coedge_endpoint_roots[coedge][0],
                    last_root=geometry.coedge_endpoint_roots[coedge][1],
                )
                for coedge in loop
            )
            for loop in loops
        )
        decision = TrimDomain.from_curve_loops(curve_loops).classify(
            np.asarray(((1.5, 0.0), (0.0, 0.0), (3.0, 0.0)))
        )
        np.testing.assert_array_equal(decision.resolved, (True, True, True))
        np.testing.assert_array_equal(decision.inside, (True, False, False))


@pytest.mark.parametrize(
    "case",
    (
        "closed",
        "reversed",
        "open-incidence",
        "decimal-end",
        "gap",
        "different-carrier",
        "different-source",
        "missing-source",
        "mixed-senses",
        "partial",
        "multiple-turns",
    ),
)
def test_split_conic_turn_requires_whole_authored_loop(case: str) -> None:
    circle = CircleCurve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 2.0)
    stage = CadStage(1.0, None)
    stage.curves = [circle, circle]
    stage.edge_curves = [0, 0]
    # An arbitrary decimal split is not promoted to a symbolic phase.
    stage.edge_ranges = [(0.0, 0.7), (0.7, 2.0 * np.pi)]
    stage.edge_vertices = [(0, 1), (1, 0)]
    senses = (-1, -1) if case == "reversed" else (1, 1)
    if case == "open-incidence":
        stage.edge_vertices[1] = (1, 2)
    elif case == "decimal-end":
        stage.edge_ranges[1] = (0.7, np.nextafter(2.0 * np.pi, np.inf))
    elif case == "gap":
        stage.edge_ranges[1] = (np.nextafter(0.7, np.inf), 2.0 * np.pi)
    elif case == "different-carrier":
        stage.curves[1] = CircleCurve(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1.0
        )
        stage.edge_curves[1] = 1
    elif case == "partial":
        stage.edge_ranges[1] = (0.7, np.pi)
    elif case == "multiple-turns":
        stage.edge_ranges[1] = (0.7, 4.0 * np.pi)
    elif case == "different-source":
        stage.edge_curves[1] = 1
    elif case == "missing-source":
        stage.edge_curves[1] = -1
    elif case == "mixed-senses":
        senses = (1, -1)
    endpoints = stage._closed_conic_loop_endpoints((0, 1), senses, (((0, 1),),))
    assert endpoints == (
        {(0, 0): (Fraction(0), 0), (1, 1): (Fraction(0), 1)}
        if case in ("closed", "reversed")
        else {}
    )


def _plane_conic_turns(model: BRepModel) -> list[int]:
    geometry = model.geometry
    assert geometry is not None
    turns: list[int] = []
    for face, loops in enumerate(geometry.face_loops):
        assert isinstance(model.patches[face], PlanePatch)
        for loop in loops:
            for coedge in loop:
                curve = geometry.pcurves[coedge]
                assert isinstance(curve, CircleCurve)
                first = np.asarray(curve.first_axis)
                second = np.asarray(curve.second_axis)
                turns.append(int(np.sign(first[0] * second[1] - first[1] * second[0])))
    return sorted(turns)


@pytest.mark.parametrize("format_name", ("step", "iges", "brep"))
def test_plane_conic_pcurves_of_both_senses_round_trip_within_certified_gaps(
    format_name: str, tmp_path: Path
) -> None:
    policy = CadImportPolicy(_COORDINATES, pcurve_fit=None)
    source = decode_step_bytes(_STEP_ANNULUS, policy)
    assert _plane_conic_turns(source.model) == [-1, 1]
    path = tmp_path / f"annulus.{format_name}"
    match format_name:
        case "step":
            write_step(source.model, path)
            restored = read_step(path, policy, trusted_root=tmp_path)
        case "iges":
            write_iges(source.model, path)
            restored = read_iges(path, policy, trusted_root=tmp_path)
        case _:
            write_brep_text(source.model, path)
            restored = read_brep_text(
                path, policy, trusted_root=tmp_path, source_length_unit=METER
            )
    geometry = restored.model.geometry
    assert geometry is not None
    assert restored.coverage.pcurves_from_file == len(geometry.coedge_edges) == 2
    assert _plane_conic_turns(restored.model) == [-1, 1]
    evidence = restored.tolerance_evidence
    assert evidence is not None
    for entity, certified, declared in zip(
        evidence.entities, evidence.certified_bounds, evidence.source_bounds, strict=True
    ):
        if entity.kind == "edge":
            assert certified is not None
        if certified is not None:
            limit = evidence.declared_limit if declared is None else declared
            assert certified <= limit, (entity, certified, limit)


def test_cad_stage_retains_proved_face_gaps_and_refuses_tighter_requirement() -> None:
    source = decode_step_bytes(_STEP_SQUARE, _POLICY).model
    geometry = source.geometry
    gap = 2.0**-30
    points = np.asarray(geometry.vertex_points, dtype=np.float64) + np.asarray(
        (0.0, 0.0, gap), dtype=np.float64
    )
    stage = CadStage(1.0, None, relative_geometric_tolerance=2.0 * gap)
    for index in range(points.shape[0]):
        stage.vertex(points[index : index + 1, :].reshape(3), f"vertex-{index}")
    for edge, (start, end) in enumerate(geometry.edge_vertices):
        stage.edge(
            LineCurve(points[start], points[end] - points[start]),
            start,
            end,
            f"edge-{edge}",
            parameter_range=(0.0, 1.0),
        )
    for face, loops in enumerate(geometry.face_loops):
        uses = [
            [
                StagedCoedge(
                    geometry.coedge_edges[coedge],
                    geometry.coedge_senses[coedge],
                    geometry.pcurves[coedge],
                    f"coedge-{coedge}",
                    origin="file",
                )
                for coedge in loop
            ]
            for loop in loops
        ]
        stage.face(
            source.patches[face],
            uses,
            int(np.asarray(source.orientation)[face]),
            source.physical_tags[face],
            f"face-{face}",
            outer_known=True,
        )
    stage.shells = [(list(faces), False) for faces in geometry.shell_faces]
    stage.shell_orientations = {
        index: list(signs) for index, signs in enumerate(geometry.shell_orientations)
    }
    stage.solids = [list(shells) for shells in geometry.solid_shells]
    source_digest = canonical_fingerprint(array_tree_fingerprint(points))
    published = stage.publish(
        coordinate_contract=_COORDINATES,
        source_id="bounded-face-gap",
        source_format="step",
        source_digest=source_digest,
        import_policy_id=canonical_fingerprint({"maximum_gap": 2.0 * gap}),
        tessellation=None,
    )
    published.require_current()
    bounds = published.geometry.edge_face_gap_bounds
    assert bounds is not None and len(bounds) == len(geometry.coedge_edges)
    assert all(gap <= bound <= 2.0 * gap for bound in bounds)

    stage.relative_geometric_tolerance = 0.5 * gap
    with pytest.raises(CadInterchangeError):
        stage.publish(
            coordinate_contract=_COORDINATES,
            source_id="bounded-face-gap",
            source_format="step",
            source_digest=source_digest,
            import_policy_id=canonical_fingerprint({"maximum_gap": 0.5 * gap}),
            tessellation=None,
        )
    published.require_current()


def test_native_writers_round_trip_exact_planar_topology(tmp_path: Path) -> None:
    model = decode_step_bytes(_STEP_SQUARE, _POLICY).model
    expected = _shape_counts(model)

    step_path = tmp_path / "square.step"
    step_result = write_step(model, step_path)
    step_restored = read_step(step_path, _POLICY, trusted_root=tmp_path)
    step_text = step_path.read_bytes()
    assert b"phydrax:container:" not in step_text
    assert b"phydrax:occurrence:" not in step_text
    assert b"phydrax:definition:" not in step_text
    assert step_result.path == step_path
    assert step_result.report.valid
    _assert_geometry_preserved(model, step_restored.model)
    assert _shape_counts(step_restored.model) == expected
    assert _sorted_vertices(step_restored.model) == _sorted_vertices(model)

    iges_path = tmp_path / "square.iges"
    iges_result = write_iges(model, iges_path)
    iges_restored = read_iges(iges_path, _POLICY, trusted_root=tmp_path)
    assert iges_result.path == iges_path
    assert iges_result.report.valid
    _assert_geometry_preserved(model, iges_restored.model, iges_line_pcurves=True)
    assert _shape_counts(iges_restored.model) == expected
    assert _sorted_vertices(iges_restored.model) == _sorted_vertices(model)

    brep_path = tmp_path / "square.brep"
    brep_result = write_brep_text(model, brep_path)
    brep_restored = read_brep_text(
        brep_path,
        _POLICY,
        trusted_root=tmp_path,
        source_length_unit=METER,
    )
    assert brep_result.path == brep_path
    assert brep_result.report.valid
    _assert_geometry_preserved(model, brep_restored.model)
    assert _shape_counts(brep_restored.model) == expected
    assert _sorted_vertices(brep_restored.model) == _sorted_vertices(model)


def _solid_cylinder_residuals(
    points: np.ndarray,
    radius: float,
    height: float,
    base: tuple[float, float, float],
    axis: tuple[float, float, float],
    /,
) -> np.ndarray:
    """Distance of each point from the authored solid cylinder boundary."""
    relative = points - np.asarray(base)
    axial = relative @ np.asarray(axis)
    radial = np.linalg.norm(relative - axial[:, None] * np.asarray(axis), axis=1)
    lateral = (
        np.abs(radial - radius)
        + np.maximum(0.0, -axial)
        + np.maximum(0.0, axial - height)
    )
    caps = np.minimum(np.abs(axial), np.abs(axial - height)) + np.maximum(
        0.0, radial - radius
    )
    return np.minimum(lateral, caps)


@pytest.mark.parametrize(
    ("radius", "height", "base", "axis"),
    (
        (1.0, 2.4, (0.0, 0.0, -1.2), (0.0, 0.0, 1.0)),
        (0.6, 3.0, (-1.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ),
    ids=("vertical", "transverse"),
)
@pytest.mark.parametrize("format_name", ("step", "iges", "brep"))
def test_native_cylinder_round_trips_with_exact_seams_and_poles(
    tmp_path: Path,
    radius: float,
    height: float,
    base: tuple[float, float, float],
    axis: tuple[float, float, float],
    format_name: str,
) -> None:
    # Whole-turn rings, the seam used on both chart sheets and the collapsed
    # cap poles must import from their exact period evidence, with every
    # coedge lift proved exact (no published face-gap bound).
    model = phx.geometry.brep.brep_cylinder(
        radius,
        height,
        base_center=base,
        axis=axis,
        coordinate_contract=_COORDINATES,
        source_id=f"cylinder-round-trip:{format_name}",
    )
    path = tmp_path / f"cylinder.{format_name}"
    match format_name:
        case "step":
            exported = write_step(model, path)
            restored = read_step(path, _POLICY, trusted_root=tmp_path)
        case "iges":
            exported = write_iges(model, path)
            restored = read_iges(path, _POLICY, trusted_root=tmp_path)
        case _:
            exported = write_brep_text(model, path)
            restored = read_brep_text(
                path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
            )

    assert exported.report.valid
    assert restored.report.status.name == "LOSSLESS"
    imported = restored.model
    source, geometry = model.geometry, imported.geometry
    assert source is not None and geometry is not None
    assert geometry.edge_face_gap_bounds is None
    before, after = model.topology, imported.topology
    assert (after.num_solids, after.num_faces, after.num_edges, after.num_vertices) == (
        before.num_solids,
        before.num_faces,
        before.num_edges,
        before.num_vertices,
    )
    assert _sorted_vertices(imported) == _sorted_vertices(model)
    assert sorted(
        type(geometry.curves[index]).__name__
        for index in geometry.edge_curves
        if index >= 0
    ) == sorted(
        type(source.curves[index]).__name__ for index in source.edge_curves if index >= 0
    )
    sample = np.linspace(0.0, 1.0, 7)
    for patch, box in zip(imported.patches, np.asarray(imported.parameter_bounds)):
        grid = np.stack(np.meshgrid(sample, sample, indexing="ij"), axis=-1).reshape(
            -1, 2
        )
        uv = box[0] + grid * (box[1] - box[0])
        points = np.asarray(patch.evaluate(jnp.asarray(uv)), dtype=np.float64)
        assert np.max(
            _solid_cylinder_residuals(points, radius, height, base, axis)
        ) <= 8.0 * np.finfo(np.float64).eps * (radius + height)
    for index, (first, last) in zip(
        geometry.edge_curves, np.asarray(geometry.edge_ranges), strict=True
    ):
        if index < 0:
            continue
        points = np.asarray(
            geometry.curves[index].evaluate(jnp.asarray(first + sample * (last - first))),
            dtype=np.float64,
        )
        assert np.max(
            _solid_cylinder_residuals(points, radius, height, base, axis)
        ) <= 8.0 * np.finfo(np.float64).eps * (radius + height)


def test_nist_washer_circles_recover_exact_cylinder_and_certified_plane_pcurves() -> None:
    # Carriers from NIST FTC-11 (#72/#79 and #100/#107). The bore circle is an
    # exact height section of its cylinder; the outer circle sits in a plane
    # parallel to its face, decimal-offset by ~1e-15, which is a certified gap.
    cylinder = CylinderPatch(
        (0.0, 0.0, -0.75), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 16.0
    )
    bore = CircleCurve(
        (0.0, 0.0, -1.499999999999999), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0), 16.0
    )
    pcurve = recover_pcurve(cylinder, bore, None)
    assert isinstance(pcurve, LineCurve)
    height = Fraction(-1.499999999999999) - Fraction(-0.75)
    assert pcurve.origin_exact == (Fraction(0), height)
    assert pcurve.direction_exact == (Fraction(1), Fraction(0))
    region = SurfaceRegion(cylinder, np.asarray(((0.0, -1.0), (2.0 * np.pi, 0.0))))
    assert exact_source_lift(bore, pcurve, region, 0.0, 2.0 * np.pi)
    rotated = CircleCurve((0.0, 0.0, -1.5), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 16.0)
    with pytest.raises(ValueError, match="No exact p-curve source map"):
        recover_pcurve(cylinder, rotated, None)

    plane = PlanePatch((0.0, -23.75, -1.5), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    outer = CircleCurve(
        (0.0, 0.0, -1.500000000000001), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0), 31.5
    )
    chart = recover_pcurve(plane, outer, None)
    assert isinstance(chart, CircleCurve)
    assert chart.center_exact == (Fraction(0), Fraction(23.75))
    assert chart.first_axis_exact == (Fraction(0), Fraction(-1))
    assert chart.second_axis_exact == (Fraction(-1), Fraction(0))
    assert chart.radius_exact == Fraction(31.5)
    gap = circle_correspondence(outer, chart, plane)
    assert gap is not None
    assert Fraction(gap) >= abs(Fraction(-1.500000000000001) - Fraction(-1.5))
    assert gap <= 2.0 * float(abs(Fraction(-1.500000000000001) - Fraction(-1.5)))
    tilted = CircleCurve((0.0, 0.0, -1.5), (0.0, -0.6, 0.8), (1.0, 0.0, 0.0), 31.5)
    with pytest.raises(ValueError, match="No exact p-curve source map"):
        recover_pcurve(plane, tilted, None)


def test_exact_bspline_surface_round_trips_through_native_codecs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_optional_occt(monkeypatch)
    model = _native_bspline_model()

    step_path = tmp_path / "exact-spline.step"
    step_export = write_step(model, step_path)
    step_restored = read_step(step_path, _POLICY, trusted_root=tmp_path)
    assert step_export.report.valid
    _assert_geometry_preserved(model, step_restored.model)

    iges_path = tmp_path / "exact-spline.iges"
    iges_export = write_iges(model, iges_path)
    iges_restored = read_iges(iges_path, _POLICY, trusted_root=tmp_path)
    assert iges_export.report.valid
    assert any(
        loss.path.endswith(".pcurve-carrier") and "degree-one B-spline" in loss.rationale
        for loss in iges_export.report.losses
    )
    _assert_geometry_preserved(model, iges_restored.model, iges_line_pcurves=True)

    brep_path = tmp_path / "exact-spline.brep"
    brep_export = write_brep_text(model, brep_path)
    brep_restored = read_brep_text(
        brep_path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert brep_export.report.valid
    # BRep text declares each pole's identity as its record ordinal.
    _assert_geometry_preserved(
        _native_bspline_model(ordinal_pole_ids=True),
        brep_restored.model,
        allow_physical_tag_loss=True,
    )
    assert any(loss.path == "face:0.physical-tag" for loss in brep_export.report.losses)
    # The source declares no pole identities: STEP and IGES re-import none, while
    # BRep text declares record ordinals and the export says so.
    for report in (step_export.report, iges_export.report):
        assert not any(loss.path.endswith(".pole-identity") for loss in report.losses)
    assert any(
        loss.path == "face:0.pole-identity" and loss.changes_interpretation
        for loss in brep_export.report.losses
    )

    identified = _native_bspline_model(ordinal_pole_ids=True)
    identified_brep = write_brep_text(identified, tmp_path / "identified.brep")
    assert not any(
        loss.path.endswith(".pole-identity") for loss in identified_brep.report.losses
    )
    for writer, name in ((write_step, "step"), (write_iges, "iges")):
        export = writer(identified, tmp_path / f"identified.{name}")
        assert any(
            loss.path == "face:0.pole-identity" and "does not encode" in loss.rationale
            for loss in export.report.losses
        )


def test_physical_tags_use_standard_fields_or_declare_loss(
    tmp_path: Path,
) -> None:
    source = decode_step_bytes(_STEP_SQUARE, _POLICY).model
    geometry = source.geometry
    assert geometry is not None
    source_tag = "authored-face-tag"
    model = BRepModel(
        geometry=geometry,
        patches=source.patches,
        parameter_bounds=np.asarray(source.parameter_bounds),
        orientation=np.asarray(source.orientation),
        physical_tags=(source_tag,),
        coordinate_contract=source.coordinate_contract,
        report=source.report,
        trim_domains=source.trim_domains,
    )

    step_path = tmp_path / "physical-tag.step"
    step_export = write_step(model, step_path)
    step_restored = read_step(step_path, _POLICY, trusted_root=tmp_path)
    assert step_export.report.valid and step_restored.report.valid
    _assert_geometry_topology_equal(model, step_restored.model)
    assert step_restored.model.physical_tags == (source_tag,)
    assert "physical tags" in step_export.report.preserved_fields

    iges_path = tmp_path / "physical-tag.iges"
    iges_export = write_iges(model, iges_path)
    iges_restored = read_iges(iges_path, _POLICY, trusted_root=tmp_path)
    assert iges_export.report.valid and iges_restored.report.valid
    _assert_geometry_topology_equal(model, iges_restored.model)
    assert iges_restored.model.physical_tags == ("plane",)
    assert any(loss.path == "face:0.physical-tag" for loss in iges_export.report.losses)

    brep_path = tmp_path / "physical-tag.brep"
    brep_export = write_brep_text(model, brep_path)
    brep_restored = read_brep_text(
        brep_path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert brep_export.report.valid and brep_restored.report.valid
    _assert_geometry_topology_equal(model, brep_restored.model)
    assert brep_restored.model.physical_tags == ("plane",)
    assert any(loss.path == "face:0.physical-tag" for loss in brep_export.report.losses)


def test_exact_single_piece_tspline_export_and_multi_piece_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_optional_occt(monkeypatch)
    model = _native_tspline_model()
    source_patch = model.patches[0]
    assert isinstance(source_patch, TSplineSurfacePatch)
    expected_points, expected_weights, expected_u_knots, expected_v_knots = (
        _tspline_exact_bezier_data(source_patch)
    )
    expected_point_array = np.asarray(
        [[[float(value) for value in point] for point in row] for row in expected_points]
    )
    expected_weight_array = np.asarray(
        [[float(value) for value in row] for row in expected_weights]
    )
    parameters = jnp.asarray(((0.2, 0.3), (0.5, 0.5), (0.8, 0.7)))
    expected_values = np.asarray(source_patch.evaluate(parameters))

    def assert_exact_restoration(
        restored: BRepModel,
        report: AdapterReport,
        physical_tags: tuple[str, ...],
    ) -> None:
        _assert_geometry_topology_equal(model, restored)
        assert model.physical_tags == ("plane",)
        assert restored.physical_tags == physical_tags
        restored_patch = restored.patches[0]
        assert isinstance(restored_patch, BSplineSurfacePatch)
        assert np.array_equal(
            np.asarray(restored_patch.control_points), expected_point_array
        )
        assert np.array_equal(np.asarray(restored_patch.weights), expected_weight_array)
        assert np.array_equal(
            np.asarray(restored_patch.u_knots),
            np.asarray([float(value) for value in expected_u_knots]),
        )
        assert np.array_equal(
            np.asarray(restored_patch.v_knots),
            np.asarray([float(value) for value in expected_v_knots]),
        )
        values = np.asarray(restored_patch.evaluate(parameters))
        assert np.max(np.abs(values - expected_values)) <= (
            8.0 * np.finfo(np.float64).eps
        )
        assert any(
            loss.path == "face:0.tspline-refinement"
            and "T-mesh refinement" in loss.rationale
            for loss in report.losses
        )
        if physical_tags != model.physical_tags:
            assert any(loss.path == "face:0.physical-tag" for loss in report.losses)

    step_path = tmp_path / "tspline-exact.step"
    step_export = write_step(model, step_path)
    step_restored = read_step(step_path, _POLICY, trusted_root=tmp_path)
    assert_exact_restoration(step_restored.model, step_export.report, model.physical_tags)

    iges_path = tmp_path / "tspline-exact.iges"
    iges_export = write_iges(model, iges_path)
    iges_restored = read_iges(iges_path, _POLICY, trusted_root=tmp_path)
    assert_exact_restoration(iges_restored.model, iges_export.report, model.physical_tags)

    brep_path = tmp_path / "tspline-exact.brep"
    brep_export = write_brep_text(model, brep_path)
    brep_restored = read_brep_text(
        brep_path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert_exact_restoration(brep_restored.model, brep_export.report, ("bspline",))

    unsupported = _native_tspline_model(refined=True)
    fit_policy = CadExportPolicy(intersection_approximation=CadCurveFitPolicy(1.0e-8))
    for format_name, extension in (
        ("step", ".step"),
        ("iges", ".iges"),
        ("brep", ".brep"),
    ):
        destination = tmp_path / f"tspline-refined{extension}"
        with pytest.raises(CadInterchangeError) as error:
            if format_name == "step":
                write_step(unsupported, destination, policy=fit_policy)
            elif format_name == "iges":
                write_iges(unsupported, destination, policy=fit_policy)
            else:
                write_brep_text(unsupported, destination, policy=fit_policy)
        assert error.value.refusal.reason == "unsupported-entity"
        assert "multi-piece T-spline" in error.value.refusal.message
        assert "no approximation is performed" in error.value.refusal.message
        assert not destination.exists()


def test_brep_text_fixture_checks_shape_cycles_and_payload_extent() -> None:
    fixture = _brep_text_empty_compound()
    imported = decode_brep_text_bytes(fixture, _POLICY, source_length_unit=METER)
    assert imported.report.valid
    assert _shape_counts(imported.model) == (0, 0, 0, 0)

    cyclic = fixture.replace(
        b"TShapes 1\nCo\n0000000\n*\n+1 0",
        b"TShapes 1\nCo\n0000000\n+1 0\n*\n+1 0",
    )
    with pytest.raises(CadInterchangeError) as error:
        decode_brep_text_bytes(cyclic, _POLICY, source_length_unit=METER)
    assert error.value.refusal.reason == "cyclic-reference"

    oversized = fixture.replace(b"Curve2ds 0", b"Curve2ds 1000000000")
    with pytest.raises(CadInterchangeError) as error:
        decode_brep_text_bytes(oversized, _POLICY, source_length_unit=METER)
    assert error.value.refusal.reason == "malformed"

    oversized_shapes = fixture.replace(b"TShapes 1", b"TShapes 2")
    with pytest.raises(CadInterchangeError) as error:
        decode_brep_text_bytes(oversized_shapes, _POLICY, source_length_unit=METER)
    assert error.value.refusal.reason == "malformed"


@pytest.mark.parametrize("kind", ("curve", "surface"))
def test_step_nested_offset_carriers_decode_iteratively(
    kind: str, tmp_path: Path
) -> None:
    result = decode_step_bytes(_nested_step_carrier_fixture(kind, 96), _POLICY)
    assert result.report.valid
    source_geometry = result.model.geometry
    assert source_geometry is not None
    if kind == "curve":
        edge_curve = next(index for index in source_geometry.edge_curves if index >= 0)
        source_carrier = source_geometry.curves[edge_curve]
        carrier = source_carrier
        depth = 0
        while isinstance(carrier, OffsetCurve):
            depth += 1
            carrier = carrier.base
    else:
        source_carrier = result.model.patches[0]
        carrier = source_carrier
        depth = 0
        while isinstance(carrier, OffsetSurface):
            depth += 1
            carrier = carrier.base
    assert depth == 97

    path = tmp_path / f"nested-{kind}.brep"
    exported = write_brep_text(result.model, path)
    restored = read_brep_text(
        path, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    assert exported.report.valid and restored.report.valid
    assert any(
        loss.changes_interpretation and "source identifiers" in loss.rationale
        for loss in exported.report.losses
    )
    if kind == "curve":
        assert any(
            loss.path == "geometry.exact-scalar-representation"
            for loss in exported.report.losses
        )

    restored_geometry = restored.model.geometry
    assert restored_geometry is not None
    if kind == "curve":
        edge_curve = next(index for index in restored_geometry.edge_curves if index >= 0)
        restored_carrier = restored_geometry.curves[edge_curve]
        carrier = restored_carrier
        restored_depth = 0
        while isinstance(carrier, OffsetCurve):
            restored_depth += 1
            carrier = carrier.base
    else:
        restored_carrier = restored.model.patches[0]
        carrier = restored_carrier
        restored_depth = 0
        while isinstance(carrier, OffsetSurface):
            restored_depth += 1
            carrier = carrier.base
    assert restored_depth == 97

    def carrier_digest(value: StrictModule) -> str:
        return canonical_fingerprint(_carrier_payload(value))

    assert carrier_digest(source_carrier) == carrier_digest(restored_carrier)
    assert source_geometry.edge_curves == restored_geometry.edge_curves
    assert source_geometry.edge_vertices == restored_geometry.edge_vertices
    assert source_geometry.coedge_edges == restored_geometry.coedge_edges
    assert source_geometry.coedge_senses == restored_geometry.coedge_senses
    assert source_geometry.face_loops == restored_geometry.face_loops
    assert source_geometry.shell_faces == restored_geometry.shell_faces
    assert source_geometry.shell_orientations == restored_geometry.shell_orientations
    assert source_geometry.solid_shells == restored_geometry.solid_shells
    assert np.array_equal(
        np.asarray(source_geometry.vertex_points),
        np.asarray(restored_geometry.vertex_points),
    )
    assert np.array_equal(
        np.asarray(source_geometry.edge_ranges),
        np.asarray(restored_geometry.edge_ranges),
    )
    assert tuple(carrier_digest(value) for value in source_geometry.curves) == tuple(
        carrier_digest(value) for value in restored_geometry.curves
    )
    assert tuple(
        array_tree_fingerprint(value) for value in source_geometry.curves
    ) == tuple(array_tree_fingerprint(value) for value in restored_geometry.curves)
    assert tuple(
        array_tree_fingerprint(value) for value in source_geometry.pcurves
    ) == tuple(array_tree_fingerprint(value) for value in restored_geometry.pcurves)
    assert tuple(
        _carrier_payload(value)["definition"] for value in source_geometry.pcurves
    ) == tuple(
        _carrier_payload(value)["definition"] for value in restored_geometry.pcurves
    )
    assert tuple(carrier_digest(value) for value in result.model.patches) == tuple(
        carrier_digest(value) for value in restored.model.patches
    )
    assert np.array_equal(
        np.asarray(result.model.parameter_bounds),
        np.asarray(restored.model.parameter_bounds),
    )
    assert np.array_equal(
        np.asarray(result.model.orientation), np.asarray(restored.model.orientation)
    )
    assert result.model.physical_tags == restored.model.physical_tags
    assert tuple(
        None if domain is None else canonical_fingerprint(domain.identity_payload())
        for domain in result.model.trim_domains
    ) == tuple(
        None if domain is None else canonical_fingerprint(domain.identity_payload())
        for domain in restored.model.trim_domains
    )
    if kind == "surface":
        step_path = tmp_path / "nested-surface.step"
        step_export = write_step(result.model, step_path)
        step_restored = read_step(step_path, _POLICY, trusted_root=tmp_path)
        assert step_export.report.valid and step_restored.report.valid
        _assert_geometry_preserved(result.model, step_restored.model)


def test_brep_text_allocation_failure_propagates_before_result_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = decode_step_bytes(_STEP_SQUARE, _POLICY).model
    path = tmp_path / "allocation-source.brep"
    write_brep_text(model, path)
    payload = path.read_bytes()

    def fail_allocation(*args: object, **kwargs: object) -> object:
        raise MemoryError("injected CAD carrier allocation failure")

    monkeypatch.setattr(_brep_text_module.np, "empty", fail_allocation)
    with pytest.raises(MemoryError, match="injected CAD carrier allocation failure"):
        decode_brep_text_bytes(payload, _POLICY, source_length_unit=METER)
    assert path.read_bytes() == payload


def test_step_iges_reference_cycles_and_declared_counts_refuse() -> None:
    cyclic_step = _STEP_SQUARE.replace(
        b"ENDSEC;\nEND-ISO-10303-21;",
        b"#999=REPRESENTATION_MAP(#999,#999);\nENDSEC;\nEND-ISO-10303-21;",
    )
    with pytest.raises(CadInterchangeError) as error:
        decode_step_bytes(cyclic_step, _POLICY)
    assert error.value.refusal.reason == "cyclic-reference"

    cyclic_iges = _iges_bytes(((308, ("0", _hollerith("cycle"), "1", "1"), 0),))
    with pytest.raises(CadInterchangeError) as error:
        decode_iges_bytes(cyclic_iges, _POLICY)
    assert error.value.refusal.reason == "cyclic-reference"

    huge_vertex_count = _iges_bytes(((502, ("1000000000",), 0),))
    with pytest.raises(CadInterchangeError) as error:
        decode_iges_bytes(huge_vertex_count, _POLICY)
    assert error.value.refusal.reason == "malformed"


def test_iges_deep_composite_curve_uses_an_explicit_stack() -> None:
    imported = decode_iges_bytes(_iges_square_with_nested_composite(1100), _POLICY)
    assert imported.report.valid
    assert _shape_counts(imported.model) == (4, 4, 1, 1)


def test_iges_nested_definition_dag_is_retained_without_expansion(
    tmp_path: Path,
) -> None:
    imported = decode_iges_bytes(_iges_nested_empty(root_uses=2), _POLICY)
    geometry = imported.model.geometry
    assert geometry is not None
    graph = geometry.assembly_graph
    assert graph is not None
    _assert_shared_empty_definitions(graph)
    renamed = decode_iges_bytes(
        _iges_nested_empty("display-a", "display-b", root_uses=2), _POLICY
    )
    renamed_geometry = renamed.model.geometry
    assert renamed_geometry is not None
    renamed_graph = renamed_geometry.assembly_graph
    assert renamed_graph is not None
    assert renamed_graph.identity_payload() == graph.identity_payload()
    _assert_shared_empty_definitions(renamed_graph)

    output = tmp_path / "nested.iges"
    exported = write_iges(imported.model, output)
    payload = output.read_bytes()
    assert b"PhydraxDefinition" not in payload
    assert b"PhydraxUse" not in payload
    assert b"PhydraxPath" not in payload
    restored = read_iges(output, _POLICY, trusted_root=tmp_path)
    restored_geometry = restored.model.geometry
    assert exported.report.valid and restored.report.valid
    _assert_source_identifier_loss(exported.report)
    assert restored_geometry is not None
    restored_graph = restored_geometry.assembly_graph
    assert restored_graph is not None
    _assert_shared_empty_definitions(restored_graph)

    step_path = tmp_path / "nested.step"
    step_export = write_step(imported.model, step_path)
    step_restored = read_step(step_path, _POLICY, trusted_root=tmp_path)
    assert step_export.report.valid and step_restored.report.valid
    _assert_source_identifier_loss(step_export.report)
    step_geometry = step_restored.model.geometry
    assert step_geometry is not None
    assert step_geometry.assembly_graph is not None
    _assert_shared_empty_definition_dag(step_geometry.assembly_graph)
    assert any(
        loss.changes_interpretation and "wrapper nodes" in loss.rationale
        for loss in step_export.report.losses
    )


def test_brep_text_retains_nested_empty_assembly_definition_dag(
    tmp_path: Path,
) -> None:
    imported = decode_iges_bytes(_iges_nested_empty(root_uses=2), _POLICY)
    geometry = imported.model.geometry
    assert geometry is not None
    graph = geometry.assembly_graph
    assert graph is not None
    _assert_shared_empty_definitions(graph)

    output = tmp_path / "nested-empty.brep"
    exported = write_brep_text(imported.model, output)
    restored = read_brep_text(
        output, _POLICY, trusted_root=tmp_path, source_length_unit=METER
    )
    restored_geometry = restored.model.geometry
    assert exported.report.valid and restored.report.valid
    _assert_source_identifier_loss(exported.report)
    assert restored_geometry is not None
    restored_graph = restored_geometry.assembly_graph
    assert restored_graph is not None
    _assert_shared_empty_definitions(restored_graph)

    def nested_definition_signature(graph: BRepAssemblyGraph) -> tuple[object, ...]:
        definitions = {value.definition_id: value for value in graph.definitions}
        root = graph.roots[0]
        assert root.definition_id is not None
        root_definition = definitions[root.definition_id]
        assert len(root_definition.uses) == 2
        child_definition_id = root_definition.uses[0].definition_id
        assert child_definition_id is not None
        leaf_definition = definitions[child_definition_id]
        return (
            root.solid_index,
            root.placement.identity_payload(),
            tuple(
                (use.solid_index, use.placement.identity_payload())
                for use in root_definition.uses
            ),
            len(leaf_definition.uses),
        )

    assert nested_definition_signature(restored_graph) == nested_definition_signature(
        graph
    )
