import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep._assembly import (
    BRepAffinePlacement,
    BRepAssemblyDefinition,
    BRepAssemblyGraph,
    BRepAssemblyUse,
)
from phydrax.geometry.brep._constructors import brep_box
from phydrax.geometry.brep._model import BRepGeometry, BRepModel
from phydrax.interchange._cad import assembly_use_label, CadImportPolicy
from phydrax.interchange._cad_brep_text import read_brep_text, write_brep_text
from phydrax.interchange._cad_carriers import RigidPlacement
from phydrax.interchange._cad_placement_evidence import (
    certify_placed_model,
    source_placement_evidence,
)
from phydrax.interchange._iges import read_iges, write_iges
from phydrax.interchange._step import read_step, write_step
from phydrax.units import METER


def _model_with_graph(model: BRepModel, graph: BRepAssemblyGraph, /) -> BRepModel:
    source = model.geometry
    geometry = BRepGeometry(
        vertex_points=source.vertex_points,
        vertex_exact_points=source.vertex_exact_points,
        vertex_roots=source.vertex_roots,
        edge_endpoint_roots=source.edge_endpoint_roots,
        coedge_endpoint_roots=source.coedge_endpoint_roots,
        edge_vertex_gap_bounds=source.edge_vertex_gap_bounds,
        edge_face_gap_bounds=source.edge_face_gap_bounds,
        curves=source.curves,
        edge_curves=source.edge_curves,
        edge_ranges=source.edge_ranges,
        edge_vertices=source.edge_vertices,
        pcurves=source.pcurves,
        coedge_edges=source.coedge_edges,
        coedge_senses=source.coedge_senses,
        face_loops=source.face_loops,
        shell_faces=source.shell_faces,
        shell_orientations=source.shell_orientations,
        solid_shells=source.solid_shells,
        assembly_graph=graph,
    )
    return BRepModel(
        geometry=geometry,
        patches=model.patches,
        parameter_bounds=model.parameter_bounds,
        orientation=model.orientation,
        physical_tags=model.physical_tags,
        coordinate_contract=model.coordinate_contract,
        report=model.report,
        trim_domains=model.trim_domains,
    )


def _independent_pose_error(
    points: np.ndarray,
    first_axis: tuple[float, float, float],
    rotation: np.ndarray,
    actual_translation: np.ndarray,
    translation: tuple[float, float, float],
    /,
) -> float:
    with localcontext() as context:
        context.prec = 80
        x, y = (Decimal.from_float(value) for value in first_axis[:2])
        length = (x * x + y * y).sqrt()
        expected_rotation = (
            (x / length, -y / length, Decimal(0)),
            (y / length, x / length, Decimal(0)),
            (Decimal(0), Decimal(0), Decimal(1)),
        )
        actual_rotation = tuple(
            tuple(Decimal.from_float(float(value)) for value in row) for row in rotation
        )
        offset = tuple(
            Decimal.from_float(float(actual)) - Decimal.from_float(expected)
            for actual, expected in zip(actual_translation, translation, strict=True)
        )
        maximum = Decimal(0)
        for point in points:
            local = tuple(Decimal.from_float(float(value)) for value in point)
            displacement = tuple(
                sum(
                    (
                        (actual_rotation[row][axis] - expected_rotation[row][axis])
                        * local[axis]
                        for axis in range(3)
                    ),
                    offset[row],
                )
                for row in range(3)
            )
            distance = sum((value * value for value in displacement), Decimal(0)).sqrt()
            maximum = max(maximum, distance)
        return float(maximum)


def test_placement_evidence_is_bound_to_translated_rotated_occurrence() -> None:
    contract = SpatialCoordinateContract.si()
    definition = brep_box((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), coordinate_contract=contract)
    first_axis = (0.8, 0.6, 0.0)
    translation = (4.0, -3.0, 2.0)
    placement = RigidPlacement.from_source_directions(
        np.asarray(first_axis),
        np.asarray((0.0, 0.0, 1.0)),
        np.asarray(translation),
        "translated-rotated-occurrence",
    )
    evidence = source_placement_evidence(placement)
    assert evidence is not None
    affine = BRepAffinePlacement.from_matrix(
        (
            (*placement.rotation[0], placement.translation[0]),
            (*placement.rotation[1], placement.translation[1]),
            (*placement.rotation[2], placement.translation[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    graph = BRepAssemblyGraph(
        roots=(
            BRepAssemblyUse(
                "placed-box",
                affine,
                solid_index=0,
                placement_evidence=evidence,
            ),
        ),
        definitions=(),
        solid_count=1,
    )
    model = _model_with_graph(definition, graph)
    imported = certify_placed_model(
        model,
        CadImportPolicy(contract, maximum_placement_deviation=1.0e-12),
    )
    occurrence = next(imported.geometry.iter_occurrences())
    certified = occurrence.placement_evidence
    assert certified is not None
    oracle = _independent_pose_error(
        np.asarray(definition.geometry.vertex_points),
        first_axis,
        placement.rotation,
        placement.translation,
        translation,
    )

    assert occurrence.path == ("placed-box",)
    assert not np.allclose(placement.rotation, np.eye(3))
    np.testing.assert_allclose(placement.translation, translation, atol=0.0)
    assert certified.model_extent > 0.0
    assert certified.point_deviation_bound >= oracle
    assert certified.point_deviation_bound <= 1.0e-12


def test_exchange_qualifies_reused_use_ids_and_preserves_full_paths(
    tmp_path: Path,
) -> None:
    contract = SpatialCoordinateContract.si()
    identity = BRepAffinePlacement.identity()
    graph = BRepAssemblyGraph(
        roots=(
            BRepAssemblyUse("left", identity, definition_id="root"),
            BRepAssemblyUse("right", identity, definition_id="a"),
        ),
        definitions=(
            BRepAssemblyDefinition(
                "root", (BRepAssemblyUse("x/y", identity, solid_index=0),)
            ),
            BRepAssemblyDefinition(
                "a", (BRepAssemblyUse("x/y", identity, solid_index=0),)
            ),
        ),
        solid_count=1,
    )
    model = _model_with_graph(
        brep_box((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), coordinate_contract=contract),
        graph,
    )
    expected = {
        assembly_use_label(None, "left"),
        assembly_use_label(None, "right"),
        assembly_use_label("root", "x/y"),
        assembly_use_label("a", "x/y"),
    }
    assert len(expected) == 4
    assert assembly_use_label(None, "x") != assembly_use_label("root", "x")
    assert assembly_use_label("a:b", "c") != assembly_use_label("a", "b:c")

    for extension, writer, reader in (
        ("step", write_step, read_step),
        ("iges", write_iges, read_iges),
        ("brep", write_brep_text, read_brep_text),
    ):
        path = tmp_path / f"qualified.{extension}"
        exported = writer(model, path)
        labels = [
            key for key, _ in exported.provenance if key.startswith("assembly-use:")
        ]
        assert len(labels) == len(set(labels))
        assert expected <= set(labels)
        kwargs = {"source_length_unit": METER} if extension == "brep" else {}
        restored = reader(
            path, CadImportPolicy(contract), trusted_root=tmp_path, **kwargs
        )
        imported_labels = [
            key for key, _ in restored.provenance if key.startswith("assembly-use:")
        ]
        assert len(imported_labels) == len(set(imported_labels))
        assert all(
            len(json.loads(key.removeprefix("assembly-use:"))) == 2
            for key in imported_labels
        )
        if extension == "brep":
            fields = {loss.path for loss in exported.report.losses}
            for occurrence_path in (("left", "x/y"), ("right", "x/y")):
                assert (
                    "occurrence-path:"
                    + json.dumps(
                        occurrence_path, ensure_ascii=True, separators=(",", ":")
                    )
                    in fields
                )
