from decimal import Decimal, localcontext

import numpy as np

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep._assembly import (
    BRepAffinePlacement,
    BRepAssemblyGraph,
    BRepAssemblyUse,
)
from phydrax.geometry.brep._constructors import brep_box
from phydrax.geometry.brep._model import BRepGeometry, BRepModel
from phydrax.interchange._cad import CadImportPolicy
from phydrax.interchange._cad_carriers import RigidPlacement
from phydrax.interchange._cad_placement_evidence import (
    certify_placed_model,
    source_placement_evidence,
)


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
            tuple(Decimal.from_float(float(value)) for value in row)
            for row in rotation
        )
        offset = tuple(Decimal.from_float(value) for value in translation)
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
                    Decimal(0),
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
        translation,
    )

    assert occurrence.path == ("placed-box",)
    assert not np.allclose(placement.rotation, np.eye(3))
    np.testing.assert_allclose(placement.translation, translation, atol=0.0)
    assert certified.model_extent > 0.0
    assert certified.point_deviation_bound >= oracle
    assert certified.point_deviation_bound <= 1.0e-12
