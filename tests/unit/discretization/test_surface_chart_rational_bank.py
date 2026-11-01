from fractions import Fraction

import numpy as np
import pytest

from phydrax.discretization._surface_chart_records import (
    rational_chart_tokens,
    SurfaceChartWitness,
    SurfaceChartWitnessArchiveRecipe,
    SurfaceRootMeshCellBinding,
)


def _chart_witness(
    *, exact: bool, root_binding: SurfaceRootMeshCellBinding | None = None
) -> SurfaceChartWitness:
    triangle = (
        (Fraction(0), Fraction(0)),
        (Fraction(1, 3), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    chart = np.asarray(
        [[[0.0, 0.0], [float(Fraction(1, 3)), 0.0], [0.0, 1.0]]],
        dtype=np.float64,
    )
    return SurfaceChartWitness(
        geometry_id="geometry-r1",
        topology_id="topology-r1",
        domain_id="domain-r1",
        cell_global_ids=np.asarray((71,), dtype=np.int64),
        patches=np.asarray((2,), dtype=np.int32),
        charts=chart,
        source_root_cell_ids=np.asarray((901,), dtype=np.int64),
        source_piece_ids=((2, 3, 4),),
        geometry_entity_ids=("source-face-2",),
        occurrence_paths=((),),
        chart_rational_tokens=rational_chart_tokens((triangle,)) if exact else None,
        root_mesh_cell_bindings=(root_binding,),
    )


def test_surface_chart_witness_retains_exact_nonbinary_uv() -> None:
    exact = _chart_witness(exact=True)
    represented_only = _chart_witness(exact=False)

    assert exact.exact_chart(0)[1][0] == Fraction(1, 3)
    assert represented_only.exact_chart(0)[1][0] == Fraction.from_float(
        float(Fraction(1, 3))
    )
    assert exact.exact_chart(0) != represented_only.exact_chart(0)
    assert exact.witness_id != represented_only.witness_id


def test_surface_chart_witness_binds_actual_root_mesh_cell_identity() -> None:
    binding = SurfaceRootMeshCellBinding("root-geometry-r1", "root-topology-r1", 77)
    bound = _chart_witness(exact=True, root_binding=binding)
    atlas_only = _chart_witness(exact=True)

    assert int(np.asarray(bound.source_root_cell_ids)[0]) == 901
    assert bound.root_mesh_cell_bindings == (binding,)
    assert bound.witness_id != atlas_only.witness_id


def test_quadrilateral_chart_preserves_kind_exact_coordinates_and_archive() -> None:
    chart = (
        (Fraction(0), Fraction(0)),
        (Fraction(1, 3), Fraction(0)),
        (Fraction(1, 3), Fraction(1)),
        (Fraction(0), Fraction(1)),
    )
    witness = SurfaceChartWitness(
        geometry_id="geometry-r1",
        topology_id="topology-r1",
        domain_id="domain-r1",
        cell_kind="quadrilateral",
        cell_global_ids=np.asarray((71,), dtype=np.int64),
        patches=np.asarray((2,), dtype=np.int32),
        charts=np.asarray([[tuple(map(float, point)) for point in chart]]),
        source_root_cell_ids=np.asarray((901,), dtype=np.int64),
        source_piece_ids=((2, 3, 4),),
        geometry_entity_ids=("source-face-2",),
        occurrence_paths=((),),
        chart_rational_tokens=rational_chart_tokens((chart,), cell_kind="quadrilateral"),
        root_mesh_cell_bindings=(None,),
    )
    restored = SurfaceChartWitnessArchiveRecipe.from_witness(witness).restore()
    assert restored.cell_kind == "quadrilateral"
    assert restored.exact_chart(0) == chart
    assert restored.witness_id == witness.witness_id
    with pytest.raises(ValueError, match="corner count"):
        rational_chart_tokens((chart,))
