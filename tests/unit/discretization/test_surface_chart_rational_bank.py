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


@pytest.mark.parametrize(
    "field",
    (
        "geometry_id",
        "topology_id",
        "domain_id",
        "cell_global_ids",
        "source_root_cell_ids",
        "source_piece_ids",
        "geometry_entity_ids",
        "occurrence_paths",
        "root_mesh_cell_bindings",
    ),
)
def test_surface_chart_identity_binds_fields(field: str) -> None:
    original = _chart_witness(exact=True)
    recipe = SurfaceChartWitnessArchiveRecipe.from_witness(original)
    assert recipe.restore().witness_id == original.witness_id
    changed = SurfaceChartWitness(
        geometry_id="other" if field == "geometry_id" else original.geometry_id,
        topology_id="other" if field == "topology_id" else original.topology_id,
        domain_id="other" if field == "domain_id" else original.domain_id,
        cell_global_ids=(
            np.asarray((72,), dtype=np.int64)
            if field == "cell_global_ids"
            else original.cell_global_ids
        ),
        patches=original.patches,
        charts=original.charts,
        source_root_cell_ids=(
            np.asarray((902,), dtype=np.int64)
            if field == "source_root_cell_ids"
            else original.source_root_cell_ids
        ),
        source_piece_ids=((2, 3, 5),)
        if field == "source_piece_ids"
        else original.source_piece_ids,
        geometry_entity_ids=("other",)
        if field == "geometry_entity_ids"
        else original.geometry_entity_ids,
        occurrence_paths=(("other",),)
        if field == "occurrence_paths"
        else original.occurrence_paths,
        chart_rational_tokens=original.chart_rational_tokens,
        root_mesh_cell_bindings=(
            (SurfaceRootMeshCellBinding("root-geometry", "root-topology", 77),)
            if field == "root_mesh_cell_bindings"
            else original.root_mesh_cell_bindings
        ),
    )
    assert changed.witness_id != original.witness_id


def test_surface_chart_restore_rejects_tampered_identity() -> None:
    recipe = SurfaceChartWitnessArchiveRecipe.from_witness(_chart_witness(exact=True))
    with pytest.raises(ValueError, match="differs from its recomputed identity"):
        recipe._replace(domain_id="other-domain").restore()


@pytest.mark.parametrize("token", ((2, 6), (1, 0), (1, 2), (True, 3)))
def test_surface_chart_witness_rejects_invalid_exact_tokens(
    token: tuple[int, int],
) -> None:
    original = _chart_witness(exact=True)
    tokens = ((((0, 1), (0, 1)), (token, (0, 1)), ((0, 1), (1, 1))),)
    with pytest.raises((TypeError, ValueError)):
        SurfaceChartWitness(
            geometry_id=original.geometry_id,
            topology_id=original.topology_id,
            domain_id=original.domain_id,
            cell_global_ids=original.cell_global_ids,
            patches=original.patches,
            charts=original.charts,
            source_root_cell_ids=original.source_root_cell_ids,
            source_piece_ids=original.source_piece_ids,
            geometry_entity_ids=original.geometry_entity_ids,
            occurrence_paths=original.occurrence_paths,
            chart_rational_tokens=tokens,
            root_mesh_cell_bindings=original.root_mesh_cell_bindings,
        )


def test_surface_chart_identity_binds_kind_patch_and_chart_bits() -> None:
    original = _chart_witness(exact=True)
    tokens = original.chart_rational_tokens
    chart = np.asarray(original.charts).copy()

    def make(
        kind: str,
        patches: np.ndarray,
        pieces: tuple[tuple[int, ...], ...],
        charts: np.ndarray,
        exact_tokens: tuple[tuple[tuple[tuple[int, int], tuple[int, int]], ...], ...],
    ) -> SurfaceChartWitness:
        return SurfaceChartWitness(
            geometry_id=original.geometry_id,
            topology_id=original.topology_id,
            domain_id=original.domain_id,
            cell_kind=kind,
            cell_global_ids=original.cell_global_ids,
            patches=patches,
            charts=charts,
            source_root_cell_ids=original.source_root_cell_ids,
            source_piece_ids=pieces,
            geometry_entity_ids=original.geometry_entity_ids,
            occurrence_paths=original.occurrence_paths,
            chart_rational_tokens=exact_tokens,
            root_mesh_cell_bindings=original.root_mesh_cell_bindings,
        )

    patch = np.asarray((2,), dtype=np.int32)
    assert (
        make("triangle", patch, ((2, 3, 4),), chart, tokens).witness_id
        == original.witness_id
    )
    assert (
        make(
            "triangle", np.asarray((3,), dtype=np.int32), ((3, 3, 4),), chart, tokens
        ).witness_id
        != original.witness_id
    )
    chart[0, 0, 0] = -0.0
    assert (
        make("triangle", patch, ((2, 3, 4),), chart, tokens).witness_id
        != original.witness_id
    )
    quad = np.concatenate((chart, np.asarray([[[0.0, 0.5]]], dtype=np.float64)), axis=1)
    quad_tokens = (tokens[0] + (((0, 1), (1, 2)),),)
    assert (
        make("quadrilateral", patch, ((2, 3, 4),), quad, quad_tokens).witness_id
        != original.witness_id
    )
