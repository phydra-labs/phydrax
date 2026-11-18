import gc
import weakref
from copy import copy
from fractions import Fraction
from typing import Literal

import numpy as np
import pytest

from phydrax.geometry._embedding_state import MeshCertificateFinding
from phydrax.geometry._source_boundary_records import (
    _SOURCE_BOUNDARY_SAMPLES_OWNERSHIP,
    rational_chart_tokens,
    SourceBoundaryChartCover,
    SourceBoundaryChartWitness,
    SourceBoundaryDistance,
    SourceBoundaryFaceImageWitness,
    SourceBoundaryImagePullback,
    SourceBoundarySamples,
)


def test_source_distance_and_sample_records_keep_semantics_and_coverage() -> None:
    distance = SourceBoundaryDistance((0.0, 1.0), (0.1, 1.1), "certified")
    samples = SourceBoundarySamples(
        ((0.0, 0.0), (1.0, 0.0)),
        (0.1, 0.1),
        (0.0, 0.0),
        "sampled",
        complete=False,
    )

    assert distance.semantics == "certified"
    assert np.all(distance.lower <= distance.upper)
    assert samples.semantics == "sampled"
    assert not samples.complete
    assert not samples.points.flags.writeable


def test_source_distance_and_cover_reject_malformed_bounds() -> None:
    with pytest.raises(ValueError, match="lower <= upper"):
        SourceBoundaryDistance((1.0,), (0.0,), "certified")
    with pytest.raises(ValueError, match="nonnegative"):
        SourceBoundarySamples(((0.0, 0.0),), (-1.0,), (0.0,), "certified", complete=True)


def _sealed_samples() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = (
        np.asarray(((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))),
        np.asarray((0.2, np.inf)),
        np.asarray((0.0, 0.1)),
    )
    for array in arrays:
        array.setflags(write=False)
    return arrays


def test_sample_private_adoption_preserves_identity_and_public_copy() -> None:
    points, radii, errors = _sealed_samples()
    adopted = SourceBoundarySamples(
        points,
        radii,
        errors,
        "certified",
        complete=True,
        _ownership=_SOURCE_BOUNDARY_SAMPLES_OWNERSHIP,
    )
    assert adopted.points is points
    assert adopted.covering_radius is radii
    assert adopted.point_error is errors
    assert not adopted.points.flags.writeable
    mutable = np.asarray(((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)))
    public = SourceBoundarySamples(mutable, radii, errors, "sampled", complete=False)
    mutable[0, 0] = 17.0
    assert public.points[0, 0] == 0.0
    assert not np.shares_memory(public.points, mutable)
    assert not np.shares_memory(public.covering_radius, radii)


@pytest.mark.parametrize(
    "fault", ("type", "dtype", "rank", "view", "noncontiguous", "writable")
)
def test_sample_private_adoption_rejects_unowned_or_malformed_buffers(fault: str) -> None:
    points, radii, errors = _sealed_samples()
    if fault == "type":
        points = points.tolist()
    elif fault == "dtype":
        points = points.astype(np.float32)
        points.setflags(write=False)
    elif fault == "rank":
        points = np.asarray((0.0, 1.0))
        points.setflags(write=False)
    elif fault == "view":
        points = points[:]
    elif fault == "noncontiguous":
        points = np.asfortranarray(points)
        points.setflags(write=False)
    else:
        points = points.copy()
    with pytest.raises(ValueError, match="sealed, owning, C-contiguous"):
        SourceBoundarySamples(
            points,
            radii,
            errors,
            "certified",
            complete=True,
            _ownership=_SOURCE_BOUNDARY_SAMPLES_OWNERSHIP,
        )


@pytest.mark.parametrize("fault", ("nonfinite_point", "nan_radius", "negative_error"))
def test_sample_private_adoption_preserves_numerical_validation(fault: str) -> None:
    points, radii, errors = (array.copy() for array in _sealed_samples())
    if fault == "nonfinite_point":
        points[1, 0] = np.inf
    elif fault == "nan_radius":
        radii[1] = np.nan
    else:
        errors[1] = -1.0
    for array in (points, radii, errors):
        array.setflags(write=False)
    with pytest.raises(ValueError, match="finite points and nonnegative"):
        SourceBoundarySamples(
            points,
            radii,
            errors,
            "certified",
            complete=True,
            _ownership=_SOURCE_BOUNDARY_SAMPLES_OWNERSHIP,
        )


def test_chart_cover_binds_source_revision_and_restriction_witnesses() -> None:
    cover = SourceBoundaryChartCover(
        (((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),),
        (0.01,),
        "certified",
        True,
        "authored-surface",
        "revision-7",
        restriction_ids=("patch-0-root-cell-4",),
    )

    assert cover.complete
    assert cover.source_id == "authored-surface"
    assert cover.source_revision == "revision-7"
    assert cover.restriction_ids == ("patch-0-root-cell-4",)
    assert not cover.simplices.flags.writeable
    with pytest.raises(ValueError, match="unique"):
        SourceBoundaryChartCover(
            (((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),),
            (0.01,),
            "certified",
            True,
            "authored-surface",
            "revision-7",
            restriction_ids=("same", "same"),
        )


def test_chart_cover_identity_binds_exact_content_not_float_evidence() -> None:
    triangle = (((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),)

    def cover(
        source: str = "source",
        revision: str = "revision",
        semantics: Literal["certified", "sampled"] = "certified",
        complete: bool = True,
        restrictions: tuple[str, ...] = ("restriction",),
        findings: tuple[MeshCertificateFinding, ...] = (),
        simplices: tuple[tuple[tuple[float, float], ...], ...] = triangle,
        bounds: tuple[float, ...] = (0.1,),
    ) -> SourceBoundaryChartCover:
        return SourceBoundaryChartCover(
            simplices,
            bounds,
            semantics,
            complete,
            source,
            revision,
            findings=findings,
            restriction_ids=restrictions,
        )

    identity = cover().cover_id
    assert cover().cover_id == identity
    assert cover(source="other").cover_id != identity
    assert cover(revision="other").cover_id != identity
    assert cover(semantics="sampled").cover_id != identity
    assert cover(complete=False).cover_id != identity
    assert (
        cover(findings=(MeshCertificateFinding("check", "unresolved", "mesh"),)).cover_id
        != identity
    )
    assert cover(restrictions=("other",)).cover_id != identity
    assert cover(simplices=triangle * 2, bounds=(0.1, 0.1)).cover_id != identity
    assert (
        cover(
            simplices=(((0.25, 0.0), (1.0, 0.0), (0.0, 1.0)),),
            bounds=(0.2,),
        ).cover_id
        == identity
    )


@pytest.mark.parametrize(
    "field",
    (
        "mesh",
        "geometry",
        "topology",
        "source",
        "revision",
        "query",
        "cell",
        "entity",
        "chart",
        "source_entity",
        "occurrence",
        "target",
        "parameter",
        "orientation",
    ),
)
def test_source_chart_witness_identity_binds_content(field: str) -> None:
    def make(changed: str) -> SourceBoundaryChartWitness:
        target = np.asarray([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=np.float64)
        parameter = target.copy()
        if changed == "target":
            target[0, 1, 0] = 0.5
        if changed == "parameter":
            parameter[0, 1, 0] = 0.5
        return SourceBoundaryChartWitness(
            mesh_id="other" if changed == "mesh" else "mesh",
            geometry_id="other" if changed == "geometry" else "geometry",
            topology_id="other" if changed == "topology" else "topology",
            source_id="other" if changed == "source" else "source",
            source_revision="other" if changed == "revision" else "revision",
            source_query_id="other" if changed == "query" else "query",
            chart_dimension=2,
            reference_dimension=2,
            cell_global_ids=np.asarray([2 if changed == "cell" else 1], dtype=np.int64),
            entity_global_ids=np.asarray(
                [2 if changed == "entity" else 1], dtype=np.int64
            ),
            local_entity_indices=np.asarray([-1], dtype=np.int32),
            target_reference_vertices=target,
            source_chart_ids=("other" if changed == "chart" else "chart",),
            source_entity_ids=("other" if changed == "source_entity" else "face",),
            occurrence_paths=(("use",) if changed == "occurrence" else (),),
            source_parameters=parameter,
            orientations=np.asarray(
                [-1 if changed == "orientation" else 1], dtype=np.int8
            ),
        )

    identity = make("").witness_id
    assert make("").witness_id == identity
    assert make(field).witness_id != identity


@pytest.mark.parametrize(
    ("bank", "message"),
    (
        (
            ((((0, 0), (0, 1)), ((1, 1), (0, 1)), ((0, 1), (1, 1))),),
            "denominators must be positive",
        ),
        (
            ((((0, -1), (0, 1)), ((1, 1), (0, 1)), ((0, 1), (1, 1))),),
            "denominators must be positive",
        ),
        (
            ((((0, 2), (0, 1)), ((1, 1), (0, 1)), ((0, 1), (1, 1))),),
            "exact coordinate differs",
        ),
        (((((0, 1), (0, 1)),),), "wrong corner count"),
        (
            ((((0, 1),), ((1, 1), (0, 1)), ((0, 1), (1, 1))),),
            "wrong coordinate dimension",
        ),
    ),
)
def test_chart_witness_refuses_noncanonical_token_banks(
    bank: tuple[tuple[tuple[tuple[int, int], ...], ...], ...], message: str
) -> None:
    chart = np.asarray([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=np.float64)
    with pytest.raises(ValueError, match=message):
        SourceBoundaryChartWitness(
            mesh_id="mesh",
            geometry_id="geometry",
            topology_id="topology",
            source_id="source",
            source_revision="revision",
            source_query_id="query",
            chart_dimension=2,
            reference_dimension=2,
            cell_global_ids=np.asarray([1], dtype=np.int64),
            entity_global_ids=np.asarray([1], dtype=np.int64),
            local_entity_indices=np.asarray([-1], dtype=np.int32),
            target_reference_vertices=chart,
            target_reference_rational_tokens=bank,
            source_chart_ids=("chart",),
            source_entity_ids=("face",),
            occurrence_paths=((),),
            source_parameters=chart,
            orientations=np.asarray([1], dtype=np.int8),
        )


@pytest.mark.parametrize(
    "field",
    ("local", "vertices", "target_tokens", "source_tokens", "cell_keys", "edge_keys"),
)
def test_source_chart_identity_binds_exact_and_ancestry_rows(field: str) -> None:
    one = Fraction(1)
    half = Fraction(1, 2)
    rounded_half = Fraction((1 << 54) + 1, 1 << 55)
    normal = (((Fraction(0), Fraction(0)), (half, Fraction(0)), (Fraction(0), one)),)
    changed = (
        ((Fraction(0), Fraction(0)), (rounded_half, Fraction(0)), (Fraction(0), one)),
    )
    charts = np.asarray([[[0.0, 0.0], [0.5, 0.0], [0.0, 1.0]]], dtype=np.float64)

    def make(variation: str) -> SourceBoundaryChartWitness:
        cells = np.zeros((1, 4), dtype=np.int64)
        edges = np.zeros((1, 3, 2, 3), dtype=np.int64)
        if variation == "cell_keys":
            cells[0, 0] = 1
        if variation == "edge_keys":
            edges[0, 0, 0, 0] = 1
        return SourceBoundaryChartWitness(
            mesh_id="mesh",
            geometry_id="geometry",
            topology_id="topology",
            source_id="source",
            source_revision="revision",
            source_query_id="query",
            chart_dimension=2,
            reference_dimension=2,
            cell_global_ids=np.asarray([1], dtype=np.int64),
            entity_global_ids=np.asarray([1], dtype=np.int64),
            local_entity_indices=np.asarray(
                [1 if variation == "local" else 0], dtype=np.int32
            ),
            target_reference_vertices=charts,
            target_reference_rational_tokens=rational_chart_tokens(
                changed if variation == "target_tokens" else normal
            ),
            target_face_vertex_global_ids=np.asarray(
                [[1, 2, 4 if variation == "vertices" else 3]], dtype=np.int64
            ),
            source_chart_ids=("chart",),
            source_entity_ids=("face",),
            occurrence_paths=((),),
            source_parameters=charts,
            source_parameter_rational_tokens=rational_chart_tokens(
                changed if variation == "source_tokens" else normal
            ),
            orientations=np.asarray([1], dtype=np.int8),
            implicit_cell_keys=cells,
            implicit_root_edge_keys=edges,
        )

    assert make(field).witness_id != make("").witness_id


def test_source_chart_identity_binds_dimensions() -> None:
    def make(
        chart_dimension: int, reference_dimension: int
    ) -> SourceBoundaryChartWitness:
        target = np.zeros((1, chart_dimension + 1, reference_dimension), dtype=np.float64)
        target[0, 1, 0] = 1
        if chart_dimension == 2:
            target[0, 2, 1] = 1
        return SourceBoundaryChartWitness(
            mesh_id="mesh",
            geometry_id="geometry",
            topology_id="topology",
            source_id="source",
            source_revision="revision",
            source_query_id="query",
            chart_dimension=chart_dimension,
            reference_dimension=reference_dimension,
            cell_global_ids=np.asarray([1], dtype=np.int64),
            entity_global_ids=np.asarray([1], dtype=np.int64),
            local_entity_indices=np.asarray(
                [-1 if chart_dimension == reference_dimension else 0], dtype=np.int32
            ),
            target_reference_vertices=target,
            source_chart_ids=("chart",),
            source_entity_ids=("face",),
            occurrence_paths=((),),
            source_parameters=target[:, :, :chart_dimension],
            orientations=np.asarray([1], dtype=np.int8),
        )

    original = make(2, 2)
    assert original.witness_id != make(1, 2).witness_id
    assert original.witness_id != make(2, 3).witness_id


def test_source_chart_identity_binds_cad_ancestry() -> None:
    from phydrax.meshing._association import (
        GeometryAssociation,
        GeometryAssociationKind,
        GeometrySourceEntityRole,
    )
    from phydrax.meshing._plc_source import PiecewiseLinearComplex
    from phydrax.meshing.providers._native_cad_volume import NativeCadVolumeSource
    from tests.unit.meshing.test_native_cad_volume import _cad_surface

    domain, surface = _cad_surface()
    source = NativeCadVolumeSource(
        domain, surface, maximum_deviation=0.1, boundary="fixed"
    )
    plc = source.plc_source.complex
    other_plc = NativeCadVolumeSource(
        domain, surface, maximum_deviation=0.1, boundary="conforming"
    ).plc_source.complex
    equivalent_plc = NativeCadVolumeSource(
        domain, surface, maximum_deviation=0.1, boundary="fixed"
    ).plc_source.complex
    charts = plc.cad_source_charts
    assert charts is not None
    patch = int(np.asarray(charts.root_patches)[0])
    root = int(np.asarray(charts.cell_global_ids)[0])
    entity = domain.stratum_entity_ids(2)[patch]
    path = domain.source_occurrences[2][patch]
    triangle = np.asarray([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=np.float64)

    def association(parent: str) -> GeometryAssociation:
        return GeometryAssociation(
            GeometryAssociationKind.CURVE,
            source.source_id,
            source.source_revision,
            "target-edges",
            np.asarray([1], dtype=np.int64),
            ("edge",),
            np.asarray([0.0], dtype=np.float64),
            source_dimensions=np.asarray([1], dtype=np.int32),
            source_indices=np.asarray([0], dtype=np.int64),
            source_roles=(GeometrySourceEntityRole.EDGE,),
            source_parameter_intervals=((Fraction(0), Fraction(1)),),
            orientations=np.asarray([1], dtype=np.int8),
            parent_association_id=parent,
        )

    def make(
        use_cad: bool,
        root_id: int,
        curve: GeometryAssociation | None,
        alternate_plc: PiecewiseLinearComplex | None = None,
    ) -> SourceBoundaryChartWitness:
        return SourceBoundaryChartWitness(
            mesh_id="mesh",
            geometry_id="geometry",
            topology_id="topology",
            source_id=source.source_id,
            source_revision=source.source_revision,
            source_query_id="query",
            chart_dimension=2,
            reference_dimension=2,
            cell_global_ids=np.asarray([1], dtype=np.int64),
            entity_global_ids=np.asarray([1], dtype=np.int64),
            local_entity_indices=np.asarray([-1], dtype=np.int32),
            target_reference_vertices=triangle,
            source_chart_ids=(entity,),
            source_entity_ids=(entity,),
            occurrence_paths=(path,),
            source_parameters=triangle,
            orientations=np.asarray([1], dtype=np.int8),
            cad_plc=(alternate_plc or plc) if use_cad else None,
            cad_curve_association=curve,
            cad_root_cell_ids=np.asarray([root_id], dtype=np.int64) if use_cad else None,
        )

    first = make(True, root, association("parent-a"))
    for witness in (first, make(False, root, None)):
        assert witness.source_parameter_charts() == tuple(
            witness.source_parameter_chart(row)
            for row in range(len(witness.source_chart_ids))
        )
        assert witness.target_reference_charts() == tuple(
            witness.target_reference_chart(row)
            for row in range(len(witness.source_chart_ids))
        )
    assert first.witness_id == make(True, root, association("parent-a")).witness_id
    assert (
        first.witness_id
        == make(True, root, association("parent-a"), equivalent_plc).witness_id
    )
    rebuilt_plc = copy(plc)
    assert (
        first.witness_id
        == make(True, root, association("parent-a"), rebuilt_plc).witness_id
    )
    assert (
        first.witness_id
        == make(True, root, association("parent-a"), rebuilt_plc).witness_id
    )
    offsets = plc.polygon_offsets
    offsets.setflags(write=True)
    original_offset = offsets[0]
    try:
        offsets[0] = 1
        with pytest.raises(ValueError, match="polygon offsets"):
            make(True, root, association("parent-a"))
        with pytest.raises(ValueError, match="polygon offsets"):
            PiecewiseLinearComplex.from_stored_fields(plc)
    finally:
        offsets[0] = original_offset
        offsets.setflags(write=False)
    assert first.witness_id == make(True, root, association("parent-a")).witness_id
    assert first.witness_id != make(False, root, None).witness_id
    assert first.witness_id != make(True, -1, association("parent-a")).witness_id
    assert (
        make(True, -1, association("parent-a")).witness_id
        == make(True, -1, association("parent-a")).witness_id
    )
    assert (
        first.witness_id
        != make(True, root, association("parent-a"), other_plc).witness_id
    )
    assert first.witness_id != make(True, root, association("parent-b")).witness_id
    forged = copy(association("parent-a"))
    object.__setattr__(forged, "source_entity_ids", ("other-edge",))
    with pytest.raises(ValueError, match="recomputed identity"):
        make(True, root, forged)

    patch_entities = domain.stratum_entity_ids(2)
    other = next(index for index in range(len(patch_entities)) if index != patch)

    def two_patches(second_chart: str) -> SourceBoundaryChartWitness:
        return SourceBoundaryChartWitness(
            mesh_id="mesh",
            geometry_id="geometry",
            topology_id="topology",
            source_id=source.source_id,
            source_revision=source.source_revision,
            source_query_id="query",
            chart_dimension=2,
            reference_dimension=2,
            cell_global_ids=np.asarray([1, 2], dtype=np.int64),
            entity_global_ids=np.asarray([1, 2], dtype=np.int64),
            local_entity_indices=np.asarray([-1, -1], dtype=np.int32),
            target_reference_vertices=np.repeat(triangle, 2, axis=0),
            source_chart_ids=(entity, second_chart),
            source_entity_ids=(entity, patch_entities[other]),
            occurrence_paths=(path, domain.source_occurrences[2][other]),
            source_parameters=np.repeat(triangle, 2, axis=0),
            orientations=np.asarray([1, 1], dtype=np.int8),
            cad_plc=plc,
            cad_curve_association=association("parent-a"),
            cad_root_cell_ids=np.asarray([-1, -1], dtype=np.int64),
        )

    assert two_patches(patch_entities[other]).source_entity_ids
    with pytest.raises(ValueError, match="exact plane chart"):
        two_patches(entity)


def test_face_image_identity_and_exact_integer_degeneracy() -> None:
    def make(
        triangle: tuple[tuple[Fraction, ...], ...],
    ) -> SourceBoundaryFaceImageWitness:
        return SourceBoundaryFaceImageWitness(
            mesh_id="mesh",
            geometry_id="geometry",
            topology_id="topology",
            source_id="source",
            source_revision="revision",
            source_query_id="query",
            cell_global_ids=np.asarray((1,), dtype=np.int64),
            entity_global_ids=np.asarray((2,), dtype=np.int64),
            local_entity_indices=np.asarray((0,), dtype=np.int64),
            source_chart_ids=("chart",),
            source_entity_ids=("face",),
            occurrence_paths=((),),
            source_region_sides=((None, "region"),),
            orientations=np.asarray((1,), dtype=np.int8),
            image_triangle_rational_tokens=rational_chart_tokens((triangle,)),
        )

    zero = Fraction(0)
    triangle = (
        (zero, zero, zero),
        (Fraction(1, 3), zero, zero),
        (zero, Fraction(1, 2), zero),
    )
    changed = (
        (zero, zero, zero),
        (Fraction(2, 3), zero, zero),
        (zero, Fraction(1, 2), zero),
    )
    first = make(triangle)
    assert first.witness_id == make(triangle).witness_id
    assert first.witness_id != make(changed).witness_id
    assert first.image_triangle(0) == triangle
    with pytest.raises(ValueError, match="degenerate"):
        make(
            (
                (zero, zero, zero),
                (Fraction(1, 3), zero, zero),
                (Fraction(2, 3), zero, zero),
            )
        )
    with pytest.raises(ValueError, match="canonical rational"):
        SourceBoundaryImagePullback(
            1, 0, (((2, 4), (0, 1), (0, 1)), ((0, 1), (0, 1), (0, 1))), "chart"
        )

    rng = np.random.default_rng(293)
    for _ in range(80):
        numerators = rng.integers(-8, 9, size=(3, 3), dtype=np.int64)
        denominators = rng.integers(1, 9, size=(3, 3), dtype=np.int64)
        random_triangle = tuple(
            tuple(
                Fraction(int(numerators[i, j]), int(denominators[i, j])) for j in range(3)
            )
            for i in range(3)
        )
        first_edge = tuple(
            random_triangle[1][i] - random_triangle[0][i] for i in range(3)
        )
        second_edge = tuple(
            random_triangle[2][i] - random_triangle[0][i] for i in range(3)
        )
        nonzero = any(
            first_edge[i] * second_edge[j] - first_edge[j] * second_edge[i]
            for i, j in ((0, 1), (0, 2), (1, 2))
        )
        if nonzero:
            assert make(random_triangle).image_triangle(0) == random_triangle
        else:
            with pytest.raises(ValueError, match="degenerate"):
                make(random_triangle)


def test_live_cad_chart_preparation_does_not_retain_source() -> None:
    from phydrax.meshing._association import (
        GeometryAssociation,
        GeometryAssociationKind,
        GeometrySourceEntityRole,
    )
    from phydrax.meshing.providers._native_cad_volume import NativeCadVolumeSource
    from tests.unit.meshing.test_native_cad_volume import _cad_surface

    def construct() -> tuple[
        weakref.ReferenceType[object], weakref.ReferenceType[object]
    ]:
        domain, surface = _cad_surface()
        source = NativeCadVolumeSource(
            domain, surface, maximum_deviation=0.1, boundary="fixed"
        )
        plc = source.plc_source.complex
        charts = plc.cad_source_charts
        assert charts is not None
        patch = int(np.asarray(charts.root_patches)[0])
        entity = domain.stratum_entity_ids(2)[patch]
        association = GeometryAssociation(
            GeometryAssociationKind.CURVE,
            source.source_id,
            source.source_revision,
            "target-edges",
            np.asarray([1], dtype=np.int64),
            ("edge",),
            np.asarray([0.0], dtype=np.float64),
            source_dimensions=np.asarray([1], dtype=np.int32),
            source_indices=np.asarray([0], dtype=np.int64),
            source_roles=(GeometrySourceEntityRole.EDGE,),
            source_parameter_intervals=((Fraction(0), Fraction(1)),),
            orientations=np.asarray([1], dtype=np.int8),
        )
        triangle = np.asarray([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=np.float64)
        witness = SourceBoundaryChartWitness(
            mesh_id="mesh",
            geometry_id="geometry",
            topology_id="topology",
            source_id=source.source_id,
            source_revision=source.source_revision,
            source_query_id="query",
            chart_dimension=2,
            reference_dimension=2,
            cell_global_ids=np.asarray([1], dtype=np.int64),
            entity_global_ids=np.asarray([1], dtype=np.int64),
            local_entity_indices=np.asarray([-1], dtype=np.int32),
            target_reference_vertices=triangle,
            source_chart_ids=(entity,),
            source_entity_ids=(entity,),
            occurrence_paths=(domain.source_occurrences[2][patch],),
            source_parameters=triangle,
            orientations=np.asarray([1], dtype=np.int8),
            cad_plc=plc,
            cad_curve_association=association,
            cad_root_cell_ids=np.asarray([-1], dtype=np.int64),
        )
        assert witness.source_chart_ids == (entity,)
        return weakref.ref(domain), weakref.ref(plc)

    domain_reference, plc_reference = construct()
    gc.collect()
    assert domain_reference() is None
    assert plc_reference() is None
