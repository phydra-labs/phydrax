from typing import Literal

import numpy as np
import pytest

from phydrax.geometry._embedding_state import MeshCertificateFinding
from phydrax.geometry._source_boundary_records import (
    _SOURCE_BOUNDARY_SAMPLES_OWNERSHIP,
    SourceBoundaryChartCover,
    SourceBoundaryChartWitness,
    SourceBoundaryDistance,
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
    assert cover(
        simplices=(((0.25, 0.0), (1.0, 0.0), (0.0, 1.0)),),
        bounds=(0.2,),
    ).cover_id == identity


@pytest.mark.parametrize(
    "field",
    (
        "mesh", "geometry", "topology", "source", "revision", "query",
        "cell", "entity", "chart", "source_entity", "occurrence",
        "target", "parameter", "orientation",
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
            entity_global_ids=np.asarray([2 if changed == "entity" else 1], dtype=np.int64),
            local_entity_indices=np.asarray([-1], dtype=np.int32),
            target_reference_vertices=target,
            source_chart_ids=("other" if changed == "chart" else "chart",),
            source_entity_ids=("other" if changed == "source_entity" else "face",),
            occurrence_paths=(("use",) if changed == "occurrence" else (),),
            source_parameters=parameter,
            orientations=np.asarray([-1 if changed == "orientation" else 1], dtype=np.int8),
        )

    identity = make("").witness_id
    assert make("").witness_id == identity
    assert make(field).witness_id != identity
