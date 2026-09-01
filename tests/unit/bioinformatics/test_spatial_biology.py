#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.bioinformatics.foundation import (
    BiospecimenLineage,
    ExchangeabilityPlan,
    ExperimentalUnitPlan,
)
from phydrax.bioinformatics.interchange._ngff import (
    image_pyramid_to_ome_ngff,
    is_ome_ngff_metadata,
    NGFFMetadataError,
    ome_ngff_to_image_pyramid,
)
from phydrax.bioinformatics.spatial._assay import (
    SpatialAssay,
    SpatialAssayData,
    SpatialSampleRecord,
)
from phydrax.bioinformatics.spatial._frame import (
    AffineSpatialTransform,
    MICROMETRE,
    MILLIMETRE,
    PIXEL,
    SpatialCoordinates,
    SpatialFrame,
    unit_conversion_transform,
)
from phydrax.bioinformatics.spatial._imaging import (
    BoundedImagePatchPlan,
    extract_bounded_image_patch,
    ImagePatchStatus,
    pyramid_level_tile_bounds,
)
from phydrax.bioinformatics.spatial._morphology import (
    MorphologyPlan,
    summarize_label_morphology,
)
from phydrax.bioinformatics.spatial._neighbors import (
    assay_neighbor_graph,
    build_spatial_neighbor_graph,
    NeighborGraphStatus,
    SpatialNeighborPlan,
)
from phydrax.bioinformatics.spatial._registration import (
    register_spatial_points,
    SpatialRegistrationPlan,
)
from phydrax.bioinformatics.spatial._statistics import (
    assay_autocorrelation_test,
    spatial_autocorrelation_test,
    SpatialStatisticStatus,
)


def _lineage(
    donors: int = 1,
) -> tuple[BiospecimenLineage, tuple[int, ...], tuple[int, ...]]:
    kinds = []
    parents = []
    children = []
    biological = []
    technical = []
    subjects = []
    observations = []
    for donor in range(donors):
        start = 5 * donor
        subjects.append(start)
        observations.append(start + 4)
        kinds.extend(
            (
                BiospecimenLineage.SUBJECT,
                BiospecimenLineage.SPECIMEN,
                BiospecimenLineage.ALIQUOT,
                BiospecimenLineage.LIBRARY,
                BiospecimenLineage.OBSERVATION,
            )
        )
        parents.extend((start, start + 1, start + 2, start + 3))
        children.extend((start + 1, start + 2, start + 3, start + 4))
        biological.extend((-1, -1, -1, -1, donor))
        technical.extend((-1, -1, -1, -1, donor))
    count = 5 * donors
    return (
        BiospecimenLineage(
            np.arange(count),
            kinds,
            parents,
            children,
            biological,
            technical,
            study_id="spatial-study",
        ),
        tuple(subjects),
        tuple(observations),
    )


def _assay() -> SpatialAssay:
    lineage, _, _ = _lineage(2)
    frame = SpatialFrame("slide", ("y", "x"), MICROMETRE)
    records = (
        SpatialSampleRecord("s0", "b0", "d0", "section-a", frame, lineage),
        SpatialSampleRecord("s1", "b1", "d0", "section-b", frame, lineage),
        SpatialSampleRecord("s2", "b2", "d1", "section-c", frame, lineage),
    )
    coordinates = SpatialCoordinates(
        jnp.asarray(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [10.0, 0.0],
                [10.0, 1.0],
                [20.0, 0.0],
                [20.0, 1.0],
                [20.0, 2.0],
                [20.0, 3.0],
            ]
        ),
        frame,
    )
    data = SpatialAssayData(
        coordinates,
        jnp.arange(8.0)[:, None],
        jnp.asarray([0, 0, 1, 1, 2, 2, 2, 2]),
        spot_weights=jnp.asarray([1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5]),
    )
    return SpatialAssay(records, data)


def test_frame_units_mismatch_and_transform_composition():
    micron = SpatialFrame("tissue", ("y", "x"), MICROMETRE)
    millimetre = SpatialFrame("tissue", ("y", "x"), MILLIMETRE)
    registered = SpatialFrame("registered", ("y", "x"), MILLIMETRE)
    to_mm = unit_conversion_transform(micron, millimetre)
    translate = AffineSpatialTransform(
        jnp.eye(2), jnp.asarray([2.0, -1.0]), millimetre, registered
    )
    composed = translate.compose(to_mm)
    points = SpatialCoordinates(jnp.asarray([[1000.0, 2000.0]]), micron)
    np.testing.assert_allclose(composed.apply(points).values, [[3.0, 1.0]])

    wrong_unit = SpatialCoordinates(jnp.asarray([[1.0, 2.0]]), millimetre)
    with pytest.raises(ValueError, match="frame/unit"):
        to_mm.apply(wrong_unit)
    with pytest.raises(ValueError, match="uncalibrated"):
        PIXEL.conversion_factor_to(MICROMETRE)


def test_neighbor_capacity_ties_and_section_isolation():
    points = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.1, 0.0]])
    tied = build_spatial_neighbor_graph(
        points,
        SpatialNeighborPlan("knn", capacity=2, k=2),
        section_index=jnp.asarray([0, 0, 0, 1]),
    )
    assert bool(tied.valid)
    np.testing.assert_array_equal(tied.indices[0], [1, 2])
    assert int(tied.evidence.tied_distance_pairs) >= 1
    assert not bool(jnp.any(tied.mask[:3] & (tied.indices[:3] == 3)))

    overflow = build_spatial_neighbor_graph(
        points[:3],
        SpatialNeighborPlan("radius", capacity=1, radius=2.0),
    )
    assert not bool(overflow.valid)
    assert int(overflow.status) == int(NeighborGraphStatus.CAPACITY_EXCEEDED)
    assert int(overflow.evidence.required_capacity) == 2
    assert not bool(jnp.any(overflow.mask))


def test_assay_supports_multiple_sections_donors_and_unequal_density():
    assay = _assay()
    assert assay.sample_count == 3
    assert assay.section_count == 3
    assert assay.donor_count == 2
    np.testing.assert_array_equal(assay.donor_index(), [0, 0, 0, 0, 1, 1, 1, 1])
    graph = assay_neighbor_graph(assay, SpatialNeighborPlan("knn", capacity=1, k=1))
    assert bool(graph.valid)
    assert int(graph.evidence.section_count) == 3

    result = assay_autocorrelation_test(
        assay,
        0,
        graph,
        jax.random.key(3),
        permutations=31,
    )
    assert bool(result.valid)
    assert int(result.evidence.donor_count) == 2
    assert float(result.evidence.effective_spot_count) < 8.0


def test_one_donor_inference_is_observably_invalid():
    points = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    graph = build_spatial_neighbor_graph(
        points,
        SpatialNeighborPlan("knn", capacity=1, k=1),
    )
    result = spatial_autocorrelation_test(
        jnp.asarray([0.0, 1.0, 3.0]),
        graph,
        jax.random.key(0),
        permutations=15,
        donor_index=jnp.zeros((3,), dtype=jnp.int32),
    )
    assert not bool(result.valid)
    assert int(result.status) == int(SpatialStatisticStatus.INSUFFICIENT_DONORS)
    assert bool(jnp.isnan(result.p_value))


def test_foundation_exchangeability_blocks_drive_permutations():
    lineage, subjects, observations = _lineage(2)
    units = ExperimentalUnitPlan(
        lineage,
        jnp.asarray(observations),
        jnp.asarray(subjects),
        jnp.asarray([0, 1]),
        block_group_ids=jnp.asarray([0, 0]),
    )
    exchangeability = ExchangeabilityPlan(
        units,
        jnp.asarray([7, 7]),
    )
    points = jnp.asarray(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0], [11.0, 0.0], [12.0, 0.0]]
    )
    sections = jnp.asarray([0, 0, 0, 1, 1, 1])
    graph = build_spatial_neighbor_graph(
        points,
        SpatialNeighborPlan("knn", capacity=1, k=1),
        section_index=sections,
    )
    result = spatial_autocorrelation_test(
        jnp.asarray([0.0, 1.0, 2.0, 8.0, 9.0, 10.0]),
        graph,
        jax.random.key(2),
        statistic="geary",
        permutations=31,
        section_index=sections,
        exchangeability_plan=exchangeability,
        observation_entity_indices=jnp.asarray(
            [observations[0]] * 3 + [observations[1]] * 3
        ),
    )
    assert bool(result.valid)
    assert int(result.evidence.donor_count) == 2
    assert int(result.evidence.exchangeability_group_count) == 1
    assert result.method_contract.method_name == "design_aware_geary_permutation_test"


def test_registration_reports_convergence_and_uncertainty():
    source_frame = SpatialFrame("moving", ("y", "x"), MICROMETRE)
    target_frame = SpatialFrame("fixed", ("y", "x"), MICROMETRE)
    moving = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 2.0]])
    fixed = moving + jnp.asarray([3.0, -2.0])
    result = register_spatial_points(
        moving,
        fixed,
        source_frame,
        target_frame,
        plan=SpatialRegistrationPlan(
            epsilon=0.02,
            outer_iterations=12,
            sinkhorn_iterations=300,
            transform_tolerance=1.0e-4,
            sinkhorn_tolerance=1.0e-6,
        ),
    )
    assert bool(result.valid)
    np.testing.assert_allclose(result.aligned_coordinates, fixed, atol=2.0e-2)
    assert bool(result.uncertainty.valid)
    assert float(result.uncertainty.translation_standard_error) >= 0.0
    assert float(result.uncertainty.rotation_identifiability) > 0.0
    assert int(result.evidence.iterations) <= 12


def test_bounded_patches_cross_chunks_and_reject_capacity_overflow():
    image = jnp.arange(36).reshape((6, 6))
    patch = extract_bounded_image_patch(
        image,
        jnp.asarray([1, 1]),
        jnp.asarray([4, 4]),
        BoundedImagePatchPlan((4, 4)),
        chunk_shape=(3, 3),
    )
    assert bool(patch.valid)
    assert bool(patch.evidence.crosses_chunk_boundary)
    assert int(patch.evidence.touched_chunk_count) == 4
    np.testing.assert_array_equal(patch.values, np.asarray(image)[1:5, 1:5])

    overflow = extract_bounded_image_patch(
        image,
        jnp.asarray([0, 0]),
        jnp.asarray([5, 4]),
        BoundedImagePatchPlan((4, 4)),
        chunk_shape=(3, 3),
    )
    assert not bool(overflow.valid)
    assert int(overflow.status) == int(ImagePatchStatus.CAPACITY_EXCEEDED)
    assert not bool(jnp.any(overflow.valid_mask))


def test_ngff_axes_multiscales_tiles_and_generic_zarr_distinction():
    attributes = {
        "multiscales": [
            {
                "version": "0.4",
                "name": "tissue",
                "axes": [
                    {"name": "c", "type": "channel"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 0.5, 0.5]}
                        ],
                    },
                    {
                        "path": "1",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0, 1.0]},
                            {"type": "translation", "translation": [0.0, 0.25, 0.25]},
                        ],
                    },
                ],
            }
        ]
    }
    metadata = ome_ngff_to_image_pyramid(
        attributes,
        {"0": (2, 9, 10), "1": (2, 5, 5)},
        {"0": (1, 4, 4), "1": (1, 4, 4)},
    )
    assert [axis.name for axis in metadata.axes] == ["c", "y", "x"]
    assert metadata.levels[0].chunk_grid_shape == (2, 3, 3)
    assert pyramid_level_tile_bounds(metadata.levels[0], (1, 2, 2)) == (
        (1, 8, 8),
        (2, 9, 10),
    )
    assert (
        image_pyramid_to_ome_ngff(metadata)["multiscales"][0]["axes"][1]["unit"]
        == "micrometer"
    )
    assert is_ome_ngff_metadata(attributes)
    assert not is_ome_ngff_metadata({"zarr_format": 2, "shape": [9, 10]})
    with pytest.raises(NGFFMetadataError, match="generic Zarr"):
        ome_ngff_to_image_pyramid(
            {"zarr_format": 2},
            {"0": (9, 10)},
            {"0": (4, 4)},
        )


def test_label_morphology_includes_cubical_topology_and_overflow():
    labels = jnp.asarray(
        [
            [1, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 1, 2],
        ]
    )
    summary = summarize_label_morphology(labels, MorphologyPlan(2))
    assert bool(summary.valid)
    np.testing.assert_array_equal(summary.labels, [1, 2])
    assert int(summary.euler_characteristic[0]) == 0
    assert float(summary.area[0]) == 8.0

    overflow = summarize_label_morphology(labels, MorphologyPlan(1))
    assert not bool(overflow.valid)
    assert int(overflow.evidence.required_objects) == 2
    assert not bool(jnp.any(overflow.object_valid))
