from fractions import Fraction

import numpy as np
import pytest

from phydrax.meshing._polyhedral_generation import (
    _spanning_guard_sites,
    _spans_three_space,
    same_region_component_ids,
)


def test_power_piece_join_is_limited_to_same_site_region_and_unconstrained_faces() -> (
    None
):
    cell_sites = np.asarray([0, 0, 0, 1, 0], dtype=np.int32)
    cell_regions = np.asarray([2, 2, 7, 2, 2], dtype=np.int32)
    face_cells = np.asarray([[0, 1], [1, 2], [0, 3], [3, -1], [4, -1]], dtype=np.int32)
    face_facets = np.asarray([-1, -1, -1, -1, -1], dtype=np.int32)

    components = same_region_component_ids(
        cell_sites,
        cell_regions,
        face_cells,
        face_facets,
        np.asarray([0, 0], dtype=np.int8),
    )

    np.testing.assert_array_equal(components, [0, 0, 1, 0, 2])


def test_power_piece_join_preserves_authored_facets_and_site_splits() -> None:
    cell_sites = np.asarray([0, 0, 0], dtype=np.int32)
    cell_regions = np.asarray([0, 0, 0], dtype=np.int32)
    face_cells = np.asarray([[0, 1], [1, 2]], dtype=np.int32)

    across_facet = same_region_component_ids(
        cell_sites,
        cell_regions,
        face_cells,
        np.asarray([4, -1], dtype=np.int32),
        np.asarray([0], dtype=np.int8),
    )
    split_site = same_region_component_ids(
        cell_sites,
        cell_regions,
        face_cells,
        np.asarray([-1, -1], dtype=np.int32),
        np.asarray([1], dtype=np.int8),
    )

    np.testing.assert_array_equal(across_facet, [0, 1, 1])
    np.testing.assert_array_equal(split_site, [0, 1, 2])


def test_power_component_lineage_rejects_invalid_face_owners() -> None:
    with pytest.raises(ValueError, match="face owners"):
        same_region_component_ids(
            np.asarray([0], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            np.asarray([[0, 1]], dtype=np.int32),
            np.asarray([-1], dtype=np.int32),
            np.asarray([0], dtype=np.int8),
        )


def test_rank_deficient_power_sites_get_provably_hidden_spanning_guards() -> None:
    sites = np.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]], dtype=np.float64)
    weights = np.asarray([0.0, -0.02], dtype=np.float64)
    domain = np.asarray(
        [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)],
        dtype=np.float64,
    )

    prepared_sites, prepared_weights = _spanning_guard_sites(sites, weights, domain)

    assert not _spans_three_space(sites)
    assert _spans_three_space(prepared_sites)
    np.testing.assert_array_equal(prepared_sites[: len(sites)], sites)
    np.testing.assert_array_equal(prepared_weights[: len(weights)], weights)
    for guard, guard_weight in zip(
        prepared_sites[len(sites) :],
        prepared_weights[len(weights) :],
        strict=True,
    ):
        for point in domain:
            guard_power = sum(
                (Fraction(float(g)) - Fraction(float(x))) ** 2
                for g, x in zip(guard, point, strict=True)
            ) - Fraction(float(guard_weight))
            site_power = sum(
                (Fraction(float(s)) - Fraction(float(x))) ** 2
                for s, x in zip(sites[0], point, strict=True)
            ) - Fraction(float(weights[0]))
            assert guard_power > site_power


def test_full_dimensional_power_sites_are_not_augmented() -> None:
    sites = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    weights = np.zeros((sites.shape[0],), dtype=np.float64)

    prepared_sites, prepared_weights = _spanning_guard_sites(sites, weights, sites)

    assert _spans_three_space(sites)
    assert prepared_sites is sites
    assert prepared_weights is weights
