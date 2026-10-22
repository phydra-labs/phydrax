# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Compare bounded exact sphere arrangements against independent native clipping."""

from fractions import Fraction
from typing import TypedDict

import numpy as np
import pytest

import phydrax.discretization._sphere_chart_deformation as owner
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
    CoordinateLiveStorage,
)
from phydrax.discretization._sphere_chart_piece_bank import (
    SphereChartPieceBankBuilder,
)
from phydrax.geometry._sphere_material_atlas import (
    SphereMaterialCellAtlas,
    SphereProjectiveActionBank,
    SphereProjectiveReferenceMap,
)
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.sphere_material import accepted_sphere_material


type AtlasPair = tuple[SphereMaterialCellAtlas, SphereMaterialCellAtlas]
type PolygonBoundary = tuple[
    tuple[owner._Point, ...],
    tuple[int, ...],
    tuple[owner._VertexKey, ...],
    tuple[int | None, ...],
]
type PolygonResult = tuple[*PolygonBoundary, int]


class FanMetadata(TypedDict):
    source_cell_global_id: int
    target_cell_global_id: int
    source_row: int
    target_row: int
    source_atlas_row: int
    target_atlas_row: int
    geometry_entity_id: str
    occurrence_path: tuple[str, ...]
    native_boundary_labels: tuple[int, ...]
    native_clip_id: str


@pytest.fixture(scope="module")
def atlases() -> AtlasPair:
    return accepted_sphere_material(1.2)[2], accepted_sphere_material(1.0)[2]


def _oracle(
    source: SphereMaterialCellAtlas,
    target: SphereMaterialCellAtlas,
    source_row: int,
    target_row: int,
    actions: SphereProjectiveActionBank,
    storage: CoordinateLiveStorage,
    budget: CoordinateEnclosureBudget,
) -> tuple[SphereProjectiveReferenceMap, PolygonResult]:
    mapping = source.projective_reference_map(
        int(source.cell_global_ids[source_row]), target,
        int(target.cell_global_ids[target_row]), budget=budget, actions=actions,
    )
    result = owner._native_polygon(
        mapping, 10**8, 10**7, owner._support_edges(actions, source_row, target_row),
        {}, storage, source_row=source_row,
        target_corner_keys=tuple(actions.corner_key(True, target_row, slot) for slot in range(3)),
        target_corner_points={}, budget=budget,
    )
    return mapping, result


@pytest.mark.parametrize(
    ("first", "second"),
    (
        ((2, 3, 5), (7, 11, 13)),
        ((2, 3, 5), (-4, -6, -10)),
        ((0, 0, 0), (7, 11, 13)),
    ),
)
def test_constructed_cross_retains_only_its_actual_numeric_zeros(
    atlases: AtlasPair, first: owner._Integer3, second: owner._Integer3,
) -> None:
    source, target = atlases
    with coordinate_reference_budget() as budget, budget.live_storage() as storage:
        actions = SphereProjectiveActionBank.authenticate(
            source, target, budget=budget, storage=storage
        )
        arrangement = owner._SphereArrangementOperands(actions, storage, budget=budget)
        owner._SphereEdgeOverlay(arrangement)._begin(10**7, 10**8)
        cross = arrangement._meet(first, second)
        assert sum(a * b for a, b in zip(cross, first, strict=True)) == 0
        assert sum(a * b for a, b in zip(cross, second, strict=True)) == 0
        arrangement._release_prepare_products()
        before = budget.work_units
        assert arrangement._dot(cross, first) == 0
        assert arrangement._dot(second, cross) == 0
        assert budget.work_units - before == 2
        # An unrelated current operand still executes its first exact dot.
        unrelated = (19, 23, 29)
        before = budget.work_units
        assert arrangement._dot(first, unrelated) == sum(
            a * b for a, b in zip(first, unrelated, strict=True)
        )
        assert budget.work_units - before == 6
        assert arrangement.cross_zeros
        another = owner._SphereArrangementOperands(actions, storage, budget=budget)
        assert not another.cross_zeros


def _cyclic_equivalence(actual: PolygonResult, expected: PolygonResult) -> int:
    assert len(actual[0]) == len(expected[0])
    if not actual[0]:
        return 0
    count = len(actual[0])
    shift = next(
        offset for offset in range(count)
        if all(actual[0][slot] == expected[0][(slot + offset) % count] for slot in range(count))
    )
    assert actual[1] == tuple(expected[1][(slot + shift) % count] for slot in range(count))
    return shift


def test_candidate_arrangement_matches_independent_exact_clipping_and_coverage(
    atlases: AtlasPair,
) -> None:
    source, target = atlases
    source_coverage = [Fraction(0)] * source.num_charts
    target_coverage = [Fraction(0)] * target.num_charts
    with coordinate_reference_budget() as budget, budget.live_storage() as storage:
        actions = SphereProjectiveActionBank.authenticate(source, target, budget=budget, storage=storage)
        sources, targets, orientations = owner._candidates(source, target, 10**6, 10**8, budget=budget)
        arrangement = owner._SphereArrangementOperands(actions, storage, budget=budget)
        overlay = owner._SphereEdgeOverlay(arrangement, orientations)
        temporary_before = budget.temporary_bytes_upper
        live_before = storage.bound
        overlay.prepare(sources, targets, maximum_work=10**7, maximum_memory=10**8)
        assert budget.temporary_bytes_upper == (
            temporary_before + storage.bound - live_before
        )
        for source_row, target_row in zip(sources, targets, strict=True):
            source_row, target_row = int(source_row), int(target_row)
            actual = overlay.polygon(source_row, target_row, maximum_work=10**7, maximum_memory=10**8)
            mapping, expected = _oracle(source, target, source_row, target_row, actions, storage, budget)
            shift = _cyclic_equivalence(actual, expected)
            polygon = actual[0]
            if not polygon:
                continue
            images = mapping.reference_images(polygon)
            assert owner._sphere_arrangement_images(
                arrangement, source_row, target_row, polygon, actual[2]
            ) == images
            for slot, corner in enumerate(actual[3]):
                expected_corner = expected[3][(slot + shift) % len(polygon)]
                if expected_corner is not None:
                    assert corner == expected_corner
                if corner is not None:
                    assert images[slot][1] == owner._TARGET_CORNERS[corner]
            source_areas = owner._fan_areas(polygon)
            target_areas = owner._projective_fan_areas(mapping, images, source_areas)
            assert target_areas == owner._fan_areas(tuple(point for _, point in images))
            assert all(area > 0 for area in source_areas)
            assert all(area > 0 for area in target_areas)
            source_coverage[source_row] += sum(source_areas, Fraction(0))
            target_coverage[target_row] += sum(target_areas, Fraction(0))
    assert source_coverage == [Fraction(1, 2)] * source.num_charts
    assert target_coverage == [Fraction(1, 2)] * target.num_charts


def test_coincident_edges_and_exact_zero_area_contacts(atlases: AtlasPair) -> None:
    atlas = atlases[1]
    corners = np.asarray(atlas.physical_corner_global_ids)
    edges = {}
    pair = None
    for row, vertices in enumerate(corners):
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(vertices[first]), int(vertices[second]))))
            if edge in edges:
                pair = edges[edge], row
                break
            edges[edge] = row
        if pair is not None:
            break
    assert pair is not None
    with coordinate_reference_budget() as budget, budget.live_storage() as storage:
        actions = SphereProjectiveActionBank.authenticate(atlas, atlas, budget=budget, storage=storage)
        arrangement = owner._SphereArrangementOperands(actions, storage, budget=budget)
        overlay = owner._SphereEdgeOverlay(arrangement)
        overlay.prepare(
            (pair[0], pair[0]), (pair[0], pair[1]),
            maximum_work=10**7, maximum_memory=10**8,
        )
        coincident = overlay.polygon(pair[0], pair[0], maximum_work=10**7, maximum_memory=10**8)
        _, expected = _oracle(atlas, atlas, pair[0], pair[0], actions, storage, budget)
        _cyclic_equivalence(coincident, expected)
        assert sum(owner._fan_areas(coincident[0]), Fraction(0)) == Fraction(1, 2)
        touching = overlay.polygon(*pair, maximum_work=10**7, maximum_memory=10**8)
        _, contact_oracle = _oracle(atlas, atlas, *pair, actions, storage, budget)
        assert touching[0] == contact_oracle[0] == ()


def _first_arrangement(
    source: SphereMaterialCellAtlas,
    target: SphereMaterialCellAtlas,
    work_limit: int | None = None,
) -> tuple[PolygonBoundary, int]:
    with coordinate_reference_budget() as budget, budget.live_storage() as storage:
        actions = SphereProjectiveActionBank.authenticate(source, target, budget=budget, storage=storage)
        sources, targets, orientations = owner._candidates(source, target, 10**6, 10**8, budget=budget)
        start = budget.work_units
        with budget.bound_stage(10**7 if work_limit is None else work_limit, 10**8):
            arrangement = owner._SphereArrangementOperands(actions, storage, budget=budget)
            overlay = owner._SphereEdgeOverlay(arrangement, orientations)
            overlay.prepare(sources, targets, maximum_work=10**7, maximum_memory=10**8)
            actual = overlay.polygon(int(sources[0]), int(targets[0]), maximum_work=10**7, maximum_memory=10**8)
        return actual[:4], budget.required_work_units - start


def test_arrangement_exact_observed_work_boundary(atlases: AtlasPair) -> None:
    expected, required = _first_arrangement(*atlases)
    assert required > 0
    actual, _ = _first_arrangement(*atlases, work_limit=required)
    assert actual == expected
    with pytest.raises(CoordinateEnclosureResourceError):
        _first_arrangement(*atlases, work_limit=required - 1)


def test_native_orientation_intermediates_reject_changed_current_ray_bytes(
    atlases: AtlasPair,
) -> None:
    source, target = atlases
    with coordinate_reference_budget() as budget, budget.live_storage() as storage:
        sources, targets, numeric = owner._candidates(
            source, target, 10**6, 10**8, budget=budget
        )
        assert numeric is not None
        actions = SphereProjectiveActionBank.authenticate(
            source, target, budget=budget, storage=storage
        )
        original = source.directions
        changed = np.array(original, copy=True)
        changed.flat[0] = np.nextafter(changed.flat[0], np.inf)
        object.__setattr__(source, "directions", changed)
        try:
            overlay = owner._SphereEdgeOverlay(
                owner._SphereArrangementOperands(actions, storage, budget=budget), numeric
            )
            with pytest.raises(ValueError, match="stale or mismatched rays"):
                overlay.prepare(
                    sources, targets, maximum_work=10**7, maximum_memory=10**8
                )
        finally:
            object.__setattr__(source, "directions", original)


def test_native_orientation_intermediates_are_not_cross_request_authority(
    atlases: AtlasPair,
) -> None:
    source, target = atlases
    with coordinate_reference_budget() as producer:
        sources, targets, numeric = owner._candidates(
            source, target, 10**6, 10**8, budget=producer
        )
    assert numeric is not None
    with coordinate_reference_budget() as consumer, consumer.live_storage() as storage:
        actions = SphereProjectiveActionBank.authenticate(
            source, target, budget=consumer, storage=storage
        )
        overlay = owner._SphereEdgeOverlay(
            owner._SphereArrangementOperands(actions, storage, budget=consumer), numeric
        )
        with pytest.raises(ValueError, match="this completed numeric operation"):
            overlay.prepare(sources, targets, maximum_work=10**7, maximum_memory=10**8)


def test_native_orientation_intermediates_bind_only_one_actual_operation(
    atlases: AtlasPair,
) -> None:
    source, target = atlases
    with coordinate_reference_budget() as budget, budget.live_storage() as storage:
        sources, targets, numeric = owner._candidates(
            source, target, 10**6, 10**8, budget=budget
        )
        assert numeric is not None
        first = SphereProjectiveActionBank.authenticate(
            source, target, budget=budget, storage=storage
        )
        owner._SphereEdgeOverlay(
            owner._SphereArrangementOperands(first, storage, budget=budget), numeric
        ).prepare(sources, targets, maximum_work=10**7, maximum_memory=10**8)
        second = SphereProjectiveActionBank.authenticate(
            source, target, budget=budget, storage=storage
        )
        with pytest.raises(ValueError, match="this completed numeric operation"):
            owner._SphereEdgeOverlay(
                owner._SphereArrangementOperands(second, storage, budget=budget), numeric
            ).prepare(sources, targets, maximum_work=10**7, maximum_memory=10**8)


def test_radial_displacement_terms_release_scratch_and_retain_exact_row_arrays(
    atlases: AtlasPair,
) -> None:
    from phydrax.geometry import _domain_numerics

    atlas = atlases[0]
    directions = np.asarray(atlas.directions)
    lows, highs = _domain_numerics.interval_vector_norm((directions, directions))
    expected = np.max(
        np.maximum(
            np.maximum(np.nextafter(highs - 1.0, np.inf), np.nextafter(1.0 - lows, np.inf)),
            0.0,
        ),
        axis=1,
    )
    with coordinate_reference_budget() as observed:
        before = observed.temporary_bytes_upper
        actual = owner._sphere_radial_displacement_terms(atlas, budget=observed)
        assert observed.temporary_bytes_upper == before
        assert observed.retained_basis_bytes > 0
        np.testing.assert_array_equal(actual.unit_error, expected)
        required = observed.required_work_units
    with coordinate_reference_budget() as admitted:
        with admitted.bound_stage(required, 10**8):
            np.testing.assert_array_equal(
                owner._sphere_radial_displacement_terms(atlas, budget=admitted).unit_error,
                expected,
            )
    with coordinate_reference_budget() as refused:
        with pytest.raises(CoordinateEnclosureResourceError):
            with refused.bound_stage(required - 1, 10**8):
                owner._sphere_radial_displacement_terms(atlas, budget=refused)


def test_actual_polygon_fan_bulk_rows_equal_individual_canonical_rows(
    atlases: AtlasPair,
) -> None:
    from phydrax.discretization._sphere_chart_piece_bank import (
        SphereChartPieceBankBuilder,
    )

    source, target = atlases
    with coordinate_reference_budget() as budget, budget.live_storage() as storage:
        sources, targets, numeric = owner._candidates(
            source, target, 10**6, 10**8, budget=budget
        )
        actions = SphereProjectiveActionBank.authenticate(
            source, target, budget=budget, storage=storage
        )
        operands = owner._SphereArrangementOperands(actions, storage, budget=budget)
        overlay = owner._SphereEdgeOverlay(operands, numeric)
        overlay.prepare(sources, targets, maximum_work=10**7, maximum_memory=10**8)
        for source_row, target_row in zip(sources, targets, strict=True):
            source_row, target_row = int(source_row), int(target_row)
            polygon, labels, rays, _, _ = overlay.polygon(
                source_row, target_row, maximum_work=10**7, maximum_memory=10**8
            )
            if len(polygon) > 3:
                break
        else:
            raise AssertionError("actual independent native meshes must yield a genuine nontriangle overlap")
        mapping = source.projective_reference_map(
            int(source.cell_global_ids[source_row]), target,
            int(target.cell_global_ids[target_row]), actions=actions, budget=budget,
        )
        images = owner._sphere_arrangement_images(
            operands, source_row, target_row, polygon, rays
        )
        source_areas = owner._fan_areas(polygon)
        target_areas = owner._projective_fan_areas(mapping, images, source_areas)
        bounds = tuple(
            owner._sphere_piece_geometry(mapping, (images[0], images[slot], images[slot + 1]))[0]
            for slot in range(1, len(polygon) - 1)
        )
        builder = SphereChartPieceBankBuilder(budget=budget)
        try:
            expected_source = builder._prepare_kernel_points(polygon)
            expected_target = builder._prepare_kernel_points(image[1] for image in images)
            actual_source = operands._prepare_kernel_points(builder, source_row, rays)
            actual_target = operands._prepare_kernel_points(
                builder, target_row, rays, target=True
            )
            assert all(
                actual is expected
                for actual, expected in zip(actual_source, expected_source, strict=True)
            )
            assert all(
                actual is expected
                for actual, expected in zip(actual_target, expected_target, strict=True)
            )
        finally:
            builder.close()
    clip = owner.canonical_fingerprint({
        "kind": "sphere-exact-great-circle-arrangement-boundary",
        "map": mapping.map_id, "vertices": owner._tokens(polygon), "support_labels": labels,
    })
    metadata = FanMetadata(
        source_cell_global_id=int(source.cell_global_ids[source_row]),
        target_cell_global_id=int(target.cell_global_ids[target_row]),
        source_row=int(source.physical_rows[source_row]),
        target_row=int(target.physical_rows[target_row]),
        source_atlas_row=source_row, target_atlas_row=target_row,
        geometry_entity_id=source.geometry_entity_ids[source_row],
        occurrence_path=source.occurrence_paths[source_row],
        native_boundary_labels=labels, native_clip_id=clip,
    )
    # These distinct conservative values exercise storage, not a new
    # adaptation acceptance or a change to its displacement controls.
    displacements = tuple(10.0 + slot for slot in range(len(source_areas)))
    ids = tuple(owner.canonical_fingerprint({
        "kind": "sphere-exact-reference-piece", "clip": clip,
        "source": owner._tokens((polygon[0], polygon[slot], polygon[slot + 1])),
        "target": owner._tokens((images[0][1], images[slot][1], images[slot + 1][1])),
        "indices": (0, slot, slot + 1),
    }) for slot in range(1, len(polygon) - 1))

    results, work = [], []
    for bulk in (False, True):
        with coordinate_reference_budget() as budget:
            builder = SphereChartPieceBankBuilder(budget=budget)
            try:
                mapped = builder._prepare_kernel_map(mapping)
                old = builder._prepare_kernel_points(polygon)
                new = builder._prepare_kernel_points(image[1] for image in images)
                if bulk:
                    builder._append_kernel_fan(
                        mapped, old, new, **metadata,
                        projective_bounds=bounds,
                        source_reference_areas=source_areas,
                        target_reference_areas=target_areas,
                        displacement_bounds=displacements, piece_ids=ids,
                    )
                else:
                    for index, slot in enumerate(range(1, len(polygon) - 1)):
                        builder._append_kernel_piece(
                            mapped, (old[0], old[slot], old[slot + 1]),
                            (new[0], new[slot], new[slot + 1]), **metadata,
                            projective_bounds=bounds[index],
                            source_reference_area=source_areas[index],
                            target_reference_area=target_areas[index],
                            native_polygon_vertex_indices=(0, slot, slot + 1),
                            displacement_bound=displacements[index], piece_id=ids[index],
                        )
                results.append(builder.finish())
                work.append(budget.work_units)
            finally:
                builder.close()
    assert results[0].bank_id == results[1].bank_id
    assert work[1] < work[0]
    assert tuple(piece.displacement_bound for piece in results[1]) == displacements
    np.testing.assert_array_equal(results[0].piece_rows, results[1].piece_rows)
    np.testing.assert_array_equal(results[0].reference_vertices, results[1].reference_vertices)


def _arrangement_point_reuse(
    atlases: AtlasPair, *, limit: int | None = None,
) -> tuple[int, int]:
    with coordinate_reference_budget() as budget, budget.live_storage() as storage:
        actions = SphereProjectiveActionBank.authenticate(
            *atlases, budget=budget, storage=storage
        )
        arrangement = owner._SphereArrangementOperands(actions, storage, budget=budget)
        owner._SphereEdgeOverlay(arrangement)._begin(10**7, 10**8)
        rays = arrangement._cell(False, 0).rays
        builder = SphereChartPieceBankBuilder(budget=budget, maximum_memory_bytes=10**8)
        try:
            before = budget.work_units
            first = arrangement._prepare_kernel_points(builder, 0, rays)
            first_work = budget.work_units - before
            before = budget.work_units
            with budget.bound_stage(10**7 if limit is None else limit, 10**8):
                second = arrangement._prepare_kernel_points(builder, 0, rays)
            assert all(a is b for a, b in zip(first, second, strict=True))
            return first_work, budget.work_units - before
        finally:
            builder.close()


def test_arrangement_reuses_owned_point_tokens_at_exact_work_boundary(
    atlases: AtlasPair,
) -> None:
    first_work, required = _arrangement_point_reuse(atlases)
    assert 0 < required < first_work
    assert _arrangement_point_reuse(atlases, limit=required) == (first_work, required)
    with pytest.raises(CoordinateEnclosureResourceError):
        _arrangement_point_reuse(atlases, limit=required - 1)


def test_arrangement_point_tokens_reject_another_builder_or_request(
    atlases: AtlasPair,
) -> None:
    with coordinate_reference_budget() as budget, budget.live_storage() as storage:
        actions = SphereProjectiveActionBank.authenticate(
            *atlases, budget=budget, storage=storage
        )
        arrangement = owner._SphereArrangementOperands(actions, storage, budget=budget)
        owner._SphereEdgeOverlay(arrangement)._begin(10**7, 10**8)
        rays = arrangement._cell(False, 0).rays
        first = SphereChartPieceBankBuilder(budget=budget, maximum_memory_bytes=10**8)
        second = SphereChartPieceBankBuilder(budget=budget, maximum_memory_bytes=10**8)
        try:
            tokens = arrangement._prepare_kernel_points(first, 0, rays)
            with pytest.raises(ValueError, match="actual live builder"):
                arrangement._prepare_kernel_points(second, 0, rays)
            with pytest.raises(ValueError):
                second._prepare_kernel_points(tokens)
            with coordinate_reference_budget() as other_budget:
                foreign = SphereChartPieceBankBuilder(
                    budget=other_budget, maximum_memory_bytes=10**8
                )
                try:
                    with pytest.raises(ValueError, match="actual live builder"):
                        arrangement._prepare_kernel_points(foreign, 0, rays)
                finally:
                    foreign.close()
        finally:
            first.close()
            second.close()


def test_private_point_token_does_not_trust_mutated_public_fraction() -> None:
    with coordinate_reference_budget() as budget:
        builder = SphereChartPieceBankBuilder(budget=budget, maximum_memory_bytes=10**8)
        try:
            point = (Fraction(1, 7), Fraction(2, 9))
            token = builder._prepare_kernel_points((point,))[0]
            object.__setattr__(point[0], "_numerator", 3)
            fresh = builder._prepare_kernel_points((point,))[0]
            assert token.exact == ((1, 7), (2, 9))
            assert fresh.exact == ((3, 7), (2, 9))
            assert builder._prepare_kernel_points((token,))[0] is token
        finally:
            builder.close()
