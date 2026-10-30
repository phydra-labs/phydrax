import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.foams import (
    FoamEquilibriumPlan,
    FoamMaterialPlan,
    PreparedFoamEquilibrium,
)
from phydrax.geometry.multiregion_surface import (
    LabelFieldSurfaceExtractionPlan,
    LabelFieldSurfaceExtractionResult,
    LabelFieldSurfaceExtractionStatus,
    LabelFieldSurfaceLineage,
    MultiRegionSurfaceStatus,
    MultiRegionSurfaceValidationPolicy,
)
from phydrax.threshold_dynamics import LabelFieldState


def _state(labels: np.ndarray, label_ids: tuple[str, ...]) -> LabelFieldState:
    counts = np.bincount(labels.reshape(-1), minlength=len(label_ids))
    return LabelFieldState(
        jnp.asarray(labels, dtype=jnp.int32),
        jnp.asarray(counts > 0),
        jnp.asarray(3, dtype=jnp.int32),
        jnp.asarray(0.125, dtype=jnp.float64),
        label_ids=label_ids,
        route_id="label-extraction-test-route",
        prepared_id="label-extraction-test-prepared",
        site_id="label-extraction-test-sites",
    )


def _image_affine(
    spacing: float, *, origin: tuple[float, float, float] | None = None
) -> np.ndarray:
    affine = np.eye(4)
    affine[:3, :3] *= spacing
    affine[:3, 3] = (
        (0.5 * spacing,) * 3 if origin is None else np.asarray(origin)
    )
    return affine


def _sphere(resolution: int, radius: float = 0.3) -> tuple[np.ndarray, float]:
    spacing = 1.0 / resolution
    axis = (np.arange(resolution) + 0.5) * spacing
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    labels = (np.linalg.norm(points - 0.5, axis=-1) >= radius).astype(np.int32)
    return labels, spacing


def _extract_sphere(resolution: int) -> LabelFieldSurfaceExtractionResult:
    labels, spacing = _sphere(resolution)
    plan = LabelFieldSurfaceExtractionPlan(
        ("bubble", "ambient"),
        index_to_world=_image_affine(spacing),
        boundary_region_id="ambient",
        validation_policy=MultiRegionSurfaceValidationPolicy(
            profile="manifold_two_region"
        ),
    )
    return plan.prepare(labels.shape).extract(_state(labels, ("bubble", "ambient")))


def test_two_phase_sphere_volume_and_area_converge() -> None:
    coarse = _extract_sphere(5)
    fine = _extract_sphere(9)
    assert coarse.accepted and fine.accepted
    volume = 4.0 * np.pi * 0.3**3 / 3.0
    area = 4.0 * np.pi * 0.3**2
    coarse_volume = coarse.evidence.extracted_finite_region_volumes[0]
    fine_volume = fine.evidence.extracted_finite_region_volumes[0]
    coarse_area = coarse.evidence.extracted_pair_areas[0]
    fine_area = fine.evidence.extracted_pair_areas[0]
    assert abs(fine_volume / volume - 1.0) < abs(coarse_volume / volume - 1.0)
    assert fine.lineage.source_binding_id
    assert fine.lineage.source_route_id == "label-extraction-test-route"
    assert fine.lineage.source_prepared_id == "label-extraction-test-prepared"
    assert fine.lineage.source_site_id == "label-extraction-test-sites"
    assert abs(fine_area / area - 1.0) < abs(coarse_area / area - 1.0)
    assert fine.evidence.collision_certified
    assert fine.lineage.source_epoch == 3
    assert fine.lineage.source_time == 0.125


def _three_cell_labels(resolution: int) -> np.ndarray:
    spacing = 1.0 / resolution
    axis = (np.arange(resolution) + 0.5) * spacing
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    centers = np.asarray(((0.25, 0.25, 0.5), (0.75, 0.25, 0.5), (0.5, 0.78, 0.5)))
    return np.argmin(
        np.sum((points[..., None, :] - centers) ** 2, axis=-1), axis=-1
    ).astype(np.int32)


def test_triple_junction_is_oriented_valid_and_prepares_foam_equilibrium() -> None:
    resolution = 3
    labels = _three_cell_labels(resolution)
    spacing = 1.0 / resolution
    policy = MultiRegionSurfaceValidationPolicy(profile="dry_foam")
    plan = LabelFieldSurfaceExtractionPlan(
        ("cell-a", "cell-b", "cell-c"),
        index_to_world=_image_affine(spacing),
        validation_policy=policy,
    )
    result = plan.prepare(labels.shape).extract(
        _state(labels, ("cell-a", "cell-b", "cell-c"))
    )
    assert result.accepted
    assert result.topology is not None
    assert result.state is not None
    assert result.surface is not None
    validation = result.evidence.validation
    assert validation is not None
    assert validation.status is MultiRegionSurfaceStatus.ACCEPTED
    assert validation.label_orientation_consistent
    assert validation.maximum_edge_valence == 3
    assert validation.maximum_vertex_regions == 4
    assert result.evidence.ambiguous_cell_count > 0
    assert result.evidence.ambiguity_resolved
    material = FoamMaterialPlan.soap_film(result.topology.region_ids, 0.025)
    equilibrium = PreparedFoamEquilibrium(
        FoamEquilibriumPlan(), result.surface, material, result.state
    )
    assert equilibrium.surface.prepared_id == result.surface.prepared_id


def test_oblique_reflection_and_tiny_material_survive_image_extraction() -> None:
    resolution = 3
    spacing = 1.0 / resolution
    labels = _three_cell_labels(resolution)
    ids = ("cell-a", "cell-b", "cell-c")
    state = _state(labels, ids)
    policy = MultiRegionSurfaceValidationPolicy(profile="dry_foam")
    base_affine = _image_affine(spacing)
    base = (
        LabelFieldSurfaceExtractionPlan(
            ids, index_to_world=base_affine, validation_policy=policy
        )
        .prepare(labels.shape)
        .extract(state)
    )
    reflection_shear = np.asarray(
        (
            (-1.0, 0.20, 0.05, 1.2),
            (0.15, 0.90, 0.25, -0.4),
            (0.02, 0.10, 1.10, 0.7),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    oblique_affine = reflection_shear @ base_affine
    oblique = (
        LabelFieldSurfaceExtractionPlan(
            ids, index_to_world=oblique_affine, validation_policy=policy
        )
        .prepare(labels.shape)
        .extract(state)
    )
    assert base.accepted and oblique.accepted
    assert base.candidate_seed is not None and oblique.candidate_seed is not None
    source_keys = base.candidate_seed.vertex_source_keys
    assert source_keys is not None
    assert oblique.candidate_seed.vertex_source_keys == source_keys
    assert len(set(source_keys)) == base.candidate_seed.positions.shape[0]
    np.testing.assert_array_equal(
        oblique.candidate_seed.vertex_global_ids,
        base.candidate_seed.vertex_global_ids,
    )
    expected_positions = (
        np.asarray(base.candidate_seed.positions)
        @ reflection_shear[:3, :3].T
        + reflection_shear[:3, 3]
    )
    np.testing.assert_allclose(
        oblique.candidate_seed.positions, expected_positions, atol=1.0e-12
    )
    np.testing.assert_array_equal(
        np.sort(oblique.candidate_seed.faces, axis=1),
        np.sort(base.candidate_seed.faces, axis=1),
    )
    np.testing.assert_array_equal(
        oblique.candidate_seed.face_labels, base.candidate_seed.face_labels
    )
    scale = abs(float(np.linalg.det(reflection_shear[:3, :3])))
    np.testing.assert_allclose(
        oblique.evidence.extracted_finite_region_volumes,
        np.asarray(base.evidence.extracted_finite_region_volumes) * scale,
        rtol=1.0e-10,
    )

    tiny_labels = np.zeros((5, 5, 5), dtype=np.int32)
    tiny_labels[2, 2, 2] = 1
    tiny_ids = ("ambient", "tiny-material")
    tiny_state = _state(tiny_labels, tiny_ids)
    tiny_affine = reflection_shear @ _image_affine(0.2)
    tiny = (
        LabelFieldSurfaceExtractionPlan(
            tiny_ids,
            index_to_world=tiny_affine,
            boundary_region_id="ambient",
            validation_policy=MultiRegionSurfaceValidationPolicy(
                profile="manifold_two_region"
            ),
        )
        .prepare(tiny_labels.shape)
        .extract(tiny_state)
    )
    assert tiny.accepted
    assert tiny.lineage.source_label_counts == (124, 1)
    assert "tiny-material" in tiny.lineage.region_ids
    assert tiny.candidate_seed is not None
    assert np.any(np.all(tiny.candidate_seed.face_labels == (0, 1), axis=1))
    expected_volume = abs(float(np.linalg.det(tiny_affine[:3, :3])))
    np.testing.assert_allclose(
        tiny.evidence.source_finite_region_volumes, (expected_volume,), rtol=1.0e-12
    )


def test_image_topology_grows_from_source_and_sparse_holes_are_unknown() -> None:
    result = _extract_sphere(5)
    assert result.accepted
    capacity = result.evidence.capacity
    assert capacity.admitted
    assert capacity.required.as_mapping() == capacity.capacity.as_mapping()

    plan = LabelFieldSurfaceExtractionPlan(
        ("bubble", "ambient"), index_to_world=np.eye(4)
    )
    incomplete = np.asarray(((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)))
    with pytest.raises(ValueError, match="missing categorical samples"):
        plan.prepare((2, 2, 2), site_coordinates=incomplete)


def test_manifold_profile_reports_unrepresentable_material_valence() -> None:
    labels = np.indices((2, 2, 2)).sum(axis=0).astype(np.int32) % 3
    state = _state(labels, ("a", "b", "c"))
    result = (
        LabelFieldSurfaceExtractionPlan(
            ("a", "b", "c"),
            index_to_world=np.eye(4),
            validation_policy=MultiRegionSurfaceValidationPolicy(
                profile="manifold_two_region"
            ),
        )
        .prepare(labels.shape)
        .extract(state)
    )
    assert result.evidence.status is LabelFieldSurfaceExtractionStatus.UNSUPPORTED_VALENCE
    assert result.evidence.unsupported_edge_count > 0
    assert result.surface is None


def test_sparse_site_order_does_not_change_extracted_surface() -> None:
    labels, spacing = _sphere(4)
    coordinates = (
        np.stack(np.meshgrid(*(np.arange(4),) * 3, indexing="ij"), axis=-1).reshape(
            (-1, 3)
        )
        + 1
    )
    flat = labels.reshape(-1)
    permutation = np.random.default_rng(11).permutation(flat.size)
    plan = LabelFieldSurfaceExtractionPlan(
        ("bubble", "ambient"),
        index_to_world=_image_affine(
            spacing,
            origin=(-0.5 * spacing, -0.5 * spacing, -0.5 * spacing),
        ),
        boundary_region_id="ambient",
    )
    first = plan.prepare((6, 6, 6), site_coordinates=coordinates).extract(
        _state(flat, ("bubble", "ambient"))
    )
    second = plan.prepare((6, 6, 6), site_coordinates=coordinates[permutation]).extract(
        _state(flat[permutation], ("bubble", "ambient"))
    )
    assert first.accepted and second.accepted
    assert first.lineage.source_state_id == second.lineage.source_state_id
    assert first.candidate_seed is not None and second.candidate_seed is not None
    assert first.candidate_seed.seed_id == second.candidate_seed.seed_id
    np.testing.assert_array_equal(first.candidate_seed.faces, second.candidate_seed.faces)
    np.testing.assert_allclose(
        first.candidate_seed.positions, second.candidate_seed.positions
    )


def test_lineage_constructor_rejects_noncanonical_contract_identifier() -> None:
    with pytest.raises(ValueError, match="source_state_id"):
        LabelFieldSurfaceLineage(
            source_label_ids=("bubble", "ambient"),
            source_label_counts=(1, 7),
            region_ids=("bubble", "ambient"),
            region_source_label_indices=(0, 1),
            boundary_region_id="ambient",
            source_epoch=0,
            source_time=0.0,
            source_state_id=" noncanonical ",
            source_binding_id="binding",
            source_route_id="route",
            source_prepared_id="source-prepared",
            source_site_id="source-sites",
            prepared_id="prepared",
            candidate_seed_id=None,
        )
