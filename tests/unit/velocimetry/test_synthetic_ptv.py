#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.velocimetry.synthetic import (
    generate_ptv_case,
    PTVScenarioKind,
    PTVScenarioPlan,
)


def _small_plan(kind: PTVScenarioKind, **overrides) -> PTVScenarioPlan:
    options = {
        "image_shape": (28, 30),
        "frame_count": 3,
        "camera_count": 2,
        "particle_capacity": 6,
        "particle_count": 4,
        "focal_length": 32.0,
        "camera_baseline": 0.5,
        "particle_diameter": 1.6,
        "seed": 29,
    }
    options.update(overrides)
    return PTVScenarioPlan(kind, **options)


def test_baseline_generation_is_deterministic_and_uses_world_xyz_truth() -> None:
    plan = _small_plan(PTVScenarioKind.BASELINE, read_noise_std=0.02)

    first = generate_ptv_case(plan)
    second = generate_ptv_case(plan)

    assert first.scenario_id == second.scenario_id
    assert first.world_coordinate_convention == "right-handed-x-y-z"
    assert first.images.shape == (3, 2, 28, 30)
    assert first.world_positions_xyz.shape == (3, 6, 3)
    assert first.visible.shape == (3, 2, 6)
    assert first.projection_status.shape == (3, 2, 6)
    np.testing.assert_array_equal(first.images, second.images)
    np.testing.assert_array_equal(first.world_positions_xyz, second.world_positions_xyz)
    np.testing.assert_array_equal(first.trajectory_ids, second.trajectory_ids)
    assert bool(jnp.all(jnp.isfinite(first.images)))
    assert bool(jnp.all(first.trajectory_ids[: plan.particle_count] >= 0))
    assert bool(jnp.all(first.trajectory_ids[plan.particle_count :] == -1))


def test_calibration_and_refraction_keep_true_and_nominal_rigs_explicit() -> None:
    calibration = generate_ptv_case(
        _small_plan(PTVScenarioKind.CALIBRATION, calibration_perturbation=0.02)
    )
    refraction = generate_ptv_case(
        _small_plan(PTVScenarioKind.REFRACTION, refractive_index=1.2)
    )

    assert not np.array_equal(
        np.asarray(calibration.true_rig.cameras[0].intrinsics.focal_length),
        np.asarray(calibration.nominal_rig.cameras[0].intrinsics.focal_length),
    )
    assert refraction.true_rig.cameras[0].refraction is not None
    assert refraction.nominal_rig.cameras[0].refraction is None
    assert bool(jnp.any(refraction.projection_valid))


def test_degenerate_rays_use_coincident_camera_centers() -> None:
    case = generate_ptv_case(_small_plan(PTVScenarioKind.DEGENERATE_RAYS))

    np.testing.assert_allclose(
        case.true_rig.cameras[0].pose.frame.translation,
        case.true_rig.cameras[1].pose.frame.translation,
    )
    np.testing.assert_allclose(
        case.projection_pixels_rc[:, 0],
        case.projection_pixels_rc[:, 1],
    )


def test_crossings_occlusions_births_deaths_and_dense_truth_are_materialized() -> None:
    crossings = generate_ptv_case(_small_plan(PTVScenarioKind.CROSSINGS))
    occlusion = generate_ptv_case(
        _small_plan(PTVScenarioKind.OCCLUSION, occlusion_radius=2.0)
    )
    lifecycle = generate_ptv_case(_small_plan(PTVScenarioKind.BIRTHS_DEATHS))
    dense_plan = _small_plan(
        PTVScenarioKind.DENSE,
        particle_capacity=10,
        particle_count=None,
    )
    dense = generate_ptv_case(dense_plan)

    np.testing.assert_allclose(
        crossings.world_positions_xyz[1, 0],
        crossings.world_positions_xyz[1, 1],
    )
    assert int(jnp.sum(occlusion.visible[0, 0])) < int(
        jnp.sum(occlusion.particle_active[0])
    )
    assert bool(jnp.any(lifecycle.particle_active[0] != lifecycle.particle_active[-1]))
    assert dense_plan.particle_count == dense_plan.particle_capacity
    assert int(jnp.sum(dense.particle_active[0])) == dense_plan.particle_capacity


@pytest.mark.parametrize(
    "kind",
    [
        PTVScenarioKind.BASELINE,
        PTVScenarioKind.CALIBRATION,
        PTVScenarioKind.REFRACTION,
        PTVScenarioKind.DEGENERATE_RAYS,
        PTVScenarioKind.CROSSINGS,
        PTVScenarioKind.OCCLUSION,
        PTVScenarioKind.BIRTHS_DEATHS,
        PTVScenarioKind.DENSE,
    ],
)
def test_all_ptv_families_remain_finite(kind: PTVScenarioKind) -> None:
    particle_count = None if kind is PTVScenarioKind.DENSE else 4
    case = generate_ptv_case(_small_plan(kind, particle_count=particle_count))

    assert case.evidence.finite
    assert bool(jnp.all(jnp.isfinite(case.world_positions_xyz)))
    assert bool(jnp.all(jnp.isfinite(case.images)))
