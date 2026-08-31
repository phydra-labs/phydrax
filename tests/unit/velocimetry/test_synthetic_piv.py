#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from dataclasses import fields

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.velocimetry.synthetic import (
    generate_piv_case,
    PIVScenarioKind,
    PIVScenarioPlan,
    PIVSyntheticCase,
)


def _small_plan(kind: PIVScenarioKind, **overrides) -> PIVScenarioPlan:
    options = {
        "image_shape": (24, 28),
        "particle_capacity": 32,
        "particle_density": 0.02,
        "particle_diameter": 1.8,
        "seed": 17,
    }
    options.update(overrides)
    return PIVScenarioPlan(kind, **options)


def test_translation_truth_uses_row_then_column_components() -> None:
    case = generate_piv_case(
        _small_plan(
            PIVScenarioKind.TRANSLATION,
            displacement_rc=(2.0, -1.25),
        )
    )

    valid_displacement = np.asarray(case.truth.displacement_rc[case.truth.valid])
    np.testing.assert_allclose(
        valid_displacement,
        np.broadcast_to((2.0, -1.25), valid_displacement.shape),
    )
    np.testing.assert_allclose(
        np.asarray(case.second_positions_rc - case.first_positions_rc),
        np.broadcast_to((2.0, -1.25), case.first_positions_rc.shape),
    )
    assert case.image_pair.geometry.coordinate_convention == "row-down-column-right"


@pytest.mark.parametrize(
    "kind",
    [
        PIVScenarioKind.NO_MOTION,
        PIVScenarioKind.AFFINE,
        PIVScenarioKind.SHEAR,
        PIVScenarioKind.ROTATION,
        PIVScenarioKind.SPATIAL_FREQUENCY,
    ],
)
def test_motion_families_generate_finite_fixed_shape_truth(
    kind: PIVScenarioKind,
) -> None:
    case = generate_piv_case(
        _small_plan(
            kind,
            displacement_rc=(0.2, -0.1),
            affine_gradient_rc=(0.01, -0.02, 0.03, 0.01),
            shear=0.04,
            rotation_radians=0.06,
            spatial_amplitude_rc=(1.0, 1.5),
            spatial_frequency_rc=(1.0, 3.0),
        )
    )

    assert case.image_pair.first.shape == (24, 28)
    assert case.truth.displacement_rc.shape == (24, 28, 2)
    assert bool(jnp.all(jnp.isfinite(case.truth.displacement_rc)))
    if kind is PIVScenarioKind.NO_MOTION:
        assert bool(jnp.all(case.truth.displacement_rc == 0.0))
    else:
        assert bool(jnp.any(case.truth.displacement_rc != 0.0))


def test_density_diameter_noise_dropout_mask_and_boundary_are_deterministic() -> None:
    plan = _small_plan(
        PIVScenarioKind.TRANSLATION,
        particle_density=0.035,
        particle_diameter=2.4,
        read_noise_std=0.03,
        dropout_probability=1.0,
        mask_fraction=0.16,
        boundary_fraction=0.5,
        seed=911,
    )

    first = generate_piv_case(plan)
    second = generate_piv_case(plan)

    assert first.scenario_id == second.scenario_id
    np.testing.assert_array_equal(first.image_pair.first, second.image_pair.first)
    np.testing.assert_array_equal(first.image_pair.second, second.image_pair.second)
    np.testing.assert_array_equal(first.first_active, second.first_active)
    np.testing.assert_array_equal(first.second_active, second.second_active)
    assert int(jnp.sum(first.first_active)) == plan.particle_count
    assert int(jnp.sum(first.second_active)) < plan.particle_count
    assert bool(jnp.any(~first.image_pair.first_mask))
    assert bool(jnp.any(first.first_rasterization.evidence.truncated))
    assert first.evidence.finite


def test_piv_case_exposes_no_tracking_identity_field() -> None:
    field_names = {field.name for field in fields(PIVSyntheticCase)}
    assert "trajectory_ids" not in field_names
    assert "track_ids" not in field_names
    assert "particle_ids" not in field_names


def test_plan_rejects_capacity_that_cannot_represent_requested_density() -> None:
    with pytest.raises(ValueError, match="capacity"):
        PIVScenarioPlan(
            PIVScenarioKind.TRANSLATION,
            image_shape=(32, 32),
            particle_capacity=8,
            particle_density=0.1,
        )
