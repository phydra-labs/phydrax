#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from phydrax.applications.skeletal_muscle.personalization import (
    commit_physical_relative_force_calibration,
    PhysicalRelativeForceCalibrationPlan,
    PhysicalRelativeForceCalibrationStatus,
)


def test_identifiable_scale_and_offset_are_recovered_in_newtons():
    relative = jnp.asarray([0.0, 0.1, 0.25, 0.45, 0.7, 1.0])
    nuisance = jnp.ones((relative.shape[0], 1))
    prepared = PhysicalRelativeForceCalibrationPlan(
        nuisance,
        ("load-cell-zero",),
        protocol_id="mvc-ramp-2026-09-03",
        asset_id="load-cell-LC-17-cal-2026-08",
    ).prepare()
    state = prepared.initialize(100.0)
    observed = 520.0 * relative + 3.25
    uncertainty = jnp.full_like(relative, 0.5)
    candidate = prepared.evaluate(state, relative, observed, uncertainty)

    assert bool(candidate.evidence.successful)
    assert bool(candidate.evidence.scale_identifiable)
    assert float(candidate.evidence.nuisance_confounding_fraction) < 1.0
    committed = commit_physical_relative_force_calibration(candidate, state)
    np.testing.assert_allclose(
        committed.scale_newton_per_relative_force, 520.0, rtol=2e-5
    )
    np.testing.assert_allclose(committed.nuisance_coefficients_newton, [3.25], rtol=2e-5)
    observation = prepared.observe(committed, jnp.asarray([0.2, 0.8]))
    np.testing.assert_allclose(observation.force_newton, [104.0, 416.0], rtol=2e-5)
    assert observation.protocol_id == "mvc-ramp-2026-09-03"
    assert observation.asset_id == "load-cell-LC-17-cal-2026-08"


def test_nuisance_column_equal_to_relative_force_is_rejected_as_unidentifiable():
    relative = jnp.linspace(0.0, 1.0, 8)
    prepared = PhysicalRelativeForceCalibrationPlan(
        relative[:, None],
        ("confounded-gain",),
        protocol_id="negative-control",
        asset_id="synthetic-asset",
    ).prepare()
    state = prepared.initialize(250.0)
    candidate = prepared.evaluate(
        state,
        relative,
        400.0 * relative,
        jnp.ones_like(relative),
    )

    assert not bool(candidate.evidence.successful)
    assert not bool(candidate.evidence.scale_identifiable)
    assert int(candidate.evidence.status) & int(
        PhysicalRelativeForceCalibrationStatus.SCALE_NOT_IDENTIFIABLE
    )
    rolled_back = commit_physical_relative_force_calibration(candidate, state)
    assert float(rolled_back.scale_newton_per_relative_force) == 250.0
    assert int(rolled_back.calibration_epoch) == 0


def test_masked_protocol_samples_retain_identifiability_and_explicit_uncertainty():
    relative = jnp.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    nuisance = jnp.ones((6, 1))
    prepared = PhysicalRelativeForceCalibrationPlan(
        nuisance,
        ("offset",),
        protocol_id="masked-ramp",
        asset_id="force-plate-2",
    ).prepare()
    state = prepared.initialize(1.0)
    observed = 300.0 * relative - 2.0
    observed = observed.at[2].set(jnp.nan)
    uncertainty = jnp.ones_like(relative)
    candidate = prepared.evaluate(
        state,
        relative,
        observed,
        uncertainty,
        sample_mask=jnp.asarray([True, True, False, True, True, True]),
    )

    assert bool(candidate.evidence.successful)
    assert int(candidate.evidence.sample_count) == 5
    np.testing.assert_allclose(
        candidate.proposed.scale_newton_per_relative_force, 300.0, rtol=2e-5
    )
