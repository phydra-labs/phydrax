# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Observable inverse contracts: confounding, correlated QC and leakage vetoes."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.interactive_column_inference import twin_data, twin_experiment, twin_model
from phydrax._array_archive import write_array_archive
from phydrax._fingerprint import array_tree_fingerprint
from phydrax.applications.geophysics._inference import (
    column_local_information,
    ColumnCalibrationProblem,
    ColumnObservationData,
    load_column_inference,
)
from phydrax.applications.geophysics._observations import prepare_geophysical_observations


def test_linear_gaussian_combinations_expose_unconstrained_direction():
    with jax.enable_x64(True):
        result = column_local_information(
            lambda z: jnp.atleast_1d(2 * z[0] + z[1]), jnp.ones(2)
        )
        assert int(result.rank) == 1
        null = jnp.asarray([1.0, -2.0])
        np.testing.assert_allclose(
            result.unidentifiable_projector @ null, null, atol=1e-12
        )
        np.testing.assert_allclose(result.fisher @ null, 0.0, atol=1e-12)
        # The identified combination has variance 1/||[2,1]||², NOT finite
        # posterior variance in the orthogonal unconstrained direction.
        direction = jnp.asarray([2.0, 1.0]) / jnp.sqrt(5.0)
        np.testing.assert_allclose(
            direction @ result.covariance_on_identifiable_subspace @ direction,
            0.2,
            atol=1e-12,
        )
        full = column_local_information(
            lambda z: jnp.asarray([2 * z[0] + z[1], z[0] - 2 * z[1]]), jnp.ones(2)
        )
        assert int(full.rank) == 2
        np.testing.assert_allclose(
            full.covariance_on_identifiable_subspace, 0.2 * jnp.eye(2), atol=1e-12
        )


def test_correlated_missing_observations_use_principal_covariance_not_zero_residuals():
    with jax.enable_x64(True):
        plan, initial, _, _ = twin_model()
        experiment = twin_experiment(
            initial, steps=2, signals=("temperature",), times=(0.0,)
        )
        binding = experiment.bindings[0]
        product = prepare_geophysical_observations(
            binding.operator,
            (0.0,),
            [[285.0, np.nan]],
            [[2.0, np.nan]],
            quantity=binding.operator.quantity,
            time=experiment.time,
            target_support_id=binding.operator.transfer.target.support_id,
            availability=np.asarray([[True, False]]),
            case_id="explicit-missing-profile",
        )
        data = ColumnObservationData(
            experiment,
            (product,),
            provenance="user-supplied-profile-with-missing-second-level",
            role="calibration",
            covariance=[[4.0, 1.5], [1.5, 1.0]],
        )
        np.testing.assert_array_equal(data.indices, [0])
        np.testing.assert_allclose(data.whitener @ jnp.asarray([2.0]), [1.0], atol=1e-12)
        np.testing.assert_allclose(data.covariance.matrix, [[4.0]], atol=1e-12)
        with pytest.raises(ValueError, match="marginal"):
            ColumnObservationData(
                experiment,
                (product,),
                provenance="wrong-marginal",
                role="calibration",
                covariance=[[9.0, 1.5], [1.5, 1.0]],
            )
        assert bool(experiment.predict(plan).successful)


def test_holdout_role_and_duplicate_products_cannot_enter_fit():
    with jax.enable_x64(True):
        plan, initial, space, truth = twin_model()
        experiment = twin_experiment(
            initial, steps=2, signals=("temperature",), times=(0.0, 2.0)
        )
        holdout = twin_data(experiment, space.apply(plan, truth), role="holdout")
        with pytest.raises(ValueError, match="calibration-role"):
            ColumnCalibrationProblem(plan, space, (holdout,))
        training = ColumnObservationData(
            experiment,
            holdout.products,
            provenance="explicit-training",
            role="calibration",
        )
        with pytest.raises(ValueError, match="double-count"):
            ColumnCalibrationProblem(plan, space, (training, training))
        with pytest.raises(ValueError, match="outside"):
            space.check(space.upper + space.scales)


def test_phase_boundary_is_not_reported_as_a_regular_inverse_map():
    with jax.enable_x64(True):
        plan, _, space, truth = twin_model()
        initial = plan.initialize(
            [100.0, 110.0],
            [0.2, 0.4],
            [plan.thermodynamics.reference_temperature, 290.0],
            [100.0, 100.0],
            surface_temperature=295.0,
        )
        experiment = twin_experiment(
            initial, steps=2, signals=("temperature",), times=(0.0, 2.0)
        )
        data = twin_data(experiment, space.apply(plan, truth), role="calibration")
        problem = ColumnCalibrationProblem(plan, space, (data,))
        assert not bool(problem.information(truth).derivative_valid)
        assert not bool(problem.fisher_action(truth, jnp.ones_like(truth)).valid)
        # A numerical admission failure must not look like an exact zero residual.
        bad = eqx.tree_at(lambda p: p.rain_fall_speed, plan, jnp.asarray(10000.0))
        failed_problem = ColumnCalibrationProblem(bad, space, (data,))
        assert np.all(np.isnan(failed_problem.residual(truth / space.scales)))


@pytest.mark.parametrize("index", [True, -1, 0.5, 1])
def test_inference_archive_rejects_invalid_continuation_index_before_loading_state(
    tmp_path, index
):
    with jax.enable_x64(True):
        plan, initial, space, truth = twin_model()
        experiment = twin_experiment(
            initial, steps=2, signals=("temperature",), times=(0.0,)
        )
        data = twin_data(experiment, space.apply(plan, truth), role="calibration")
        problem = ColumnCalibrationProblem(plan, space, (data,))
        arrays = {"parameters": truth}
        path = tmp_path / "invalid-index.npz"
        # A malformed external manifest must be rejected before it can choose a
        # different experiment or attempt to open any physical-state checkpoint.
        write_array_archive(
            path,
            manifest={
                "kind": "interactive-column-inference",
                "problem_id": problem.problem_id,
                "checkpoint": path.name + ".column.npz",
                "evidence_id": array_tree_fingerprint(arrays),
                "experiment_index": index,
            },
            arrays=arrays,
        )
        with pytest.raises(ValueError, match="experiment index"):
            load_column_inference(path, problem)
