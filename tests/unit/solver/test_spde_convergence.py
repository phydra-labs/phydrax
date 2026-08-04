import jax.numpy as jnp
import pytest

import phydrax as phx


def _strong_level(resolution, error, coupling_id="shared-driver"):
    return phx.solver.SPDEConvergenceLevel(
        resolution,
        work=1.0 / resolution,
        strong_error=error,
        error_budget=phx.solver.SPDEErrorBudget(
            temporal=error,
            sampling=0.1 * error,
        ),
        mean_square=1.0 + error,
        realization_id="global-realization",
        coupling_id=coupling_id,
    )


def test_spde_convergence_study_reports_rates_and_requires_shared_coupling():
    levels = (
        _strong_level(0.05, 0.0025),
        _strong_level(0.2, 0.04),
        _strong_level(0.1, 0.01),
    )
    study = phx.solver.SPDEConvergenceStudy(
        "time",
        levels,
        reference_id="analytic-reference",
    )

    assert jnp.array_equal(study.resolutions, jnp.asarray([0.2, 0.1, 0.05]))
    assert jnp.allclose(study.pairwise_rates(), jnp.asarray([2.0, 2.0]))
    assert study.regression_rate() == pytest.approx(2.0)
    assert study.sampling_is_subordinate()
    assert study.mean_square_stable(upper_bound=1.05)

    with pytest.raises(ValueError, match="share one explicit coupling_id"):
        phx.solver.SPDEConvergenceStudy(
            "time",
            (
                _strong_level(0.2, 0.04, coupling_id="coarse-driver"),
                _strong_level(0.1, 0.01, coupling_id="fine-driver"),
            ),
            reference_id="invalid-strong-comparison",
        )


def test_weak_observable_and_noise_truncation_keep_sampling_and_cutoff_distinct():
    samples = jnp.asarray([[1.0, -1.0], [2.0, 0.0], [3.0, 1.0], [4.0, 2.0]])
    weak = phx.solver.weak_observable_estimate(
        samples,
        lambda value: jnp.sum(value**2),
        8.0,
        name="energy",
    )
    study = phx.solver.NoiseTruncationStudy.from_compatible_spectrum(
        jnp.asarray([1.0, 0.4, 0.1]),
        jnp.asarray([-0.5, -2.0, -4.0]),
        (0, 1, 2, 3),
        horizon=0.25,
        operator_id="stable-linear-operator",
        basis_id="ordered-noise-basis",
        observable_mode_weights={"mean": jnp.asarray([1.0, 0.0, 0.0])},
    )

    raw = jnp.asarray([level.raw_covariance_residual for level in study.levels])
    finite_horizon = jnp.asarray(
        [level.finite_horizon_solution_residual for level in study.levels]
    )

    assert weak.estimate == pytest.approx(9.0)
    assert weak.standard_error > 0.0
    assert raw[0] > raw[1] > raw[2] > raw[3]
    assert finite_horizon[0] > finite_horizon[1] > finite_horizon[2]
    assert finite_horizon[3] == 0.0
    assert study.levels[0].stationary_solution_residual is not None
    assert study.recommended_rank(0.11, metric="raw") == 2
    assert study.recommended_rank(0.0, metric="finite_horizon") == 3
    assert study.levels[1].weak_observable_residuals["mean"] == 0.0
