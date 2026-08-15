#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


la = phx.linalg
eig = la.eigen


def _problem(matrix, operator_id):
    return eig.SchurEigenproblem(
        la.DenseLinearOperator(jnp.asarray(matrix), operator_id=operator_id)
    )


def test_schur_observables_distinguish_spectral_and_numerical_abscissae():
    matrix = jnp.asarray([[-2.0, 10.0], [0.0, -1.0]])
    observables = eig.schur_spectral_observables(
        _problem(matrix, "nonnormal-observables")
    )

    assert observables.status == int(eig.SpectralObservableStatus.SUCCESS)
    assert observables.successful
    assert observables.spectral_radius == pytest.approx(2.0)
    assert observables.spectral_abscissa == pytest.approx(-1.0)
    assert observables.numerical_abscissa > 3.5
    assert observables.continuous_time_stability == int(
        eig.SpectralStabilityStatus.STABLE
    )
    assert observables.discrete_time_stability == int(
        eig.SpectralStabilityStatus.UNSTABLE
    )
    assert observables.continuous_stability_margin == pytest.approx(1.0)
    assert observables.discrete_stability_margin == pytest.approx(-1.0)
    assert observables.departure_from_normality > 0.9


def test_spectral_algebraic_observables_match_dense_invariants():
    matrix = jnp.asarray(
        [
            [1.0 + 1.0j, 2.0, 0.0],
            [0.0, -2.0 + 0.5j, 1.0j],
            [0.0, 0.0, 0.5 - 0.25j],
        ]
    )
    result = eig.schur_eigensolve(_problem(matrix, "algebraic-observables"))
    observables = eig.schur_spectral_observables(result)
    eigenvalues = jnp.linalg.eigvals(matrix)

    assert observables.status == int(eig.SpectralObservableStatus.SUCCESS)
    assert jnp.allclose(observables.trace, jnp.trace(matrix))
    assert jnp.allclose(observables.determinant, jnp.linalg.det(matrix))
    assert observables.log_absolute_determinant == pytest.approx(
        float(jnp.sum(jnp.log(jnp.abs(eigenvalues))))
    )
    assert jnp.allclose(
        observables.determinant_phase,
        jnp.linalg.det(matrix) / jnp.abs(jnp.linalg.det(matrix)),
    )
    assert observables.frobenius_norm == pytest.approx(float(jnp.linalg.norm(matrix)))
    assert observables.spectral_centroid == pytest.approx(complex(jnp.trace(matrix) / 3))


def test_singular_spectrum_has_exact_phase_and_log_determinant_semantics():
    matrix = jnp.diag(jnp.asarray([0.0, 0.5, 2.0]))
    observables = eig.schur_spectral_observables(_problem(matrix, "singular-observables"))

    assert observables.status == int(eig.SpectralObservableStatus.SUCCESS)
    assert observables.singular
    assert observables.determinant == 0.0
    assert observables.determinant_phase == 0.0
    assert jnp.isneginf(observables.log_absolute_determinant)
    assert observables.determinant_finite
    assert observables.discrete_time_stability == int(
        eig.SpectralStabilityStatus.UNSTABLE
    )


def test_stability_tolerance_defines_marginal_bands():
    continuous_matrix = jnp.diag(jnp.asarray([-1e-4, -0.5]))
    continuous = eig.schur_spectral_observables(
        _problem(continuous_matrix, "continuous-marginal-observables"),
        stability_tolerance=1e-3,
    )
    discrete_matrix = jnp.diag(jnp.asarray([1.0 + 2e-4, 0.5]))
    discrete = eig.schur_spectral_observables(
        _problem(discrete_matrix, "discrete-marginal-observables"),
        stability_tolerance=1e-3,
    )
    assert continuous.continuous_time_stability == int(
        eig.SpectralStabilityStatus.MARGINAL
    )
    assert continuous.discrete_time_stability == int(eig.SpectralStabilityStatus.STABLE)
    assert discrete.continuous_time_stability == int(eig.SpectralStabilityStatus.UNSTABLE)
    assert discrete.discrete_time_stability == int(eig.SpectralStabilityStatus.MARGINAL)
    with pytest.raises(ValueError, match="non-negative"):
        eig.schur_spectral_observables(
            _problem(continuous_matrix, "bad-tolerance-observables"),
            stability_tolerance=-1.0,
        )


def test_observables_propagate_source_tolerance_failure_without_hiding_values():
    matrix = jnp.asarray([[1.0, 8.0, 0.0], [-2.0, 1.0, 3.0], [0.0, 0.0, 4.0]])
    policy = eig.SchurSolvePolicy(
        tolerance=eig.SchurTolerancePolicy(relative=0.0, absolute=0.0)
    )
    result = eig.schur_eigensolve(
        _problem(matrix, "failed-source-observables"),
        policy=policy,
    )
    observables = eig.schur_spectral_observables(result)

    assert result.status == int(eig.SchurSolveStatus.RESIDUAL_TOLERANCE_NOT_MET)
    assert observables.status == int(eig.SpectralObservableStatus.SOURCE_FAILURE)
    assert observables.finite
    assert jnp.isfinite(observables.spectral_radius)


def test_prepared_observables_are_jittable_and_preserve_provenance():
    problem = _problem(jnp.asarray([[-0.5, 2.0], [0.0, -0.25]]), "jit-observables")
    prepared = eig.prepare_schur_eigensolve(problem)
    compiled = jax.jit(eig.schur_spectral_observables)
    observables = compiled(prepared)

    assert observables.status == int(eig.SpectralObservableStatus.SUCCESS)
    assert observables.provenance.plan_id == prepared.plan.plan_id
    assert observables.provenance.prepared_id == prepared.prepared_id
    assert observables.provenance.numeric_version == 0
