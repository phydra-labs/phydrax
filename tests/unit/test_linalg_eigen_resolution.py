from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _general(matrix: Any, mass: Any = None) -> Any:
    operator = phx.linalg.DenseLinearOperator(jnp.asarray(matrix))
    mass_operator = (
        None if mass is None else phx.linalg.DenseLinearOperator(jnp.asarray(mass))
    )
    return phx.linalg.eigen.general_eigensolve(
        phx.linalg.eigen.GeneralEigenproblem(operator, mass_operator)
    )


def _assert_successful(result: Any) -> None:
    assert bool(result.successful)
    assert bool(result.diagnostics.converged)
    assert bool(result.diagnostics.output_finite)
    assert jnp.all(jnp.isfinite(result.diagnostics.right_relative_residuals))
    assert jnp.all(jnp.isfinite(result.diagnostics.left_relative_residuals))
    assert jnp.max(result.diagnostics.right_relative_residuals) < 1e-8
    assert jnp.max(result.diagnostics.left_relative_residuals) < 1e-8


def test_linalg_eigen_resolution_scenario_1() -> None:
    coarse = _general(jnp.diag(jnp.asarray([1.0, 2.0, 4.0])))
    fine = _general(jnp.diag(jnp.asarray([4.0, 1.0, 2.0 + 1e-9])))
    _assert_successful(coarse)
    _assert_successful(fine)
    report = phx.linalg.eigen.compare_general_eigen_resolutions(coarse, fine)

    assert report.matched_count == 3
    assert report.trusted_count == 3
    assert len(set(np.asarray(report.fine_indices).tolist())) == 3
    assert jnp.max(report.chordal_distances) < 1e-8
    matrix = jnp.diag(jnp.asarray([2.0, 3.0]))
    mass = jnp.diag(jnp.asarray([1.0, 0.0]))
    coarse = _general(matrix, mass)
    fine = _general(matrix + jnp.diag(jnp.asarray([1e-10, 0.0])), mass)
    _assert_successful(coarse)
    _assert_successful(fine)
    report = phx.linalg.eigen.compare_general_eigen_resolutions(coarse, fine)

    assert report.matched_count == 2
    assert jnp.count_nonzero(report.homogeneous_classes == 1) == 1
    assert jnp.all(report.matched_mask)
    coarse = _general(jnp.diag(jnp.asarray([1.0, 1.0, 3.0])))
    fine = _general(jnp.diag(jnp.asarray([1.0, 1.0, 3.0])))
    _assert_successful(coarse)
    _assert_successful(fine)
    report = phx.linalg.eigen.compare_general_eigen_resolutions(coarse, fine)

    ambiguous = int(phx.linalg.eigen.GeneralEigenMatchStatus.AMBIGUOUS_CLUSTER)
    assert jnp.count_nonzero(report.statuses == ambiguous) == 2
    assert report.trusted_count == 1
    domain = phx.discretization.AxisDomain.periodic(0.0, 1.0)
    coarse = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(5),)
    ).prepare((domain,))
    fine = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(7),)
    ).prepare((domain,))
    coarse_operator = phx.discretization.spectral_derivative_operator(
        coarse,
        0,
    ).operator
    fine_operator = phx.discretization.spectral_derivative_operator(fine, 0).operator
    coarse_result = phx.linalg.eigen.general_eigensolve(
        phx.linalg.eigen.GeneralEigenproblem(coarse_operator)
    )
    fine_result = phx.linalg.eigen.general_eigensolve(
        phx.linalg.eigen.GeneralEigenproblem(fine_operator)
    )
    _assert_successful(coarse_result)
    _assert_successful(fine_result)
    transfer = phx.discretization.prepare_spectral_modal_transfer(coarse, fine)
    report = phx.discretization.compare_spectral_eigen_resolutions(
        coarse_result,
        fine_result,
        coarse,
        fine,
        transfer,
    )

    assert report.trusted_count == coarse.num_modes
    assert jnp.max(report.subspace_errors) < 1e-10
    domain = phx.discretization.AxisDomain.periodic(0.0, 1.0)
    space = phx.discretization.TensorSpectralPlan(
        (phx.discretization.FourierBasisPlan(5),)
    ).prepare((domain,))
    operator = phx.discretization.spectral_derivative_operator(space, 0).operator
    result = phx.linalg.eigen.general_eigensolve(
        phx.linalg.eigen.GeneralEigenproblem(operator)
    )
    lattice = phx.discretization.spectral.LatticeHarmonicPlan.parallelogramic(
        (3,), (9,)
    ).prepare(jnp.asarray(((2.0, 0.0),)))
    transfer = phx.discretization.prepare_spectral_modal_transfer(lattice, lattice)
    with pytest.raises(ValueError, match="does not bind the supplied spaces"):
        phx.discretization.compare_spectral_eigen_resolutions(
            result, result, space, space, transfer
        )
