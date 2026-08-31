import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _general(matrix, mass=None):
    operator = phx.linalg.DenseLinearOperator(jnp.asarray(matrix))
    mass_operator = (
        None if mass is None else phx.linalg.DenseLinearOperator(jnp.asarray(mass))
    )
    return phx.linalg.eigen.general_eigensolve(
        phx.linalg.eigen.GeneralEigenproblem(operator, mass_operator)
    )


def test_homogeneous_resolution_matching_is_one_to_one_and_permutation_safe():
    coarse = _general(jnp.diag(jnp.asarray([1.0, 2.0, 4.0])))
    fine = _general(jnp.diag(jnp.asarray([4.0, 1.0, 2.0 + 1e-9])))
    report = phx.linalg.eigen.compare_general_eigen_resolutions(coarse, fine)

    assert report.matched_count == 3
    assert report.trusted_count == 3
    assert len(set(np.asarray(report.fine_indices).tolist())) == 3
    assert jnp.max(report.chordal_distances) < 1e-8


def test_homogeneous_resolution_matching_preserves_infinite_classification():
    matrix = jnp.diag(jnp.asarray([2.0, 3.0]))
    mass = jnp.diag(jnp.asarray([1.0, 0.0]))
    coarse = _general(matrix, mass)
    fine = _general(matrix + jnp.diag(jnp.asarray([1e-10, 0.0])), mass)
    report = phx.linalg.eigen.compare_general_eigen_resolutions(coarse, fine)

    assert report.matched_count == 2
    assert jnp.count_nonzero(report.homogeneous_classes == 1) == 1
    assert jnp.all(report.matched_mask)


def test_repeated_cluster_is_not_certified_as_individual_modes():
    coarse = _general(jnp.diag(jnp.asarray([1.0, 1.0, 3.0])))
    fine = _general(jnp.diag(jnp.asarray([1.0, 1.0, 3.0])))
    report = phx.linalg.eigen.compare_general_eigen_resolutions(coarse, fine)

    ambiguous = int(phx.linalg.eigen.GeneralEigenMatchStatus.AMBIGUOUS_CLUSTER)
    assert jnp.count_nonzero(report.statuses == ambiguous) == 2
    assert report.trusted_count == 1


def test_spectral_eigenspace_evidence_compares_transferred_modes():
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
