#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as scipy_sparse

import phydrax as phx
from phydrax.discretization import (
    BasisTransformPlan,
    OperatorSpectrum,
    SpectralDecomposition,
)


def test_spectral_discretization_obeys_weighted_analysis_synthesis_identities():
    weights = np.array([0.4, 0.7, 1.1, 0.8])
    raw = np.eye(4) / np.sqrt(weights)[:, None]
    plan = SpectralDecomposition.from_eigenpairs(
        np.array([0.0, 1.0, 1.0, 3.0]),
        raw,
        weights,
        decomposition_id="weighted-coordinate-basis",
    )
    projection = np.asarray(plan.synthesis @ plan.analysis)

    assert np.allclose(plan.analysis, plan.synthesis.T * weights[None, :])
    assert np.allclose(plan.analysis @ plan.synthesis, np.eye(4), atol=1e-12)
    assert np.allclose(projection, np.eye(4), atol=1e-12)
    assert np.array_equal(plan.group_ids, np.array([0, 1, 1, 2]))
    assert plan.decomposition_id == "weighted-coordinate-basis"


def test_one_modal_transform_supports_distinct_operator_spectra():
    decomposition = SpectralDecomposition.from_eigenpairs(
        np.array([0.0, 1.0, 4.0, 9.0]),
        np.eye(4),
        np.ones((4,)),
        decomposition_id="shared-transform",
    )
    discrete = OperatorSpectrum(
        decomposition.transform,
        "fd2-laplacian",
        np.array([0.0, 0.75, 2.0, 3.25]),
        classification="discrete",
    )

    assert discrete.transform_id == decomposition.transform_id
    assert discrete.spectrum_id != decomposition.spectrum_id
    assert discrete.classification == "discrete"
    assert np.array_equal(discrete.nullspace_mask, np.array([True, False, False, False]))


def test_trigonometric_transforms_and_spherical_spectrum_are_independent():
    dct = phx.discretization.trigonometric_modal_transform("dct", 2, 8)
    dst = phx.discretization.trigonometric_modal_transform("dst", 1, 8)
    values = jnp.arange(8.0)
    tensor = phx.discretization.TensorModalTransform((dct, dst))
    tensor_values = jnp.arange(64.0).reshape((8, 8))
    degrees = jnp.asarray([0, 1, 1, 1, 2, 2, 2, 2])
    spherical = phx.discretization.spherical_laplacian_spectrum(
        dct,
        degrees,
        radius=2.0,
    )

    assert jnp.allclose(dct.synthesize(dct.analyze(values)), values, atol=1e-12)
    assert jnp.allclose(dst.synthesize(dst.analyze(values)), values, atol=1e-12)
    assert jnp.allclose(
        tensor.synthesize(tensor.analyze(tensor_values)),
        tensor_values,
        atol=1e-12,
    )
    assert spherical.transform_id == dct.transform_id
    assert jnp.array_equal(
        spherical.modal_values,
        -degrees * (degrees + 1) / 4.0,
    )
    assert jnp.array_equal(spherical.nullspace_mask, degrees == 0)


def test_transform_diagonal_solve_projects_only_under_explicit_policy():
    matrix = jnp.diag(jnp.asarray([0.0, 2.0, 3.0]))
    space = phx.linalg.ArraySpace((3,))
    operator = phx.linalg.DenseLinearOperator(
        matrix,
        source=space,
        target=space,
    )
    decomposition = SpectralDecomposition.from_eigenpairs(
        jnp.asarray([0.0, 2.0, 3.0]),
        jnp.eye(3),
        jnp.ones((3,)),
        decomposition_id="diagonal-solve",
    )
    representation = decomposition.diagonal_representation(operator)
    prepared = phx.linalg.TransformDiagonalSolvePlan(
        representation,
        compatibility="project_rhs",
        gauge="minimum_norm",
    ).prepare()

    result = prepared.solve(jnp.asarray([1.0, 4.0, 9.0]))

    assert jnp.allclose(result.value, jnp.asarray([0.0, 2.0, 3.0]))
    assert jnp.allclose(result.compatibility_residual, 1.0)
    assert jnp.allclose(result.residual_norm, 0.0)
    assert result.converged


def test_truncated_spectral_discretization_is_a_weighted_self_adjoint_projection():
    weights = np.array([0.5, 0.7, 1.0, 1.2, 0.9])
    rng = np.random.default_rng(23)
    candidate = rng.normal(size=(5, 3))
    gram = candidate.T @ (weights[:, None] * candidate)
    inverse_root = np.linalg.inv(np.linalg.cholesky(gram)).T
    eigenvectors = candidate @ inverse_root
    plan = SpectralDecomposition.from_eigenpairs(
        np.array([0.0, 1.0, 2.0]),
        eigenvectors,
        weights,
    )
    projection = np.asarray(plan.synthesis @ plan.analysis)
    mass = np.diag(weights)

    assert np.allclose(plan.analysis @ plan.synthesis, np.eye(3), atol=1e-12)
    assert np.allclose(projection @ projection, projection, atol=1e-12)
    assert np.allclose(mass @ projection, projection.T @ mass, atol=1e-12)


def test_sparse_stiffness_solver_recovers_path_graph_low_modes():
    count = 300
    stiffness = scipy_sparse.diags(
        (
            -np.ones(count - 1),
            np.concatenate(([1.0], np.full(count - 2, 2.0), [1.0])),
            -np.ones(count - 1),
        ),
        offsets=(-1, 0, 1),
        format="csr",
    )
    mass = scipy_sparse.eye(count, format="csr")

    plan = SpectralDecomposition.from_stiffness(
        stiffness,
        mass,
        n_modes=5,
        decomposition_id="path-300",
    )
    expected = 2.0 - 2.0 * np.cos(np.pi * np.arange(5) / count)

    assert np.allclose(plan.eigenvalues, expected, rtol=2e-7, atol=2e-10)
    assert np.allclose(plan.analysis @ plan.synthesis, np.eye(5), atol=1e-7)
    assert np.allclose(
        stiffness @ np.asarray(plan.synthesis),
        np.asarray(plan.synthesis) * np.asarray(plan.eigenvalues)[None, :],
        rtol=1e-6,
        atol=1e-8,
    )


def test_basis_transform_plan_reconstructs_its_modal_subspace_and_differentiates():
    nodes = jnp.arange(17, dtype=float) / 17.0
    weights = jnp.full((17,), 1.0 / 17.0)
    plan = BasisTransformPlan(
        (nodes,),
        (weights,),
        (True,),
        ("fourier",),
        (9,),
    )
    coefficients = jnp.arange(9.0)
    values = plan.synthesis_matrices[0] @ coefficients
    reconstructed = plan.synthesis_matrices[0] @ (plan.analysis_matrices[0] @ values)
    gradient = jax.grad(lambda field: jnp.sum((plan.analysis_matrices[0] @ field) ** 2))(
        values
    )

    assert jnp.allclose(
        plan.analysis_matrices[0] @ plan.synthesis_matrices[0],
        jnp.eye(9),
        rtol=1e-12,
        atol=1e-12,
    )
    assert jnp.allclose(reconstructed, values, rtol=1e-12, atol=1e-12)
    assert jnp.all(jnp.isfinite(gradient))


def test_modal_plans_reject_invalid_geometry_and_memory_limits():
    with pytest.raises(ValueError, match="positive"):
        SpectralDecomposition.from_eigenpairs(
            np.array([0.0]),
            np.ones((2, 1)),
            np.array([1.0, 0.0]),
        )
    with pytest.raises(ValueError, match="one entry per point"):
        SpectralDecomposition.from_stiffness(
            np.eye(3),
            np.ones((2,)),
            n_modes=2,
        )
    with pytest.raises(ValueError, match="max_construction_bytes"):
        SpectralDecomposition.from_stiffness(
            np.eye(4),
            np.ones((4,)),
            n_modes=2,
            max_construction_bytes=1,
        )
    with pytest.raises(ValueError, match="within available nodes"):
        BasisTransformPlan(
            (jnp.arange(4.0),),
            (None,),
            (False,),
            ("legendre",),
            (5,),
        )
    with pytest.raises(ValueError, match="max_construction_bytes"):
        BasisTransformPlan(
            (jnp.arange(4.0),),
            (None,),
            (False,),
            ("legendre",),
            (2,),
            max_construction_bytes=1,
        )
