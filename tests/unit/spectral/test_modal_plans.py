#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as scipy_sparse

from phydrax._spectral._modal import BasisTransformPlan, SpectralDiscretization


def test_spectral_discretization_obeys_weighted_analysis_synthesis_identities():
    weights = np.array([0.4, 0.7, 1.1, 0.8])
    raw = np.eye(4) / np.sqrt(weights)[:, None]
    plan = SpectralDiscretization.from_eigenpairs(
        np.array([0.0, 1.0, 1.0, 3.0]),
        raw,
        weights,
        basis_id="weighted-coordinate-basis",
    )
    projection = np.asarray(plan.synthesis @ plan.analysis)

    assert np.allclose(plan.analysis, plan.synthesis.T * weights[None, :])
    assert np.allclose(plan.analysis @ plan.synthesis, np.eye(4), atol=1e-12)
    assert np.allclose(projection, np.eye(4), atol=1e-12)
    assert np.array_equal(plan.group_ids, np.array([0, 1, 1, 2]))
    assert plan.basis_id == "weighted-coordinate-basis"


def test_truncated_spectral_discretization_is_a_weighted_self_adjoint_projection():
    weights = np.array([0.5, 0.7, 1.0, 1.2, 0.9])
    rng = np.random.default_rng(23)
    candidate = rng.normal(size=(5, 3))
    gram = candidate.T @ (weights[:, None] * candidate)
    inverse_root = np.linalg.inv(np.linalg.cholesky(gram)).T
    eigenvectors = candidate @ inverse_root
    plan = SpectralDiscretization.from_eigenpairs(
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

    plan = SpectralDiscretization.from_stiffness(
        stiffness,
        mass,
        n_modes=5,
        basis_id="path-300",
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
        SpectralDiscretization.from_eigenpairs(
            np.array([0.0]),
            np.ones((2, 1)),
            np.array([1.0, 0.0]),
        )
    with pytest.raises(ValueError, match="one entry per point"):
        SpectralDiscretization.from_stiffness(
            np.eye(3),
            np.ones((2,)),
            n_modes=2,
        )
    with pytest.raises(ValueError, match="max_construction_bytes"):
        SpectralDiscretization.from_stiffness(
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
