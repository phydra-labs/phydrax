#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._trainable import partition_trainable


def _normalized_value_and_derivative(coefficients, normalization, coordinate):
    center = normalization.center[0]
    scale = normalization.matrix[0, 0]
    normalized = scale * (coordinate - center)
    value = coefficients[0] + coefficients[1] * normalized
    derivative = coefficients[1] * scale
    return value, derivative


def test_homogeneous_constraints_preserve_holomorphy_and_train_only_nullspace():
    constraints = (
        phx.equations.HolomorphicPointConstraint.dirichlet(-1.0, 0.0),
        phx.equations.HolomorphicPointConstraint.dirichlet(1.0, 0.0),
        phx.equations.HolomorphicPointConstraint.dirichlet(
            0.0,
            0.0,
            component="imaginary",
        ),
    )
    prepared = phx.equations.HolomorphicPolynomialConstraintPlan(
        3,
        constraints,
    ).prepare()
    free = jnp.linspace(-0.3, 0.4, prepared.evidence.nullity)
    potential = phx.equations.ConstrainedHolomorphicPolynomialPotential(
        prepared,
        initial_free_coordinates=free,
    )

    assert prepared.evidence.rank == 3
    assert prepared.evidence.nullity == 5
    assert (
        jnp.linalg.norm(potential.constraint_residual())
        <= prepared.evidence.lift_tolerance
    )
    assert jnp.allclose(jnp.real(potential(-1.0)[0]), 0.0, atol=1e-12)
    assert jnp.allclose(jnp.real(potential(1.0)[0]), 0.0, atol=1e-12)
    assert jnp.allclose(jnp.imag(potential(0.0)[0]), 0.0, atol=1e-12)

    certificate = potential.holomorphic_certificate()
    assert certificate.parameter_coverage == "finite-subspace"
    assert certificate.linear_in_parameters
    assert certificate.construction_dependencies == (prepared.prepared_id,)
    trainable, _ = partition_trainable(potential)
    leaves = jax.tree.leaves(trainable)
    assert len(leaves) == 1
    assert leaves[0].shape == (prepared.evidence.nullity,)
    assert not jnp.iscomplexobj(leaves[0])

    coordinate = 0.2 - 0.15j
    jet = potential.jet(coordinate, 2)
    first = jax.jacfwd(lambda value: potential(value)[0], holomorphic=True)
    second = jax.jacfwd(first, holomorphic=True)
    assert jnp.allclose(jet.derivative(1)[0], first(coordinate), atol=1e-12)
    assert jnp.allclose(jet.derivative(2)[0], second(coordinate), atol=1e-12)

    harmonic = phx.equations.HarmonicPotential2D(potential)
    point = jnp.asarray([0.2, -0.15])
    assert jnp.allclose(jnp.trace(jax.hessian(harmonic)(point)), 0.0, atol=1e-11)


def test_affine_mixed_constraints_recover_unique_normalized_polynomial():
    normalization = phx.equations.ComplexAffineNormalization.scalar(
        center=0.25 - 0.1j,
        scale=1.5 + 0.4j,
    )
    coefficients = jnp.asarray([1.2 - 0.7j, -0.4 + 1.1j])
    value_point = -0.2 + 0.3j
    normal_point = 0.4 - 0.25j
    robin_point = -0.3 - 0.1j
    normal = jnp.asarray([0.6, 0.8])
    robin_normal = jnp.asarray([-0.8, 0.6])
    value, _ = _normalized_value_and_derivative(
        coefficients,
        normalization,
        value_point,
    )
    _, derivative = _normalized_value_and_derivative(
        coefficients,
        normalization,
        normal_point,
    )
    robin_value, robin_derivative = _normalized_value_and_derivative(
        coefficients,
        normalization,
        robin_point,
    )
    directional = (normal[0] + 1j * normal[1]) * derivative
    robin_functional = (
        0.75 * robin_value
        - 0.4 * (robin_normal[0] + 1j * robin_normal[1]) * robin_derivative
    )
    constraints = (
        phx.equations.HolomorphicPointConstraint.dirichlet(
            value_point,
            jnp.real(value),
        ),
        phx.equations.HolomorphicPointConstraint.dirichlet(
            value_point,
            jnp.imag(value),
            component="imaginary",
        ),
        phx.equations.HolomorphicPointConstraint.normal_derivative(
            normal_point,
            normal,
            jnp.real(directional),
        ),
        phx.equations.HolomorphicPointConstraint.robin(
            robin_point,
            robin_normal,
            jnp.imag(robin_functional),
            value_weight=0.75,
            normal_weight=-0.4,
            component="imaginary",
        ),
    )
    prepared = phx.equations.HolomorphicPolynomialConstraintPlan(
        1,
        constraints,
        normalization=normalization,
    ).prepare()
    potential = phx.equations.ConstrainedHolomorphicPolynomialPotential(prepared)

    assert prepared.evidence.rank == 4
    assert prepared.evidence.nullity == 0
    assert jnp.allclose(potential.coefficients[0], coefficients, atol=2e-12)
    assert (
        jnp.linalg.norm(potential.constraint_residual())
        <= prepared.evidence.lift_tolerance
    )
    certificate = potential.holomorphic_certificate()
    assert certificate.parameter_coverage == "finite-parametric-family"
    assert not certificate.linear_in_parameters
    assert certificate.parameter_mode == "real-cartesian-nullspace"


def test_rank_deficiency_is_exposed_and_inconsistent_constraints_fail_closed():
    duplicate = phx.equations.HolomorphicPointConstraint.dirichlet(0.0, 1.0)
    prepared = phx.equations.HolomorphicPolynomialConstraintPlan(
        1,
        (duplicate, duplicate),
    ).prepare()
    potential = phx.equations.ConstrainedHolomorphicPolynomialPotential(prepared)

    assert prepared.evidence.rank == 1
    assert prepared.evidence.nullity == 3
    assert jnp.allclose(jnp.real(potential(0.0)[0]), 1.0, atol=1e-12)
    assert (
        jnp.linalg.norm(potential.constraint_residual())
        <= prepared.evidence.lift_tolerance
    )
    certificate = potential.holomorphic_certificate()
    assert certificate.parameter_coverage == "finite-parametric-family"
    assert not certificate.linear_in_parameters

    overdetermined = phx.equations.HolomorphicPolynomialConstraintPlan(
        0,
        (
            phx.equations.HolomorphicPointConstraint.dirichlet(0.0, 1.0),
            phx.equations.HolomorphicPointConstraint.dirichlet(1.0, 1.0),
            phx.equations.HolomorphicPointConstraint.dirichlet(
                -1.0,
                -0.5,
                component="imaginary",
            ),
        ),
    ).prepare()
    fixed = phx.equations.ConstrainedHolomorphicPolynomialPotential(overdetermined)
    assert overdetermined.evidence.rank == 2
    assert overdetermined.evidence.nullity == 0
    assert jnp.allclose(fixed.coefficients[0], jnp.asarray([1.0 - 0.5j]))
    assert (
        jnp.linalg.norm(fixed.constraint_residual())
        <= overdetermined.evidence.lift_tolerance
    )

    inconsistent = phx.equations.HolomorphicPolynomialConstraintPlan(
        1,
        (
            duplicate,
            phx.equations.HolomorphicPointConstraint.dirichlet(0.0, 2.0),
        ),
    )
    with pytest.raises(ValueError, match="inconsistent"):
        inconsistent.prepare()


def test_point_constraint_validation_and_inactive_basis_functional():
    with pytest.raises(ValueError, match="nonzero weight"):
        phx.equations.HolomorphicPointConstraint(
            0.0,
            0.0,
            value_weight=0.0,
            normal_weight=0.0,
        )
    with pytest.raises(ValueError, match="nonzero normal"):
        phx.equations.HolomorphicPointConstraint.normal_derivative(
            0.0,
            (0.0, 0.0),
            0.0,
        )
    inactive = phx.equations.HolomorphicPolynomialConstraintPlan(
        0,
        (
            phx.equations.HolomorphicPointConstraint.normal_derivative(
                0.0,
                (1.0, 0.0),
                0.0,
            ),
        ),
    )
    with pytest.raises(ValueError, match="identically zero"):
        inactive.prepare()

    with pytest.raises(TypeError, match="real Cartesian"):
        phx.equations.ConstrainedHolomorphicPolynomialPotential(
            phx.equations.HolomorphicPolynomialConstraintPlan(
                1,
                (phx.equations.HolomorphicPointConstraint.dirichlet(0.0, 0.0),),
            ).prepare(),
            initial_free_coordinates=np.ones((3,), dtype=np.complex128),
        )
