#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax._trainable import partition_trainable


def _set_polynomial(potential, real, imaginary=None):
    real_ = jnp.asarray(real, dtype=float)
    imaginary_ = jnp.zeros_like(real_) if imaginary is None else jnp.asarray(imaginary)
    return eqx.tree_at(
        lambda value: (value.coefficient_real, value.coefficient_imag),
        potential,
        (real_, imaginary_),
    )


def test_complex_linear_uses_real_leaves_and_matches_dense_complex_oracle():
    layer = phx.nn.layers.ComplexLinear(in_size=2, out_size=2, key=jr.key(0))
    layer = eqx.tree_at(
        lambda value: (
            value.weight_real,
            value.weight_imag,
            value.bias_real,
            value.bias_imag,
        ),
        layer,
        (
            jnp.asarray([[1.0, 2.0], [-0.5, 0.25]]),
            jnp.asarray([[0.5, -1.0], [2.0, 0.0]]),
            jnp.asarray([0.2, -0.3]),
            jnp.asarray([-0.1, 0.4]),
        ),
    )
    point = jnp.asarray([0.4 + 0.2j, -0.7 + 0.5j])
    expected = layer.weight @ point + layer.bias
    assert jnp.allclose(layer(point), expected)
    trainable, _ = partition_trainable(layer)
    assert all(
        not jnp.iscomplexobj(leaf) for leaf in jax.tree_util.tree_leaves(trainable)
    )


def test_holomorphic_polynomial_horner_jets_match_closed_form():
    potential = phx.equations.HolomorphicPolynomialPotential(1, 3)
    potential = _set_polynomial(
        potential,
        [[1.0, -2.0, 0.5, 1.25]],
        [[0.25, 0.0, -0.75, 0.5]],
    )
    z = 0.3 - 0.4j
    coefficients = potential.coefficients[0]
    expected = sum(coefficients[k] * z**k for k in range(4))
    first = sum(k * coefficients[k] * z ** (k - 1) for k in range(1, 4))
    second = sum(k * (k - 1) * coefficients[k] * z ** (k - 2) for k in range(2, 4))
    jet = potential.jet(jnp.asarray(z), 2)
    assert jnp.allclose(jet.value[0], expected)
    assert jnp.allclose(jet.derivative(1)[0], first)
    assert jnp.allclose(jet.derivative(2)[0], second)
    assert potential.holomorphic_certificate().linear_in_parameters
    assert potential.holomorphic_certificate().parameter_coverage == "finite-subspace"


def test_holomorphic_mlp_satisfies_cauchy_riemann_with_real_parameters():
    model = phx.nn.models.HolomorphicMLP(
        in_size=1,
        out_size=2,
        hidden_sizes=(4, 4),
        key=jr.key(1),
    )
    point = jnp.asarray([0.2, -0.3])

    def complex_value(real_coordinates):
        return model(real_coordinates[0] + 1j * real_coordinates[1])

    derivative = jax.jacfwd(complex_value)(point)
    assert jnp.allclose(derivative[:, 0] + 1j * derivative[:, 1], 0.0, atol=2e-11)
    trainable, _ = partition_trainable(model)
    assert all(
        not jnp.iscomplexobj(leaf) for leaf in jax.tree_util.tree_leaves(trainable)
    )
    assert not model.holomorphic_certificate().linear_in_parameters
    assert (
        model.holomorphic_certificate().parameter_coverage == "finite-parametric-family"
    )

    domain = phx.domain.HyperRectangle((-1.0, -1.0), (1.0, 1.0))
    harmonic = phx.equations.HarmonicPotential2D(model)
    field = domain.Model("x")(harmonic)
    physical = phx.equations.trial_space_certificate(field)
    assert physical.coverage == "finite-parametric-family"
    assert not physical.linear_in_coefficients

    boundary = domain.component({"x": phx.domain.Boundary()})
    condition = phx.conditions.Dirichlet("u", boundary, target=0.0)
    source = phx.integration.fixed(
        phx.integration.materialize(
            phx.integration.mean_over(boundary),
            phx.domain.PointSampling(8),
            key=jr.key(2),
        )
    )
    solver = phx.solver.FunctionalSolver(
        functions={"u": field},
        terms=(phx.terms.ResidualPenalty(condition, source),),
    )
    with pytest.raises(TypeError, match="not linear"):
        phx.solver.solve_linear_trial_space(solver)


def test_harmonic_and_biharmonic_potential_representations_are_exact():
    harmonic_potential = _set_polynomial(
        phx.equations.HolomorphicPolynomialPotential(1, 2),
        [[0.0, 0.0, 1.0]],
    )
    harmonic = phx.equations.HarmonicPotential2D(harmonic_potential)
    point = jnp.asarray([0.3, -0.2])
    assert jnp.allclose(harmonic(point), point[0] ** 2 - point[1] ** 2)
    assert jnp.allclose(jnp.trace(jax.hessian(harmonic)(point)), 0.0, atol=1e-11)
    harmonic_certificate = phx.equations.trial_space_certificate(
        phx.domain.HyperRectangle((-1.0, -1.0), (1.0, 1.0)).Model("x")(harmonic)
    )
    assert harmonic_certificate.coverage == "finite-subspace"
    assert harmonic_certificate.linear_in_coefficients

    biharmonic_potential = _set_polynomial(
        phx.equations.HolomorphicPolynomialPotential(2, 2),
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    biharmonic = phx.equations.BiharmonicPotential2D(biharmonic_potential)

    def laplacian(value):
        return jnp.trace(jax.hessian(biharmonic)(value))

    assert jnp.allclose(jnp.trace(jax.hessian(laplacian)(point)), 0.0, atol=2e-10)


def test_plane_elasticity_potential_satisfies_equilibrium_for_both_hypotheses():
    potential = _set_polynomial(
        phx.equations.HolomorphicPolynomialPotential(2, 3),
        [
            [0.1, 0.5, -0.2, 0.05],
            [-0.3, 0.25, 0.1, -0.04],
        ],
        [
            [0.0, -0.1, 0.08, 0.02],
            [0.2, 0.05, -0.06, 0.01],
        ],
    )
    point = jnp.asarray([0.2, -0.15])
    for hypothesis in ("plane_strain", "plane_stress"):
        material = phx.equations.PlaneIsotropicMaterial(
            2.0,
            1.5,
            hypothesis=hypothesis,
        )
        state = phx.equations.PlaneElasticityPotential2D(potential, material)
        jacobian = jax.jacfwd(state)(point)
        equilibrium = jnp.asarray(
            [
                jacobian[0, 0] + jacobian[2, 1],
                jacobian[2, 0] + jacobian[1, 1],
            ]
        )
        assert state(point).shape == (5,)
        assert jnp.allclose(equilibrium, 0.0, atol=2e-10)

        stress_only = phx.equations.PlaneElasticityPotential2D(
            potential,
            material,
            output="stress",
        )
        assert stress_only(point).shape == (3,)
