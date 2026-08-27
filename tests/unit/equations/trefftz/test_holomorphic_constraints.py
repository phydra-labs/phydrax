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


def test_constraint_operator_reuses_factorization_across_affine_targets():
    frame = phx.equations.HolomorphicPolynomialFrame.one_variable(3)
    functionals = (
        phx.equations.HolomorphicPointFunctional.value(-1.0),
        phx.equations.HolomorphicPointFunctional.value(1.0),
    )
    operator = phx.equations.HolomorphicConstraintOperatorPlan(
        frame,
        functionals,
    ).prepare()
    homogeneous = operator.affine_map(jnp.zeros((2,)))
    affine = operator.affine_map(jnp.asarray([0.5, -0.25]))

    assert operator.evidence.rank == 2
    assert operator.evidence.nullity == 6
    assert homogeneous.operator.prepared_id == affine.operator.prepared_id
    assert homogeneous.map_id != affine.map_id
    targets = jnp.asarray([[0.0, 0.0], [0.5, -0.25], [-0.2, 0.7]])
    batched = operator.minimum_norm_coefficients(targets)
    independent = jnp.stack(
        tuple(operator.minimum_norm_coefficients(target) for target in targets)
    )
    assert jnp.allclose(batched, independent)
    assert jnp.allclose(operator.target_residual(targets), 0.0, atol=1e-12)

    free = jnp.linspace(-0.3, 0.4, homogeneous.nullity)
    potential = phx.equations.ConstrainedHolomorphicPotential(
        homogeneous,
        initial_free_coordinates=free,
    )
    assert (
        jnp.linalg.norm(potential.constraint_residual()) <= homogeneous.evidence.tolerance
    )
    assert jnp.allclose(jnp.real(potential(-1.0)[0]), 0.0, atol=1e-12)
    assert jnp.allclose(jnp.real(potential(1.0)[0]), 0.0, atol=1e-12)
    trainable, _ = partition_trainable(potential)
    leaves = jax.tree.leaves(trainable)
    assert len(leaves) == 1
    assert leaves[0].shape == (operator.evidence.nullity,)
    certificate = potential.holomorphic_certificate()
    assert certificate.parameter_coverage == "finite-subspace"
    assert certificate.linear_in_parameters


def test_vector_frame_supports_coupled_outputs_and_several_variables():
    indices = phx.equations.HolomorphicMultiIndexSet.total_degree(2, 2)
    normalization = phx.equations.ComplexAffineNormalization(
        jnp.asarray([0.1 + 0.2j, -0.3 + 0.1j]),
        jnp.asarray([[1.0 + 0.2j, 0.25], [-0.1j, 0.8 - 0.3j]]),
    )
    frame = phx.equations.HolomorphicPolynomialFrame(
        indices,
        2,
        normalization=normalization,
    )
    coordinate = jnp.asarray([0.2 - 0.1j, -0.4 + 0.3j])
    coupled = phx.equations.HolomorphicPointFunctional(
        coordinate,
        (
            phx.equations.HolomorphicJetFunctionalTerm(0, (0, 0), 1.0),
            phx.equations.HolomorphicJetFunctionalTerm(1, (1, 0), -0.5j),
        ),
    )
    normal = phx.equations.HolomorphicPointFunctional.normal_derivative(
        coordinate,
        (0.6, -0.2, 0.4, 0.7),
        output_index=1,
        component="imaginary",
    )
    operator = phx.equations.HolomorphicConstraintOperatorPlan(
        frame,
        (coupled, normal),
    ).prepare()
    coefficient_map = operator.affine_map(jnp.asarray([0.2, -0.1]))
    potential = phx.equations.ConstrainedHolomorphicPotential(coefficient_map)

    assert potential(coordinate).shape == (2,)
    assert (
        jnp.linalg.norm(potential.constraint_residual())
        <= coefficient_map.evidence.tolerance
    )
    jet = potential.multi_jet(coordinate, indices)
    assert jet.value.shape == (2,)
    assert jet.derivative((1, 1)).shape == (2,)
    assert frame.basis_derivative(coordinate, (1, 1)).shape == (
        2,
        frame.real_coefficient_count,
    )


def test_nonlinear_cardinal_projection_enforces_targets_after_parameter_change():
    frame = phx.equations.HolomorphicPolynomialFrame.one_variable(1)
    functionals = (
        phx.equations.HolomorphicPointFunctional.value(-1.0),
        phx.equations.HolomorphicPointFunctional.value(1.0),
    )
    operator = phx.equations.HolomorphicConstraintOperatorPlan(
        frame,
        functionals,
    ).prepare()
    projector = phx.equations.HolomorphicConstraintProjector(operator)
    provider = phx.nn.models.HolomorphicMLP(
        in_size=1,
        out_size=1,
        hidden_sizes=(4,),
        key=jr.key(0),
    )
    projected = projector.project(provider, jnp.asarray([0.5, -0.25]))
    assert jnp.allclose(jnp.real(projected(-1.0)[0]), 0.5, atol=2e-12)
    assert jnp.allclose(jnp.real(projected(1.0)[0]), -0.25, atol=2e-12)

    changed_provider = eqx.tree_at(
        lambda value: value.layers[0].weight_real,
        provider,
        provider.layers[0].weight_real + 0.2,
    )
    changed = projector.project(changed_provider, jnp.asarray([0.5, -0.25]))
    state = changed.prepare_projection()
    points = jnp.asarray([-0.4 + 0.1j, 0.3 - 0.2j])
    batched = jax.vmap(lambda point: changed.evaluate_with_state(point, state))(points)
    direct = jax.vmap(changed)(points)
    assert jnp.allclose(batched, direct)
    assert jnp.allclose(jnp.real(changed(-1.0)[0]), 0.5, atol=2e-12)
    assert jnp.allclose(jnp.real(changed(1.0)[0]), -0.25, atol=2e-12)


def test_biharmonic_and_plane_elasticity_functionals_match_physical_wrappers():
    frame = phx.equations.HolomorphicPolynomialFrame.one_variable(3, 2)
    coefficient_count = frame.real_coefficient_count
    coefficients = jnp.linspace(-0.3, 0.5, coefficient_count)

    class _FramePotential(eqx.Module):
        def __call__(self, coordinate):
            return frame.evaluate(coordinate, coefficients)

        def jet(self, coordinate, order):
            value = self(coordinate)
            derivatives = tuple(
                frame.basis_derivative(coordinate, (current,)) @ coefficients
                for current in range(1, order + 1)
            )
            return phx.equations.HolomorphicJet(value, derivatives)

        def holomorphic_certificate(self):
            certificate = frame.linear_frame_certificate()
            return phx.equations.HolomorphicMapCertificate(
                complex_input_size=1,
                complex_output_size=2,
                construction="test-linear-frame",
                normalization_id=certificate.normalization_id,
                maximum_derivative_order=certificate.maximum_derivative_order,
                operations=("complex-polynomial",),
                parameter_coverage="finite-parametric-family",
                linear_in_parameters=False,
                construction_dependencies=(certificate.frame_id,),
            )

    provider = _FramePotential()
    coordinate = 0.25 - 0.15j
    point = jnp.asarray([jnp.real(coordinate), jnp.imag(coordinate)])
    normal = jnp.asarray([0.6, 0.8])

    biharmonic = phx.equations.BiharmonicPotential2D(provider)
    value_functional = phx.equations.biharmonic_value_functional(coordinate)
    assert jnp.allclose(
        value_functional.assemble_row(frame) @ coefficients, biharmonic(point)
    )
    normal_functional = phx.equations.biharmonic_normal_derivative_functional(
        coordinate,
        normal,
    )
    expected_normal = jax.jacfwd(biharmonic)(point) @ normal
    assert jnp.allclose(
        normal_functional.assemble_row(frame) @ coefficients, expected_normal
    )

    material = phx.equations.PlaneIsotropicMaterial(2.0, 1.5)
    elasticity = phx.equations.PlaneElasticityPotential2D(provider, material)
    state = elasticity(point)
    stress_xx = phx.equations.plane_elasticity_stress_functional(coordinate, "xx")
    displacement_y = phx.equations.plane_elasticity_displacement_functional(
        coordinate,
        material,
        "y",
    )
    traction_x = phx.equations.plane_elasticity_traction_functional(
        coordinate,
        normal,
        "x",
    )
    assert jnp.allclose(stress_xx.assemble_row(frame) @ coefficients, state[0])
    assert jnp.allclose(displacement_y.assemble_row(frame) @ coefficients, state[4])
    assert jnp.allclose(
        traction_x.assemble_row(frame) @ coefficients,
        normal[0] * state[0] + normal[1] * state[2],
    )
    assert material.material_id in displacement_y.construction_dependencies


def test_constraint_rank_compatibility_and_validation_fail_closed():
    frame = phx.equations.HolomorphicPolynomialFrame.one_variable(1)
    duplicate = phx.equations.HolomorphicPointFunctional.value(0.0)
    operator = phx.equations.HolomorphicConstraintOperatorPlan(
        frame,
        (duplicate, duplicate),
    ).prepare()
    assert operator.evidence.rank == 1
    assert operator.evidence.nullity == 3
    operator.affine_map(jnp.asarray([1.0, 1.0]))
    with pytest.raises(ValueError, match="inconsistent"):
        operator.affine_map(jnp.asarray([1.0, 2.0]))
    with pytest.raises(ValueError, match="full row rank"):
        phx.equations.HolomorphicConstraintProjector(operator)

    constant_frame = phx.equations.HolomorphicPolynomialFrame.one_variable(0)
    inactive = phx.equations.HolomorphicPointFunctional.normal_derivative(
        0.0,
        (1.0, 0.0),
    )
    with pytest.raises(ValueError, match="derivative order"):
        phx.equations.HolomorphicConstraintOperatorPlan(
            constant_frame,
            (inactive,),
        ).prepare()
