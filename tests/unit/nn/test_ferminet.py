import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.nn.quantum._ferminet import (
    _polynomial_determinant,
    _scaled_determinant_components,
    _scaled_log_determinants,
    _stable_determinant_mixture,
    _stable_signed_bilinear_product,
    _stable_signed_product,
)


def _structure(positions, *, name="molecule"):
    scale = phx.atomistic.AtomisticScaleContract("bohr", "hartree")
    return phx.atomistic.AtomicStructure(
        jnp.ones((len(positions),), dtype=jnp.int32),
        jnp.asarray(positions, dtype=jnp.float64),
        jnp.ones((len(positions),), dtype=jnp.float64),
        scale,
        name=name,
    )


def _network(nuclei, *, spin_up=2, electrons=2, determinants=4):
    return phx.nn.quantum.FermiNet(
        nuclei,
        electrons,
        spin_up,
        hidden_features=12,
        pair_features=8,
        layer_count=2,
        determinant_count=determinants,
        compute_dtype="float64",
        key=jr.key(11),
    )


def test_same_spin_exchange_is_antisymmetric_and_batched_float64_is_canonical():
    network = _network(_structure([[0.0, 0.0, 0.0]], name="He"))
    electrons = jnp.asarray([[-0.8, 0.2, 0.1], [1.1, -0.3, 0.4]], dtype=jnp.float64)
    value = network(electrons)
    exchanged = network(electrons[::-1])
    assert value.valid
    assert exchanged.valid
    assert jnp.allclose(exchanged.log_abs, value.log_abs, rtol=1e-11, atol=1e-11)
    assert jnp.allclose(exchanged.phase, -value.phase)

    batch = network(jnp.stack((electrons, 1.2 * electrons)))
    assert isinstance(batch, phx.operators.LogAmplitude)
    assert batch.log_abs.shape == (2,)
    assert batch.log_abs.dtype == jnp.float64
    assert jnp.all(batch.valid)


def test_opposite_spin_exchange_is_not_forced_to_a_spatial_sign_rule():
    network = _network(_structure([[0.0, 0.0, 0.0]], name="He"), spin_up=1, electrons=2)
    electrons = jnp.asarray([[-0.6, 0.4, 0.2], [1.2, -0.2, 0.3]], dtype=jnp.float64)
    value = network(electrons)
    exchanged = network(electrons[::-1])
    forced_antisymmetry = jnp.allclose(exchanged.log_abs, value.log_abs) & jnp.allclose(
        exchanged.phase, -value.phase
    )
    assert not forced_antisymmetry


def test_joint_nuclear_electron_translation_and_rotation_leave_amplitude_invariant():
    nuclei = _structure([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]], name="H2")
    network = _network(nuclei, spin_up=1, electrons=2)
    electrons = jnp.asarray([[-0.4, 0.6, 0.1], [0.8, -0.2, -0.3]], dtype=jnp.float64)
    rotation = jnp.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=jnp.float64,
    )
    translation = jnp.asarray([1.0, -2.0, 0.5])
    transformed_nuclei = _structure(
        nuclei.positions @ rotation.T + translation, name="H2-transformed"
    )
    transformed_network = eqx.tree_at(
        lambda value: value.nuclei, network, transformed_nuclei
    )
    baseline = network(electrons)
    transformed = transformed_network(electrons @ rotation.T + translation)
    assert jnp.allclose(transformed.log_abs, baseline.log_abs, rtol=1e-11, atol=1e-11)
    assert jnp.allclose(transformed.phase, baseline.phase)


def test_determinant_combination_is_log_stable_and_envelopes_decay():
    nuclei = _structure([[0.0, 0.0, 0.0]], name="H")
    network = _network(nuclei, spin_up=1, electrons=1, determinants=8)
    decaying = eqx.tree_at(
        lambda value: value.orbital_weight,
        network,
        jnp.zeros_like(network.orbital_weight),
    )
    decaying = eqx.tree_at(
        lambda value: value.orbital_bias,
        decaying,
        jnp.ones_like(network.orbital_bias),
    )
    amplified = eqx.tree_at(
        lambda value: value.determinant_coefficients,
        decaying,
        jnp.asarray(
            [1e200, 1e190, 1e180, 1e170, 1e160, 1e150, 1e140, 1e130],
            dtype=jnp.float64,
        ),
    )
    near = amplified(jnp.asarray([[0.7, 0.1, -0.2]], dtype=jnp.float64))
    far = amplified(jnp.asarray([[25.0, 0.0, 0.0]], dtype=jnp.float64))
    assert near.valid
    assert jnp.isfinite(near.log_abs)
    assert jnp.all(network.envelope_decay > 0.0)
    assert far.log_abs < near.log_abs


def test_sparse_two_by_two_large_decay_determinant_remains_nonzero():
    raw_orbitals = jnp.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=jnp.float64)
    log_envelope = jnp.asarray([[[-1000.0, -1.0], [-1.0, -1000.0]]], dtype=jnp.float64)
    sign, log_abs = _scaled_log_determinants(raw_orbitals, log_envelope)
    assert sign[0] == 1.0
    assert jnp.isfinite(log_abs[0])
    assert jnp.allclose(log_abs[0], -2000.0)


def test_zero_dynamic_orbital_entry_retains_exact_logdet_gradient():
    log_envelope = jnp.zeros((1, 2, 2), dtype=jnp.float64)

    def logdet(entry):
        raw_orbitals = (
            jnp.asarray([[[1.0, 0.0], [1.0, 1.0]]], dtype=jnp.float64)
            .at[0, 0, 1]
            .set(entry)
        )
        return _scaled_log_determinants(raw_orbitals, log_envelope)[1][0]

    assert jnp.allclose(jax.grad(logdet)(jnp.asarray(0.0)), -1.0)
    assert jnp.allclose(jax.grad(jax.grad(logdet))(jnp.asarray(0.0)), -1.0)


def test_subnormal_orbital_value_and_relative_jvp_do_not_overflow():
    subnormal = jnp.asarray(1e-320, dtype=jnp.float64)
    raw_orbitals = jnp.asarray([[[subnormal, 0.0], [0.0, 1.0]]], dtype=jnp.float64)
    log_envelope = jnp.asarray([[[736.0, 0.0], [0.0, 0.0]]], dtype=jnp.float64)
    tangent = jnp.asarray([[[subnormal, 0.0], [0.0, 0.0]]], dtype=jnp.float64)

    def logdet(values):
        return _scaled_log_determinants(values, log_envelope)[1]

    value, directional = jax.jvp(logdet, (raw_orbitals,), (tangent,))
    assert jnp.all(jnp.isfinite(value))
    assert jnp.allclose(directional[0], 1.0)


def test_extreme_log_scale_tangents_remain_representable_and_inactive():
    value = jnp.asarray(1.0, dtype=jnp.float64)
    large_tangent = jnp.asarray(1e300, dtype=jnp.float64)
    negative_scale = jnp.asarray(-1000.0, dtype=jnp.float64)
    expected = jnp.exp(jnp.log(large_tangent) + negative_scale)
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    positive_scale = jnp.asarray(1000.0, dtype=jnp.float64)

    _, unary_tangent = jax.jvp(
        lambda scale: _stable_signed_product(value, scale),
        (negative_scale,),
        (large_tangent,),
    )
    _, bilinear_tangent = jax.jvp(
        lambda scale: _stable_signed_bilinear_product(value, value, scale),
        (negative_scale,),
        (large_tangent,),
    )
    _, inactive_unary = jax.jvp(
        lambda scale: _stable_signed_product(value, scale),
        (jnp.asarray(1000.0),),
        (jnp.asarray(0.0),),
    )
    _, inactive_bilinear = jax.jvp(
        lambda scale: _stable_signed_bilinear_product(value, value, scale),
        (jnp.asarray(1000.0),),
        (jnp.asarray(0.0),),
    )
    _, zero_unary_scale_tangent = jax.jvp(
        lambda scale: _stable_signed_product(zero, scale),
        (positive_scale,),
        (value,),
    )
    _, zero_bilinear_scale_tangent = jax.jvp(
        lambda scale: _stable_signed_bilinear_product(value, zero, scale),
        (positive_scale,),
        (value,),
    )
    _, zero_bilinear_value_tangent = jax.jvp(
        lambda left: _stable_signed_bilinear_product(left, zero, positive_scale),
        (value,),
        (value,),
    )

    assert jnp.allclose(unary_tangent, expected, rtol=1e-12, atol=0.0)
    assert jnp.allclose(bilinear_tangent, expected, rtol=1e-12, atol=0.0)
    assert inactive_unary == 0.0
    assert inactive_bilinear == 0.0
    assert zero_unary_scale_tangent == 0.0
    assert zero_bilinear_scale_tangent == 0.0
    assert zero_bilinear_value_tangent == 0.0


def test_zero_multiplier_mixed_derivative_stays_in_signed_log_domain():
    log_scale = jnp.asarray(-1000.0, dtype=jnp.float64)
    large_tangent = jnp.asarray(1e300, dtype=jnp.float64)
    zero = jnp.asarray(0.0, dtype=jnp.float64)

    def left_direction(right):
        return jax.jvp(
            lambda left: _stable_signed_bilinear_product(left, right, log_scale),
            (zero,),
            (large_tangent,),
        )[1]

    mixed_derivative = jax.grad(left_direction)(zero)
    expected = jnp.exp(jnp.log(large_tangent) + log_scale)

    assert jnp.allclose(mixed_derivative, expected, rtol=1e-12, atol=0.0)


def test_singular_determinant_term_retains_mixture_derivative():
    log_envelope = jnp.zeros((2, 2, 2), dtype=jnp.float64)

    def mixture_log_abs(entry):
        singular = (
            jnp.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=jnp.float64).at[0, 0].set(entry)
        )
        constant = jnp.eye(2, dtype=jnp.float64)
        raw_orbitals = jnp.stack((singular, constant))
        determinant, log_scale = _scaled_determinant_components(
            raw_orbitals, log_envelope
        )
        return _stable_determinant_mixture(
            determinant,
            log_scale,
            jnp.ones((2,), dtype=jnp.float64),
        )[0]

    value = mixture_log_abs(jnp.asarray(0.0))
    gradient = jax.grad(mixture_log_abs)(jnp.asarray(0.0))
    assert jnp.allclose(value, 0.0)
    assert jnp.allclose(gradient, 1.0)


def test_zero_large_scale_term_does_not_hide_nonzero_small_scale_term():
    log_scale = jnp.asarray([0.0, -1000.0], dtype=jnp.float64)
    coefficients = jnp.ones((2,), dtype=jnp.float64)

    def mixture(nonzero_determinant):
        determinants = jnp.stack((jnp.asarray(0.0), nonzero_determinant))
        return _stable_determinant_mixture(determinants, log_scale, coefficients)

    log_abs, phase, valid = mixture(jnp.asarray(1.0))
    gradient = jax.grad(lambda value: mixture(value)[0])(jnp.asarray(1.0))
    assert valid
    assert phase == 1.0 + 0.0j
    assert jnp.isfinite(log_abs)
    assert jnp.allclose(log_abs, -1000.0)
    assert jnp.allclose(gradient, 1.0)


def test_zero_coefficient_large_scale_term_does_not_set_mixture_shift():
    determinants = jnp.ones((2,), dtype=jnp.float64)
    log_scale = jnp.asarray([0.0, -1000.0], dtype=jnp.float64)

    def mixture(coefficients):
        return _stable_determinant_mixture(determinants, log_scale, coefficients)

    coefficients = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    log_abs, phase, valid = mixture(coefficients)
    gradient = jax.grad(lambda values: mixture(values)[0])(coefficients)
    assert valid
    assert phase == 1.0 + 0.0j
    assert jnp.allclose(log_abs, -1000.0)
    assert jnp.isposinf(gradient[0])
    assert jnp.allclose(gradient[1], 1.0)


def test_inactive_huge_coefficient_does_not_hide_tiny_active_product():
    coefficients = jnp.asarray([1e300, 1e-300], dtype=jnp.float64)
    log_scale = jnp.zeros((2,), dtype=jnp.float64)

    def mixture(determinants):
        return _stable_determinant_mixture(determinants, log_scale, coefficients)

    determinants = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    log_abs, phase, valid = mixture(determinants)
    gradient = jax.grad(lambda values: mixture(values)[0])(determinants)
    assert valid
    assert phase == 1.0 + 0.0j
    assert jnp.allclose(log_abs, jnp.log(jnp.asarray(1e-300)))
    assert jnp.isposinf(gradient[0])
    assert jnp.allclose(gradient[1], 1.0)


def test_polynomial_determinant_resource_admission_and_limit_rejection():
    resource_plan = phx.operators.ElectronicVMCResourcePlan(
        4,
        determinant_count=1,
        maximum_pair_elements=48,
        maximum_determinant_work=64,
    )
    size = resource_plan.electron_count
    assert resource_plan.pair_stream_elements == 48
    assert resource_plan.determinant_work == 64
    assert bool(resource_plan.valid)
    assert resource_plan.claim == "finite-resource-admission-not-unrestricted-scaling"
    perturbation_values = [
        [0.0, 1.0, -0.5, 0.25],
        [-0.75, 0.0, 0.5, -0.25],
        [0.5, -0.5, 0.0, 0.75],
        [-0.25, 0.25, -0.75, 0.0],
    ]
    for dtype, tolerance in (
        (jnp.float32, 2e-6),
        (jnp.float64, 1e-12),
    ):
        identity = jnp.eye(size, dtype=dtype)
        perturbation = jnp.asarray(perturbation_values, dtype=dtype)
        near_identity = identity + 1e-4 * perturbation
        assert jnp.allclose(_polynomial_determinant(identity), 1.0)
        assert jnp.allclose(
            _polynomial_determinant(near_identity),
            jnp.linalg.det(near_identity),
            rtol=tolerance,
            atol=tolerance,
        )

    with pytest.raises(ValueError, match="exceeds caller limits"):
        phx.operators.ElectronicVMCResourcePlan(
            size + 1,
            determinant_count=1,
            maximum_pair_elements=resource_plan.admitted_pair_elements,
            maximum_determinant_work=resource_plan.admitted_determinant_work,
        )

    nuclei = _structure([[0.0, 0.0, 0.0]], name="resource-admission")
    with pytest.raises(ValueError, match="counts must match resource_plan"):
        phx.nn.quantum.FermiNet(
            nuclei,
            size + 1,
            3,
            determinant_count=1,
            resource_plan=resource_plan,
            key=jr.key(90),
        )
    with pytest.raises(ValueError, match="must match resource_plan"):
        phx.operators.ElectronicCoulombHamiltonian(
            nuclei,
            size + 1,
            resource_plan=resource_plan,
        )


def test_large_decay_at_distant_configuration_remains_a_nonzero_log_amplitude():
    nuclei = _structure([[0.0, 0.0, 0.0]], name="H-large-decay")
    network = _network(nuclei, spin_up=1, electrons=1, determinants=1)
    network = eqx.tree_at(
        lambda value: (value.orbital_weight, value.orbital_bias),
        network,
        (
            jnp.zeros_like(network.orbital_weight),
            jnp.ones_like(network.orbital_bias),
        ),
    )
    network = eqx.tree_at(
        lambda value: value.raw_envelope_decay,
        network,
        jnp.full_like(network.raw_envelope_decay, 1000.0),
    )
    value = network(jnp.asarray([[1.0, 0.0, 0.0]], dtype=jnp.float64))
    assert value.valid
    assert value.nonzero
    assert jnp.isfinite(value.log_abs)
    assert value.log_abs < -900.0


def test_envelope_decay_floor_survives_softplus_underflow_and_normalizes_tail():
    nuclei = _structure([[0.0, 0.0, 0.0]], name="H-decay-floor")
    network = _network(nuclei, spin_up=1, electrons=1, determinants=1)
    network = eqx.tree_at(
        lambda value: (value.orbital_weight, value.orbital_bias),
        network,
        (
            jnp.zeros_like(network.orbital_weight),
            jnp.ones_like(network.orbital_bias),
        ),
    )
    network = eqx.tree_at(
        lambda value: value.raw_envelope_decay,
        network,
        jnp.full_like(network.raw_envelope_decay, -1e6),
    )
    assert jnp.all(network.envelope_decay >= network.configuration.minimum_envelope_decay)
    near = network(jnp.asarray([[1.0, 0.0, 0.0]], dtype=jnp.float64))
    far = network(jnp.asarray([[1e7, 0.0, 0.0]], dtype=jnp.float64))
    assert near.nonzero
    assert far.nonzero
    assert far.log_abs < near.log_abs


def test_zero_determinant_coefficient_has_finite_reactivation_gradient():
    nuclei = _structure([[0.0, 0.0, 0.0]], name="H-zero-coefficient")
    network = _network(nuclei, spin_up=1, electrons=1, determinants=2)
    network = eqx.tree_at(
        lambda value: (value.orbital_weight, value.orbital_bias),
        network,
        (
            jnp.zeros_like(network.orbital_weight),
            jnp.asarray([[1.0], [2.0]], dtype=jnp.float64),
        ),
    )
    coefficients = jnp.asarray([1.0, 0.0], dtype=jnp.float64)
    electrons = jnp.asarray([[0.8, 0.0, 0.0]], dtype=jnp.float64)

    def log_amplitude(values):
        model = eqx.tree_at(lambda value: value.determinant_coefficients, network, values)
        return model(electrons).log_abs

    gradient = jax.grad(log_amplitude)(coefficients)
    assert jnp.all(jnp.isfinite(gradient))
    assert gradient[1] != 0.0


def test_parameter_gradients_coordinate_gradients_and_laplacian_are_finite():
    network = _network(_structure([[0.0, 0.0, 0.0]], name="He"), spin_up=1, electrons=2)
    electrons = jnp.asarray([[-0.7, 0.2, 0.4], [0.9, -0.3, 0.1]], dtype=jnp.float64)
    coordinate_gradient = jax.grad(lambda value: network(value).log_abs)(electrons)
    coordinate_hessian = jax.hessian(lambda value: network(value).log_abs)(electrons)
    parameter_gradient = eqx.filter_grad(lambda model: model(electrons).log_abs)(network)
    parameter_leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(parameter_gradient)
        if eqx.is_inexact_array(leaf)
    ]
    assert jnp.all(jnp.isfinite(coordinate_gradient))
    assert jnp.all(jnp.isfinite(coordinate_hessian))
    assert parameter_leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in parameter_leaves)
