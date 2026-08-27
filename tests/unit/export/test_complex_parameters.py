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


def _complex_trainable_leaf_count(value):
    trainable, _ = partition_trainable(value)
    return sum(
        int(jnp.iscomplexobj(leaf))
        for leaf in jax.tree.leaves(trainable)
        if eqx.is_array(leaf)
    )


def test_complex_interchange_state_is_canonical_and_rejects_invalid_entries():
    first = phx.export.ComplexInterchangeState.from_entries(
        "trainable-parameters",
        "external-model",
        "architecture",
        {
            "layers/1/weight": jnp.asarray([1.0 + 2.0j]),
            "layers/0/weight": jnp.asarray([3.0 - 1.0j]),
        },
        roles={
            "layers/0/weight": "weight",
            "layers/1/weight": "weight",
        },
        metadata={"source": "reference"},
    )
    second = phx.export.ComplexInterchangeState.from_entries(
        "trainable-parameters",
        "external-model",
        "architecture",
        {
            "layers/0/weight": jnp.asarray([3.0 - 1.0j]),
            "layers/1/weight": jnp.asarray([1.0 + 2.0j]),
        },
        roles={
            "layers/1/weight": "weight",
            "layers/0/weight": "weight",
        },
        metadata={"source": "reference"},
    )
    assert tuple(entry.name for entry in first.entries) == (
        "layers/0/weight",
        "layers/1/weight",
    )
    assert first.state_id == second.state_id
    assert first.entry("layers/0/weight").role == "weight"

    with pytest.raises(ValueError, match="safe relative paths"):
        phx.export.ComplexInterchangeEntry(
            "../weight",
            jnp.asarray([1.0 + 0.0j]),
            role="weight",
            trainable=True,
        )
    with pytest.raises(TypeError, match="complex64 or complex128"):
        phx.export.ComplexInterchangeEntry(
            "weight",
            jnp.asarray([1.0]),
            role="weight",
            trainable=True,
        )
    with pytest.raises(ValueError, match="finite"):
        phx.export.ComplexInterchangeEntry(
            "weight",
            jnp.asarray([jnp.nan + 0.0j]),
            role="weight",
            trainable=True,
        )


def test_complex_linear_and_low_rank_exact_round_trip():
    dense = phx.nn.layers.ComplexLinear(in_size=2, out_size=3, key=jr.key(0))
    dense_state = phx.export.export_complex_parameters(dense)
    dense_target = phx.nn.layers.ComplexLinear(in_size=2, out_size=3, key=jr.key(1))
    dense_restored = phx.export.import_complex_parameters(dense_target, dense_state)
    point = jnp.asarray([0.2 + 0.1j, -0.3 + 0.4j])
    assert jnp.array_equal(dense_restored.weight_real, dense.weight_real)
    assert jnp.array_equal(dense_restored.weight_imag, dense.weight_imag)
    assert jnp.array_equal(dense_restored.bias_real, dense.bias_real)
    assert jnp.array_equal(dense_restored.bias_imag, dense.bias_imag)
    assert jnp.array_equal(dense_restored(point), dense(point))
    assert _complex_trainable_leaf_count(dense_restored) == 0

    low_rank = phx.nn.layers.LowRankComplexLinear(
        in_size=3,
        out_size=2,
        rank=2,
        key=jr.key(2),
    )
    low_rank_state = phx.export.export_complex_parameters(low_rank)
    low_rank_target = phx.nn.layers.LowRankComplexLinear(
        in_size=3,
        out_size=2,
        rank=2,
        key=jr.key(3),
    )
    low_rank_restored = phx.export.import_complex_parameters(
        low_rank_target,
        low_rank_state,
    )
    assert jnp.array_equal(low_rank_restored.input_factor, low_rank.input_factor)
    assert jnp.array_equal(low_rank_restored.output_factor, low_rank.output_factor)
    assert jnp.array_equal(
        low_rank_restored.materialize_weight(),
        low_rank.materialize_weight(),
    )
    assert _complex_trainable_leaf_count(low_rank_restored) == 0


def test_holomorphic_mlp_round_trip_preserves_values_jets_and_architecture():
    model = phx.nn.models.HolomorphicMLP(
        in_size=2,
        out_size=2,
        hidden_sizes=(4, 3),
        linear_ranks=(None, 2, 2),
        key=jr.key(4),
    )
    state = phx.export.export_complex_parameters(model)
    target = phx.nn.models.HolomorphicMLP(
        in_size=2,
        out_size=2,
        hidden_sizes=(4, 3),
        linear_ranks=(None, 2, 2),
        key=jr.key(5),
    )
    restored = phx.export.import_complex_parameters(target, state)
    coordinate = jnp.asarray([0.2 + 0.1j, -0.3 + 0.25j])
    indices = phx.equations.HolomorphicMultiIndexSet.total_degree(2, 2)
    expected_jet = model.multi_jet(coordinate, indices)
    actual_jet = restored.multi_jet(coordinate, indices)
    assert jnp.array_equal(restored(coordinate), model(coordinate))
    assert jnp.array_equal(actual_jet.value, expected_jet.value)
    assert jnp.array_equal(
        actual_jet.derivative((1, 1)),
        expected_jet.derivative((1, 1)),
    )
    assert restored.architecture_id == model.architecture_id
    assert (
        restored.holomorphic_certificate().certificate_id
        == model.holomorphic_certificate().certificate_id
    )
    assert _complex_trainable_leaf_count(restored) == 0

    incompatible = phx.nn.models.HolomorphicMLP(
        in_size=2,
        out_size=2,
        hidden_sizes=(4, 4),
        linear_ranks=(None, 2, 2),
        key=jr.key(6),
    )
    with pytest.raises(ValueError, match="architecture mismatch"):
        phx.export.import_complex_parameters(incompatible, state)


def test_polynomial_potential_and_multivariate_frame_coordinates_round_trip():
    potential = phx.equations.HolomorphicPolynomialPotential(
        2,
        3,
        initial_scale=0.3,
        key=jr.key(7),
    )
    state = phx.export.export_complex_parameters(potential)
    target = phx.equations.HolomorphicPolynomialPotential(2, 3, key=jr.key(8))
    restored = phx.export.import_complex_parameters(target, state)
    coordinate = 0.2 - 0.3j
    assert jnp.array_equal(restored.coefficients, potential.coefficients)
    assert jnp.array_equal(restored(coordinate), potential(coordinate))
    assert jnp.array_equal(
        restored.jet(coordinate, 3).derivative(3),
        potential.jet(coordinate, 3).derivative(3),
    )

    indices = phx.equations.HolomorphicMultiIndexSet.total_degree(2, 2)
    frame = phx.equations.HolomorphicPolynomialFrame(indices, 2)
    real_coordinates = jnp.linspace(-0.5, 0.6, frame.real_coefficient_count)
    complex_coefficients = phx.export.frame_coefficients_to_complex(
        frame,
        real_coordinates,
    )
    recovered = phx.export.complex_coefficients_to_frame(
        frame,
        complex_coefficients,
    )
    assert complex_coefficients.shape == (2, indices.count)
    assert jnp.array_equal(recovered, real_coordinates)


def _constrained_holomorphic(free):
    frame = phx.equations.HolomorphicPolynomialFrame.one_variable(3)
    operator = phx.equations.HolomorphicConstraintOperatorPlan(
        frame,
        (
            phx.equations.HolomorphicPointFunctional.value(-1.0),
            phx.equations.HolomorphicPointFunctional.value(1.0),
        ),
    ).prepare()
    coefficient_map = operator.affine_map(jnp.asarray([0.2, -0.1]))
    return phx.equations.ConstrainedHolomorphicPotential(
        coefficient_map,
        initial_free_coordinates=jnp.asarray(free),
    )


def test_constrained_import_recovers_free_coordinates_and_rejects_projection():
    source = _constrained_holomorphic(jnp.linspace(-0.2, 0.3, 6))
    destination = _constrained_holomorphic(jnp.zeros((6,)))
    state = phx.export.export_complex_parameters(source)
    restored = phx.export.import_complex_parameters(destination, state)
    assert jnp.allclose(restored.free_coordinates, source.free_coordinates, atol=2e-12)
    assert jnp.allclose(
        restored.coefficient_vector, source.coefficient_vector, atol=2e-12
    )
    assert (
        jnp.linalg.norm(restored.constraint_residual())
        <= restored.coefficient_map.evidence.tolerance
    )
    assert _complex_trainable_leaf_count(restored) == 0

    entry = state.entry("coefficients")
    invalid = phx.export.ComplexInterchangeState(
        state.semantics,
        state.provider_kind,
        state.architecture_id,
        (
            phx.export.ComplexInterchangeEntry(
                "coefficients",
                entry.value.at[0, 0].add(0.5),
                role=entry.role,
                component_dtype=entry.component_dtype,
                trainable=False,
            ),
        ),
        metadata=state.metadata,
    )
    with pytest.raises(ValueError, match="affine set"):
        phx.export.import_complex_parameters(destination, invalid)


def test_meromorphic_coefficients_and_trainable_poles_round_trip():
    poles = phx.equations.PoleSet(jnp.asarray([2.0 + 0.2j]), (2,))
    frame = phx.equations.MeromorphicLinearFrame(2, poles)
    operator = phx.equations.HolomorphicConstraintOperatorPlan(
        frame,
        (
            phx.equations.HolomorphicPointFunctional.value(-0.5),
            phx.equations.HolomorphicPointFunctional.value(0.5),
        ),
    ).prepare()
    coefficient_map = operator.affine_map(jnp.asarray([0.1, -0.2]))
    source = phx.equations.ConstrainedMeromorphicPotential(
        coefficient_map,
        initial_free_coordinates=jnp.linspace(-0.15, 0.25, coefficient_map.nullity),
    )
    destination = phx.equations.ConstrainedMeromorphicPotential(coefficient_map)
    state = phx.export.export_complex_parameters(source)
    restored = phx.export.import_complex_parameters(destination, state)
    assert jnp.allclose(
        restored.coefficient_vector, source.coefficient_vector, atol=2e-12
    )
    assert restored.frame.poles.pole_set_id == source.frame.poles.pole_set_id

    trainable_poles = phx.equations.TrainablePoleSet(
        jnp.asarray([1.5 + 0.2j, -2.0 + 0.3j]),
        (1, 2),
    )
    pole_state = phx.export.export_complex_parameters(trainable_poles)
    pole_target = phx.equations.TrainablePoleSet(
        jnp.asarray([3.0 + 0.0j, -3.0 + 0.0j]),
        (1, 2),
    )
    pole_restored = phx.export.import_complex_parameters(pole_target, pole_state)
    assert jnp.array_equal(pole_restored.locations, trainable_poles.locations)
    assert _complex_trainable_leaf_count(pole_restored) == 0


def test_import_precision_policy_rejects_narrowing_by_default():
    source = phx.nn.layers.ComplexLinear(in_size=2, out_size=2, key=jr.key(9))
    source = eqx.tree_at(
        lambda layer: (
            layer.weight_real,
            layer.weight_imag,
            layer.bias_real,
            layer.bias_imag,
        ),
        source,
        (
            source.weight_real.astype(jnp.float64),
            source.weight_imag.astype(jnp.float64),
            source.bias_real.astype(jnp.float64),
            source.bias_imag.astype(jnp.float64),
        ),
    )
    state = phx.export.export_complex_parameters(source)
    target = phx.nn.layers.ComplexLinear(in_size=2, out_size=2, key=jr.key(10))
    target = eqx.tree_at(
        lambda layer: (
            layer.weight_real,
            layer.weight_imag,
            layer.bias_real,
            layer.bias_imag,
        ),
        target,
        (
            target.weight_real.astype(jnp.float32),
            target.weight_imag.astype(jnp.float32),
            target.bias_real.astype(jnp.float32),
            target.bias_imag.astype(jnp.float32),
        ),
    )
    with pytest.raises(ValueError, match="lose precision"):
        phx.export.import_complex_parameters(target, state)
    restored = phx.export.import_complex_parameters(
        target,
        state,
        policy=phx.export.ComplexImportPolicy(allow_precision_loss=True),
    )
    assert restored.weight_real.dtype == jnp.float32
    assert jnp.allclose(restored.weight, source.weight.astype(jnp.complex64))
