import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.nn.operator import FunctionSamples, OperatorAxis, OperatorBatch
from phydrax.nn.operator.adapters import GroupAveragedOperator
from phydrax.nn.operator.architectures import LatticeEquivariantCNO
from phydrax.nn.operator.layers import (
    InvariantBasisTransferPlan,
    InvariantFilterBasis,
    LatticeEquivariantConvND,
    TensorNormActivation,
    TensorRMSNorm,
)
from phydrax.nn.operator.representations import (
    FiniteOrthogonalGroup,
    TensorFieldBlock,
    TensorFieldLayout,
    TensorType,
)


def _layout(kind, dimension=2, *, name=None, multiplicity=1):
    specifications = {
        "scalar": TensorType((), dimension=dimension),
        "pseudoscalar": TensorType((), parity=-1, dimension=dimension),
        "vector": TensorType(("contravariant",), dimension=dimension),
        "pseudovector": TensorType(("contravariant",), parity=-1, dimension=dimension),
        "rank_two": TensorType(("contravariant", "contravariant"), dimension=dimension),
    }
    return TensorFieldLayout(
        (
            TensorFieldBlock(
                kind if name is None else name,
                specifications[kind],
                multiplicity=multiplicity,
            ),
        )
    )


@pytest.mark.parametrize(
    ("input_kind", "output_kind"),
    (
        ("scalar", "scalar"),
        ("scalar", "vector"),
        ("vector", "scalar"),
        ("vector", "vector"),
        ("pseudovector", "vector"),
        ("rank_two", "rank_two"),
    ),
)
def test_invariant_lattice_convolution_intertwines_d4_tensor_actions(
    input_kind, output_kind
):
    group = FiniteOrthogonalGroup.d4()
    input_layout = _layout(input_kind)
    output_layout = _layout(output_kind)
    basis = InvariantFilterBasis(
        group,
        input_layout,
        output_layout,
        kernel_shape=3,
    )
    layer = LatticeEquivariantConvND(
        basis,
        use_bias=output_kind == "scalar",
        dtype=jnp.float64,
        key=jr.key(3),
    )
    values = jr.normal(jr.key(4), (5, 5, input_layout.channel_count))
    mask = jr.bernoulli(jr.key(5), 0.8, (5, 5))
    quadrature = 0.5 + jr.uniform(jr.key(6), (5, 5))

    reference = layer(values, source_mask=mask, target_mask=mask, quadrature=quadrature)
    for element in range(group.order):
        transformed_values = group.field_action(values, input_layout, element)
        transformed_mask = group.spatial_action(mask, element)
        transformed_quadrature = group.spatial_action(quadrature, element)
        transformed_output = layer(
            transformed_values,
            source_mask=transformed_mask,
            target_mask=transformed_mask,
            quadrature=transformed_quadrature,
        )
        expected = group.field_action(reference, output_layout, element)
        assert jnp.allclose(transformed_output, expected, atol=2e-9, rtol=2e-9)


def test_lattice_convolution_is_periodic_translation_equivariant_and_zero_without_support():
    group = FiniteOrthogonalGroup.c4()
    layout = _layout("scalar")
    layer = LatticeEquivariantConvND(
        InvariantFilterBasis(group, layout, layout, kernel_shape=3),
        dtype=jnp.float64,
        key=jr.key(8),
    )
    values = jr.normal(jr.key(9), (7, 7, 1))
    shifted = jnp.roll(values, shift=(2, -3), axis=(0, 1))
    assert jnp.allclose(
        layer(shifted),
        jnp.roll(layer(values), shift=(2, -3), axis=(0, 1)),
        atol=2e-10,
        rtol=2e-10,
    )
    unsupported = layer(values, source_mask=jnp.zeros((7, 7), dtype=bool))
    assert jnp.array_equal(unsupported, jnp.zeros_like(unsupported))


def test_equivariant_biases_are_only_available_to_ordinary_scalars():
    group = FiniteOrthogonalGroup.d4()
    scalar = _layout("scalar")
    for output_kind in ("pseudoscalar", "vector", "pseudovector"):
        output = _layout(output_kind)
        basis = InvariantFilterBasis(group, scalar, output, kernel_shape=1)
        with pytest.raises(ValueError, match="ordinary scalar"):
            LatticeEquivariantConvND(basis, use_bias=True)


def test_tensor_norm_activation_and_rms_normalization_are_d4_equivariant():
    group = FiniteOrthogonalGroup.d4()
    layout = TensorFieldLayout(
        (
            TensorFieldBlock("scalar", TensorType((), dimension=2), multiplicity=2),
            TensorFieldBlock("vector", TensorType(("contravariant",), dimension=2)),
            TensorFieldBlock("pseudo", TensorType((), parity=-1, dimension=2)),
        )
    )
    values = jr.normal(jr.key(10), (5, 5, layout.channel_count))
    normalize = TensorRMSNorm(layout, dtype=jnp.float64)
    activate = TensorNormActivation(layout, jax.nn.silu)
    reference = activate(normalize(values))
    for element in range(group.order):
        transformed = group.field_action(values, layout, element)
        expected = group.field_action(reference, layout, element)
        assert jnp.allclose(
            activate(normalize(transformed)), expected, atol=2e-10, rtol=2e-10
        )


def test_group_average_equivariantizes_a_directional_lattice_model():
    group = FiniteOrthogonalGroup.d4()
    layout = _layout("scalar")
    directional = lambda values: jnp.roll(values, 1, axis=0)
    averaged = GroupAveragedOperator(directional, group, layout, layout)
    values = jr.normal(jr.key(11), (5, 5, 1))
    reference = averaged(values)

    for element in range(group.order):
        transformed = averaged(group.field_action(values, layout, element))
        expected = group.field_action(reference, layout, element)
        assert jnp.allclose(transformed, expected, atol=2e-10, rtol=2e-10)


def test_scalar_invariant_basis_transfer_reports_exact_central_embedding_and_rejects_loss():
    source_group = FiniteOrthogonalGroup.c4()
    target_group = FiniteOrthogonalGroup.cube_rotations()
    source_layout = _layout("scalar", 2, name="field")
    target_layout = _layout("scalar", 3, name="field")
    source = InvariantFilterBasis(
        source_group, source_layout, source_layout, kernel_shape=3
    )
    target = InvariantFilterBasis(
        target_group, target_layout, target_layout, kernel_shape=3
    )
    exact_plan = InvariantBasisTransferPlan(
        source,
        target,
        residual_tolerance=1e-10,
    )
    central = jnp.zeros((3, 3, 1, 1)).at[1, 1, 0, 0].set(1.0)
    source_coefficients = jnp.einsum(
        "ri,i->r",
        source.basis.reshape(source.rank, -1),
        central.reshape(-1),
    )
    report = exact_plan.transfer(source_coefficients)
    assert report.relative_residual < 1e-10
    assert report.source_fingerprint == source.fingerprint
    assert report.target_fingerprint == target.fingerprint

    generic = jnp.arange(source.rank, dtype=float) + 1.0
    rejecting_plan = InvariantBasisTransferPlan(
        source,
        target,
        residual_tolerance=1e-12,
    )
    with pytest.raises(ValueError, match="residual exceeds"):
        rejecting_plan.transfer(generic)


def test_cross_dimensional_basis_transfer_rejects_tensor_component_reinterpretation():
    source_group = FiniteOrthogonalGroup.c4()
    target_group = FiniteOrthogonalGroup.cube_rotations()
    source_vector = _layout("vector", 2, name="field")
    target_vector = _layout("vector", 3, name="field")
    source = InvariantFilterBasis(
        source_group, source_vector, source_vector, kernel_shape=1
    )
    target = InvariantFilterBasis(
        target_group, target_vector, target_vector, kernel_shape=1
    )
    with pytest.raises(ValueError, match="scalar and pseudoscalar"):
        InvariantBasisTransferPlan(source, target)


def test_lattice_equivariant_cno_is_d4_equivariant_jittable_and_differentiable():
    group = FiniteOrthogonalGroup.d4()
    layout = _layout("scalar")
    model = LatticeEquivariantCNO(
        group,
        layout,
        layout,
        width=2,
        depth=1,
        kernel_size=3,
        key=jr.key(20),
    )
    nodes = jnp.arange(5, dtype=float) / 5.0
    values = jr.normal(jr.key(21), (5, 5, 1))
    evaluate = jax.jit(lambda field: model((field, nodes, nodes)))
    reference = evaluate(values)
    for element in range(group.order):
        transformed = evaluate(group.field_action(values, layout, element))
        expected = group.field_action(reference, layout, element)
        assert jnp.allclose(transformed, expected, atol=3e-5, rtol=3e-5)

    gradient = jax.grad(lambda field: jnp.sum(evaluate(field) ** 2))(values)
    assert jnp.all(jnp.isfinite(gradient))
    assert model.operator_contract.capabilities.requires_structured_tensors
    assert model.operator_contract.capabilities.symmetry_groups == ("D4",)

    with pytest.raises(ValueError, match="equal lattice sizes"):
        model((jnp.ones((5, 7, 1)), nodes, jnp.arange(7, dtype=float) / 7.0))

    source_axes = (
        OperatorAxis("x", nodes, periodic=True),
        OperatorAxis("y", nodes, periodic=True),
    )
    shifted_axes = (
        OperatorAxis("x", nodes + 0.1, periodic=True),
        OperatorAxis("y", nodes, periodic=True),
    )
    mismatched = OperatorBatch(
        inputs={"source": FunctionSamples(values=values, axes=source_axes)},
        queries={"query": FunctionSamples(values=None, axes=shifted_axes)},
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="coordinates must coincide"):
        jax.block_until_ready(model(mismatched))
