import jax
import jax.numpy as jnp
import pytest

from phydrax.nn.operator.representations import (
    FiniteOrthogonalGroup,
    TensorFieldBlock,
    TensorFieldLayout,
    TensorType,
)


def _mixed_layout(dimension=2):
    return TensorFieldLayout(
        (
            TensorFieldBlock(
                "scalar",
                TensorType((), dimension=dimension),
                multiplicity=2,
            ),
            TensorFieldBlock(
                "vector",
                TensorType(("contravariant",), dimension=dimension),
            ),
            TensorFieldBlock(
                "pseudoscalar",
                TensorType((), parity=-1, dimension=dimension),
            ),
        )
    )


def test_tensor_layout_round_trips_packed_fields_and_schema():
    layout = _mixed_layout()
    values = jnp.arange(5 * layout.channel_count, dtype=float).reshape(
        5, layout.channel_count
    )

    unpacked = layout.unpack(values)
    assert unpacked[0].shape == (5, 2)
    assert unpacked[1].shape == (5, 1, 2)
    assert unpacked[2].shape == (5, 1)
    assert jnp.array_equal(layout.pack(unpacked), values)
    assert jnp.array_equal(
        layout.pack(dict(zip(layout.block_names, unpacked, strict=True))),
        values,
    )
    assert TensorFieldLayout.from_dict(layout.to_dict()).to_dict() == layout.to_dict()


def test_tensor_actions_respect_variance_rank_and_reflection_parity():
    reflection = jnp.diag(jnp.array([1.0, -1.0]))
    vector = TensorType(("contravariant",), dimension=2)
    covector = TensorType(("covariant",), dimension=2)
    pseudovector = TensorType(("contravariant",), parity=-1, dimension=2)
    rank_two = TensorType(
        ("contravariant", "covariant"),
        dimension=2,
    )

    value = jnp.array([2.0, 3.0])
    assert jnp.array_equal(vector.transform(value, reflection), jnp.array([2.0, -3.0]))
    assert jnp.array_equal(covector.transform(value, reflection), jnp.array([2.0, -3.0]))
    assert jnp.array_equal(
        pseudovector.transform(value, reflection),
        jnp.array([-2.0, 3.0]),
    )
    matrix = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    expected = reflection @ matrix @ reflection.T
    assert jnp.array_equal(rank_two.transform(matrix, reflection), expected)


def test_tensor_normalization_contract_rejects_non_equivariant_affine_statistics():
    layout = _mixed_layout()
    valid_scale = (2.0, 3.0, 4.0, 4.0, 5.0)
    valid_offset = (1.0, -2.0, 0.0, 0.0, 0.0)
    layout.validate_affine_normalization(valid_scale, valid_offset)

    with pytest.raises(ValueError, match="one scale per tensor copy"):
        layout.validate_affine_normalization(
            (2.0, 3.0, 4.0, 5.0, 6.0),
            valid_offset,
        )
    with pytest.raises(ValueError, match="zero offsets"):
        layout.validate_affine_normalization(
            valid_scale,
            (1.0, -2.0, 0.1, 0.0, 0.0),
        )


@pytest.mark.parametrize(
    ("construct", "order", "proper"),
    (
        (FiniteOrthogonalGroup.c4, 4, True),
        (FiniteOrthogonalGroup.d4, 8, False),
        (FiniteOrthogonalGroup.cube_rotations, 24, True),
        (FiniteOrthogonalGroup.cube_orthogonal, 48, False),
    ),
)
def test_builtin_finite_groups_have_exact_group_metadata(construct, order, proper):
    group = construct()
    assert group.order == order
    assert group.is_proper is proper
    assert group.supports_lattice_action
    identity = group.identity_index
    for left in range(group.order):
        inverse = group.inverse(left)
        assert group.compose(left, identity) == left
        assert group.compose(identity, left) == left
        assert group.compose(left, inverse) == identity
        assert group.compose(inverse, left) == identity
    restored = FiniteOrthogonalGroup.from_dict(group.to_dict())
    assert restored.fingerprint == group.fingerprint


def test_finite_group_field_actions_compose_for_mixed_tensor_fields_under_jit():
    group = FiniteOrthogonalGroup.d4()
    layout = _mixed_layout()
    values = jnp.arange(5 * 5 * layout.channel_count, dtype=float).reshape(
        5, 5, layout.channel_count
    )

    for left, right in ((1, 2), (4, 5), (7, 3)):
        composed = group.compose(left, right)
        act_right = jax.jit(
            lambda field: group.field_action(field, layout, right, spatial_axes=(0, 1))
        )
        act_left = jax.jit(
            lambda field: group.field_action(field, layout, left, spatial_axes=(0, 1))
        )
        act_composed = jax.jit(
            lambda field: group.field_action(field, layout, composed, spatial_axes=(0, 1))
        )
        assert jnp.array_equal(act_left(act_right(values)), act_composed(values))
