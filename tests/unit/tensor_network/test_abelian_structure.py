import jax.numpy as jnp

import phydrax as phx


tn = phx.tensor_network


def test_abelian_group_leg_layout_and_product_state_dense_order():
    group = tn.AbelianGroup((None, 2))
    assert group.add((2, 1), (-1, 1)) == (1, 0)
    assert group.negate((2, 1)) == (-2, 1)

    u1 = tn.AbelianGroup((None,))
    physical = tn.AbelianLeg(u1, ((0,), (1,)), (1, 1), orientation=1)
    state = tn.abelian_product_mps(
        (
            jnp.asarray([1.0, 0.0], dtype=jnp.complex128),
            jnp.asarray([0.0, 1.0], dtype=jnp.complex128),
        ),
        (physical, physical),
        (0, 1),
    )
    assert state.total_charge == (1,)
    assert jnp.allclose(state.to_dense(), jnp.asarray([0.0, 1.0, 0.0, 0.0]))
    assert jnp.allclose(tn.abelian_mps_inner(state, state), 1.0)


def test_one_site_abelian_mpo_dense_conversion_preserves_sector_order():
    group = tn.AbelianGroup((2,))
    physical = tn.AbelianLeg(group, ((0,), (1,)), (1, 1), orientation=1)
    boundary_left = tn.AbelianLeg(group, ((0,),), (1,), orientation=1)
    boundary_right = tn.AbelianLeg(group, ((0,),), (1,), orientation=-1)
    layout = tn.AbelianTensorLayout(
        (boundary_left, physical.dual(), physical, boundary_right)
    )
    tensor = tn.AbelianTensor.from_dense(
        layout,
        jnp.eye(2, dtype=jnp.complex128)[None, :, :, None],
    )
    operator = tn.AbelianMatrixProductOperator((tensor,))
    assert jnp.allclose(operator.to_dense(), jnp.eye(2))
