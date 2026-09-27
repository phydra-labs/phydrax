import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.operators.quantum import AbelianGroup


tn = phx.tensor_network


def test_abelian_structure_scenario_1() -> None:
    group = AbelianGroup((None, 2))
    assert group.add((2, 1), (-1, 1)) == (1, 0)
    assert group.negate((2, 1)) == (-2, 1)

    u1 = AbelianGroup((None,))
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
    group = AbelianGroup((2,))
    left_leg = tn.AbelianLeg(group, ((0,), (1,)), (1, 1), orientation=1)
    right_leg = left_leg.dual()
    left = tn.AbelianTensorLayout((left_leg,))
    right = tn.AbelianTensorLayout((right_leg,))
    with pytest.raises(ValueError, match="outside"):
        tn.AbelianContractionPlan(left, right, (2,), (0,))
    group = AbelianGroup((2,))
    physical = tn.AbelianLeg(group, ((0,),), (1,), orientation=1)
    operator = tn.AbelianKrausOperator(
        physical,
        physical,
        (0,),
        ((0, 0),),
        (jnp.asarray([[jnp.sqrt(0.5)]], dtype=jnp.complex64),),
    )
    channel = tn.ChargeCovariantKrausMap(
        (operator,),
        completeness_tolerance=1.0,
        require_trace_preserving=False,
    )
    state = tn.AbelianLPDO(
        physical,
        (1,),
        (jnp.ones((1, 1), dtype=jnp.complex64),),
        normalize=True,
    )
    _, evidence = tn.apply_charge_covariant_kraus(
        channel,
        state,
        maximum_purification_dimension=1,
    )
    assert evidence.trace_residual > 0.0
    assert not evidence.valid
    group = AbelianGroup((2,))
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
    group = AbelianGroup((2,))
    leg = tn.AbelianLeg(group, ((0,),), (1,), orientation=1)
    layout = tn.AbelianTensorLayout((leg,), total_charge=(1,))
    assert layout.sectors == ()

    tensor = tn.AbelianTensor.from_dense(
        layout,
        jnp.zeros((1,), dtype=jnp.complex64),
    )
    assert tensor.blocks == ()
    assert jnp.array_equal(tensor.to_dense(), jnp.zeros((1,), dtype=jnp.complex64))

    with pytest.raises(Exception, match="charge-forbidden"):
        value = tn.AbelianTensor.from_dense(
            layout,
            jnp.ones((1,), dtype=jnp.complex64),
        )
        value.to_dense().block_until_ready()
