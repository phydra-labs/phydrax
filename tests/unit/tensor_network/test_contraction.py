import equinox as eqx
import jax.numpy as jnp
import pytest

import phydrax as phx


tn = phx.tensor_network


def _matrix_structure():
    return tn.ContractionStructure(
        (
            tn.ContractionOperand(
                "left", (tn.ContractionLeg("i", 2), tn.ContractionLeg("j", 3))
            ),
            tn.ContractionOperand(
                "right", (tn.ContractionLeg("j", 3), tn.ContractionLeg("k", 4))
            ),
        ),
        ("k", "i"),
    )


def test_labeled_plan_preserves_output_order_refresh_and_jit():
    structure = _matrix_structure()
    plan = tn.plan_contraction(structure, dtype="float64", optimizer="optimal")
    left = jnp.arange(6.0).reshape((2, 3))
    right = jnp.arange(12.0).reshape((3, 4))
    prepared = tn.prepare_contraction(plan, (left, right))
    result = eqx.filter_jit(tn.execute_contraction)(prepared)

    assert result.value.shape == (4, 2)
    assert jnp.allclose(result.value, (left @ right).T)
    assert result.evidence.structure_id == structure.structure_id
    assert result.evidence.finite

    refreshed = tn.refresh_contraction(prepared, (left + 1.0, right))
    refreshed_result = tn.execute_contraction(refreshed)
    assert refreshed.prepared_id != prepared.prepared_id
    assert refreshed.operand_id != prepared.operand_id
    assert refreshed.numeric_version == prepared.numeric_version + 1
    assert refreshed_result.evidence.replay_id != result.evidence.replay_id
    assert jnp.allclose(refreshed_result.value, ((left + 1.0) @ right).T)


def test_contraction_structure_supports_hyperedges_and_rejects_resource_overflow():
    hyperedge = tn.ContractionStructure(
        tuple(
            tn.ContractionOperand(f"operand-{index}", (tn.ContractionLeg("shared", 2),))
            for index in range(3)
        ),
        (),
    )
    hyperplan = tn.plan_contraction(hyperedge, dtype="float64")
    hyperresult = tn.execute_contraction(
        tn.prepare_contraction(
            hyperplan,
            (
                jnp.asarray([1.0, 2.0]),
                jnp.asarray([3.0, 4.0]),
                jnp.asarray([5.0, 6.0]),
            ),
        )
    )
    assert jnp.allclose(hyperresult.value, 63.0)
    with pytest.raises(MemoryError, match="intermediate"):
        tn.plan_contraction(
            _matrix_structure(),
            dtype="float64",
            resources=tn.ContractionResourcePolicy(maximum_intermediate_elements=1),
        )


def test_contraction_workspace_accounts_for_precision_widening():
    structure = _matrix_structure()
    narrow = tn.plan_contraction(
        structure,
        dtype="float32",
        precision=tn.TensorNetworkPrecisionPolicy(
            storage_dtype="float32",
            contraction_dtype="float32",
            output_dtype="float32",
        ),
    )
    wide_precision = tn.TensorNetworkPrecisionPolicy(
        storage_dtype="float32",
        contraction_dtype="float64",
        output_dtype="float64",
    )
    wide = tn.plan_contraction(
        structure,
        dtype="float32",
        precision=wide_precision,
    )
    assert wide.cost.peak_live_bytes > narrow.cost.peak_live_bytes
    with pytest.raises(MemoryError, match="workspace"):
        tn.plan_contraction(
            structure,
            dtype="float32",
            precision=wide_precision,
            resources=tn.ContractionResourcePolicy(
                maximum_workspace_bytes=narrow.cost.peak_live_bytes
            ),
        )


def test_contraction_planner_deadline_is_enforced_during_search():
    with pytest.raises(TimeoutError, match="planning exceeded"):
        tn.plan_contraction(
            _matrix_structure(),
            dtype="float32",
            planner=tn.ContractionPlannerPolicy(maximum_planning_seconds=1e-9),
        )


def test_prepared_mps_and_mpo_inner_consumers_match_native_environments():
    identity = jnp.eye(2, dtype=jnp.complex128)
    pauli_x = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    state = tn.product_mps(jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.complex128))
    operator = tn.product_mpo(jnp.stack((pauli_x, identity)))
    mps_result = tn.execute_contraction(tn.prepare_mps_inner_contraction(state, state))
    mpo_result = tn.execute_contraction(
        tn.prepare_mpo_inner_contraction(operator, operator)
    )

    assert jnp.allclose(mps_result.value, tn.mps_inner(state, state))
    assert jnp.allclose(mpo_result.value, tn.mpo_inner(operator, operator))
