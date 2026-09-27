from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.linalg as la
from tests._support.differentiation import (
    assert_coordinate_transpose_duality,
    assert_hilbert_adjoint_duality,
)
from tests.unit.linalg._cases import operator_cases


pytestmark = [pytest.mark.strict_jax, pytest.mark.filterwarnings("error")]


_MATERIALIZATION = la.MaterializationPolicy(max_entries=100_000, max_bytes=1_000_000)


def test_operator_contracts_scenario_1() -> None:
    for case in operator_cases():
        operator = case.build()
        np.testing.assert_allclose(
            operator.mv(case.primal),
            case.matrix @ case.primal,
            rtol=1e-12,
            atol=1e-12,
            err_msg=case.case_id,
        )
        np.testing.assert_allclose(
            operator.transpose_mv(case.target),
            case.matrix.T @ case.target,
            rtol=1e-12,
            atol=1e-12,
            err_msg=case.case_id,
        )
        np.testing.assert_allclose(
            operator.adjoint_mv(case.target),
            jnp.conj(case.matrix.T) @ case.target,
            rtol=1e-12,
            atol=1e-12,
            err_msg=case.case_id,
        )
        np.testing.assert_allclose(
            la.materialize(operator, _MATERIALIZATION),
            case.matrix,
            rtol=1e-12,
            atol=1e-12,
            err_msg=case.case_id,
        )
        assert operator.capabilities.diagonal_assembly is case.supports_diagonal
        np.testing.assert_allclose(
            jax.jit(la.assemble_diagonal)(operator),
            jnp.diag(case.matrix),
            rtol=1e-12,
            atol=1e-12,
            err_msg=case.case_id,
        )
        assert_coordinate_transpose_duality(
            operator,
            case.primal,
            case.target,
            rtol=1e-12,
            atol=1e-12,
        )
        assert_hilbert_adjoint_duality(
            operator,
            case.primal,
            case.target,
            rtol=1e-12,
            atol=1e-12,
        )
    source_weights = jnp.asarray([2.0, 5.0], dtype=jnp.float64)
    target_weights = jnp.asarray([3.0, 7.0], dtype=jnp.float64)
    source = la.ArraySpace(
        (2,),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(source_weights),
        space_id="weighted-dual-source",
    )
    target = la.ArraySpace(
        (2,),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(target_weights),
        space_id="weighted-dual-target",
    )
    matrix = jnp.asarray([[1.0, -2.0], [3.0, 0.5]], dtype=jnp.float64)
    operator = la.DenseLinearOperator(matrix, source=source, target=target)
    primal = jnp.asarray([2.0, -1.0], dtype=jnp.float64)
    covector = jnp.asarray([0.25, 4.0], dtype=jnp.float64)
    algebraic = la.dual_transpose(operator)
    hilbert = la.adjoint(operator)
    expected_hilbert = (
        jnp.reciprocal(source_weights)[:, None] * matrix.T * target_weights[None, :]
    ) @ covector

    assert isinstance(algebraic, la.DualTransposeLinearOperator)
    np.testing.assert_allclose(algebraic.mv(covector), matrix.T @ covector)
    np.testing.assert_allclose(hilbert.mv(covector), expected_hilbert)
    assert not bool(jnp.allclose(algebraic.mv(covector), hilbert.mv(covector)))
    assert algebraic.source.compatible(la.DualSpace(target))
    assert algebraic.target.compatible(la.DualSpace(source))
    assert hilbert.source.compatible(target)
    assert hilbert.target.compatible(source)
    primal_block = jnp.stack((primal, 2.0 * primal, -primal), axis=-1)
    target_block = jnp.stack((covector, -0.5 * covector, 3.0 * covector), axis=-1)
    expected_adjoint_block = (
        jnp.reciprocal(source_weights)[:, None] * matrix.T * target_weights[None, :]
    ) @ target_block
    np.testing.assert_allclose(operator.mv_block(primal_block), matrix @ primal_block)
    np.testing.assert_allclose(
        operator.transpose_mv_block(target_block),
        matrix.T @ target_block,
    )
    np.testing.assert_allclose(
        operator.adjoint_mv_block(target_block),
        expected_adjoint_block,
    )
    assert_coordinate_transpose_duality(
        operator,
        primal,
        covector,
        rtol=1e-12,
        atol=1e-12,
    )
    assert_hilbert_adjoint_duality(
        operator,
        primal,
        covector,
        rtol=1e-12,
        atol=1e-12,
    )
    source = la.ArraySpace((2,), dtype=jnp.float64, space_id="composition-source")
    middle = la.ArraySpace((3,), dtype=jnp.float64, space_id="composition-middle")
    target = la.ArraySpace((2,), dtype=jnp.float64, space_id="composition-target")
    right = la.DenseLinearOperator(
        jnp.asarray([[1.0, 2.0], [-1.0, 0.5], [3.0, -2.0]]),
        source=source,
        target=middle,
    )
    left = la.DenseLinearOperator(
        jnp.asarray([[2.0, -1.0, 0.25], [0.0, 4.0, -3.0]]),
        source=middle,
        target=target,
    )
    covector = jnp.asarray([1.5, -0.25], dtype=jnp.float64)
    composed = la.dual_transpose(left @ right)
    reversed_composition = la.dual_transpose(right) @ la.dual_transpose(left)
    np.testing.assert_allclose(composed.mv(covector), reversed_composition.mv(covector))

    identity = la.IdentityLinearOperator(source)
    dual_identity = la.dual_transpose(identity)
    source_covector = jnp.asarray([3.0, -2.0], dtype=jnp.float64)
    np.testing.assert_allclose(dual_identity.mv(source_covector), source_covector)
    assert la.dual_transpose(dual_identity) is identity

    weights = jnp.asarray([2.0, 3.0, 5.0], dtype=jnp.float64)
    primal_space = la.ArraySpace(
        (3,),
        dtype=jnp.float64,
        pairing=la.DiagonalPairing(weights),
        space_id="mass-primal",
    )
    dual_space = la.DualSpace(primal_space)
    mass = la.DenseLinearOperator(
        jnp.diag(weights),
        source=primal_space,
        target=dual_space,
        operator_id="mass-map",
    )
    inverse_mass = la.DenseLinearOperator(
        jnp.diag(jnp.reciprocal(weights)),
        source=dual_space,
        target=primal_space,
        operator_id="inverse-mass-map",
    )
    vector = jnp.asarray([1.5, -2.0, 0.25], dtype=jnp.float64)
    covector3 = jnp.asarray([4.0, -3.0, 2.0], dtype=jnp.float64)
    np.testing.assert_allclose((inverse_mass @ mass).mv(vector), vector)
    np.testing.assert_allclose((mass @ inverse_mass).mv(covector3), covector3)

    tree_space = la.PyTreeSpace(
        {
            "first": jnp.zeros((2,), dtype=jnp.float64),
            "second": jnp.zeros((1,), dtype=jnp.float64),
        }
    )
    tree_dual = la.DualSpace(tree_space)
    tree_covector = {
        "first": jnp.asarray([1.0, -2.0]),
        "second": jnp.asarray([3.0]),
    }
    tree_vector = {
        "first": jnp.asarray([4.0, 5.0]),
        "second": jnp.asarray([-1.0]),
    }
    validated = tree_dual.validate(tree_covector)
    assert jnp.array_equal(validated["first"], tree_covector["first"])
    assert jnp.array_equal(validated["second"], tree_covector["second"])
    assert jnp.allclose(tree_dual.pair(tree_covector, tree_vector), -9.0)


def test_block_actions_preserve_columns_and_fusion_declarations() -> None:
    matrix = jnp.asarray([[2.0, -1.0], [0.5, 3.0]])
    space = la.ArraySpace((2,), dtype=matrix.dtype)
    generic = la.FunctionLinearOperator(
        lambda vector: matrix @ vector,
        source=space,
        target=space,
        transpose_action=lambda vector: matrix.T @ vector,
    )
    block = jnp.asarray([[1.0, 2.0, -3.0], [4.0, -5.0, 6.0]])
    assert not generic.supports_fused_block_action
    np.testing.assert_allclose(generic.mv_block(block), matrix @ block)
    np.testing.assert_allclose(generic.transpose_mv_block(block), matrix.T @ block)
    np.testing.assert_allclose(generic.adjoint_mv_block(block), matrix.T @ block)

    diagonal = la.DiagonalLinearOperator(jnp.asarray([2.0, -3.0]))
    assert diagonal.supports_fused_block_action
    np.testing.assert_array_equal(
        diagonal.mv_block(block),
        jnp.asarray([2.0, -3.0])[:, None] * block,
    )
