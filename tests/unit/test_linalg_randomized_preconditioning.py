#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


la = phx.linalg


def _psd_properties(*, positive_definite=False):
    evidence = {
        "self_adjoint": "construction",
        "positive_semidefinite": "construction",
    }
    if positive_definite:
        evidence["positive_definite"] = "construction"
    return la.OperatorProperties(
        self_adjoint=True,
        positive_semidefinite=True,
        positive_definite=positive_definite,
        evidence=evidence,
    )


def _operator(
    matrix,
    *,
    operator_id="nystrom-test",
    positive_definite=False,
):
    values = jnp.asarray(matrix)
    space = la.ArraySpace((values.shape[0],), dtype=values.dtype)
    return la.DenseLinearOperator(
        values,
        source=space,
        target=space,
        properties=_psd_properties(positive_definite=positive_definite),
        operator_id=operator_id,
    )


def test_exact_low_rank_nystrom_action_matches_shifted_inverse_and_jit():
    matrix = jnp.diag(jnp.asarray([4.0, 2.0, 0.0, 0.0]))
    operator = _operator(matrix)
    builder = la.RandomizedNystromPreconditionerBuilder(
        2,
        oversampling=0,
        shift=0.5,
        seed=3,
    )
    action = builder.prepare(operator, materialization=la.MaterializationPolicy())
    rhs = jnp.asarray([1.0, -2.0, 3.0, 0.5])
    expected = jnp.linalg.solve(matrix + 0.5 * jnp.eye(4), rhs)

    assert jnp.allclose(action.apply(rhs), expected, rtol=2e-5, atol=2e-6)
    assert jnp.allclose(
        jax.jit(lambda value: action.apply(value))(rhs),
        expected,
        rtol=2e-5,
        atol=2e-6,
    )
    assert int(action.diagnostics.effective_rank) == 2
    assert action.diagnostics.setup_matvec_count == 2
    assert bool(action.diagnostics.valid)
    assert action.properties.certifies("positive_definite")


def test_randomized_nystrom_preconditions_native_pcg_without_changing_solution():
    diagonal = jnp.asarray([20.0, 8.0, 2.0, 0.5])
    base = _operator(jnp.diag(diagonal), operator_id="nystrom-pcg-base")
    shifted = _operator(
        jnp.diag(diagonal + 0.25),
        operator_id="nystrom-pcg-shifted",
        positive_definite=True,
    )
    rhs = jnp.asarray([1.0, 2.0, -1.0, 0.5])
    builder = la.RandomizedNystromPreconditionerBuilder(
        3,
        oversampling=1,
        shift=0.25,
        seed=4,
    )
    result = la.solve(
        la.LinearSystem(shifted),
        rhs,
        policy=la.LinearSolvePolicy(
            la.PCG(),
            preconditioning=la.PreconditioningPolicy(
                builder,
                setup_operator=base,
            ),
            differentiation=la.DifferentiationPolicy("mathematical"),
        ),
    )

    assert bool(result.successful)
    assert jnp.allclose(result.value, rhs / (diagonal + 0.25), rtol=1e-6, atol=1e-7)
    assert result.provenance.preconditioner_id is not None
    assert result.provenance.preconditioner_setup_matvec_count == 4


def test_numeric_refresh_reuses_or_redraws_probes_without_shape_changes():
    first = _operator(jnp.diag(jnp.asarray([5.0, 3.0, 1.0, 0.2])))
    second = _operator(
        jnp.diag(jnp.asarray([6.0, 2.0, 0.8, 0.1])),
        operator_id="nystrom-refresh-second",
    )
    materialization = la.MaterializationPolicy()
    reuse = la.RandomizedNystromPreconditionerBuilder(
        2,
        oversampling=1,
        shift=0.1,
        seed=7,
        probe_refresh="reuse",
    )
    redraw = la.RandomizedNystromPreconditionerBuilder(
        2,
        oversampling=1,
        shift=0.1,
        seed=7,
        probe_refresh="redraw",
    )
    reuse_initial = reuse.prepare(first, materialization=materialization)
    reuse_updated = reuse.refresh(
        reuse_initial,
        second,
        materialization=materialization,
    )
    redraw_initial = redraw.prepare(first, materialization=materialization)
    redraw_updated = redraw.refresh(
        redraw_initial,
        second,
        materialization=materialization,
    )

    assert reuse_updated.basis.shape == reuse_initial.basis.shape
    assert redraw_updated.basis.shape == redraw_initial.basis.shape
    assert reuse_updated.diagnostics.refresh_count == 1
    assert redraw_updated.diagnostics.refresh_count == 1
    assert not jnp.allclose(redraw_updated.basis, reuse_updated.basis)


def test_complex_hermitian_nystrom_action_is_finite_and_self_adjoint():
    matrix = jnp.asarray(
        [
            [3.0, 1.0j, 0.0],
            [-1.0j, 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=jnp.complex128,
    )
    action = la.RandomizedNystromPreconditionerBuilder(
        2,
        oversampling=0,
        shift=0.2,
        seed=9,
    ).prepare(_operator(matrix), materialization=la.MaterializationPolicy())
    left = jnp.asarray([1.0 + 0.5j, -0.3j, 2.0])
    right = jnp.asarray([-0.5j, 1.5, 0.25 - 0.1j])

    assert jnp.all(jnp.isfinite(action.apply(left)))
    assert jnp.allclose(
        jnp.vdot(left, action.apply(right)),
        jnp.vdot(action.apply(left), right),
        rtol=1e-6,
        atol=1e-7,
    )


def test_randomized_nystrom_cost_reports_fixed_sketch_work():
    operator = _operator(jnp.diag(jnp.arange(1.0, 7.0)))
    builder = la.RandomizedNystromPreconditionerBuilder(
        2,
        oversampling=3,
        shift=0.5,
    )
    cost = builder.cost_for(operator)

    assert cost.accepted
    assert cost.setup_matvec_count == 5
    assert cost.storage_bytes > 0
    assert cost.preparation_workspace_bytes > cost.storage_bytes


def test_randomized_nystrom_rejects_invalid_configuration_and_operator_claims():
    with pytest.raises(ValueError, match="rank"):
        la.RandomizedNystromPreconditionerBuilder(0)
    with pytest.raises(ValueError, match="shift"):
        la.RandomizedNystromPreconditionerBuilder(1, shift=0.0)
    operator = _operator(jnp.eye(2))
    with pytest.raises(ValueError, match=r"rank \+ oversampling"):
        la.RandomizedNystromPreconditionerBuilder(2, oversampling=1).cost_for(operator)

    uncertified = la.DenseLinearOperator(jnp.eye(2))
    with pytest.raises(ValueError, match="self-adjointness"):
        la.RandomizedNystromPreconditionerBuilder(
            1,
            oversampling=1,
        ).cost_for(uncertified)

    indefinite = _operator(jnp.diag(jnp.asarray([1.0, -0.5])))
    with pytest.raises(RuntimeError, match="indefinite"):
        la.RandomizedNystromPreconditionerBuilder(
            1,
            oversampling=1,
            shift=0.1,
        ).prepare(indefinite, materialization=la.MaterializationPolicy())
