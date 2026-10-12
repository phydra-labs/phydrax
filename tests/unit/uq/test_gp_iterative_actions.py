# Copyright © 2026 PHYDRA, Inc. All rights reserved.


from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _state() -> Any:
    return phx.uq.GaussianProcessLikelihoodState(
        kernel=phx.kernels.SquaredExponentialKernel(length_scale=0.4),
        noise_scale=0.05,
    )


def test_gp_iterative_actions_scenario_1() -> None:
    points = jnp.linspace(0.0, 1.0, 6)[:, None]
    residual = jnp.asarray([1.0, -0.2, 0.3, 0.1, -0.4, 0.2])
    policies = (
        phx.uq.LanczosGaussianProcessActionPolicy(jnp.ones((6,)), max_actions=4),
        phx.uq.ConjugateGradientGaussianProcessActionPolicy(4),
        phx.uq.GaussSeidelGaussianProcessActionPolicy(4),
    )
    for policy in policies:
        resolved = policy.resolve(
            points,
            state=_state(),
            residual=residual if policy.requires_residual else None,
        )
        assert resolved.active_mask.shape == (4,)
        assert resolved.residual_history.shape == (5,)
        assert resolved.operator.source.size == 4
    points = jnp.linspace(0.0, 1.0, 5)[:, None]
    discrepancy = phx.uq.ComputationAwareGaussianProcessDiscrepancy(
        points, jnp.sin(points[:, 0])
    )
    policy = phx.uq.ConjugateGradientGaussianProcessActionPolicy(3)
    with pytest.raises(ValueError, match="residual-dependent"):
        discrepancy.factor(state=_state(), actions=policy)
    condition = discrepancy.condition(
        jnp.zeros((5,)),
        points,
        state=_state(),
        actions=policy,
    )
    assert condition.mean.shape == (5,)


def test_fixed_gauss_seidel_order_resolves_under_filtered_jit() -> None:
    points = jnp.linspace(0.0, 1.0, 5)[:, None]
    observations = jnp.sin(points[:, 0])
    residual = observations - 0.1
    policy = phx.uq.GaussSeidelGaussianProcessActionPolicy(
        3,
        ordering="fixed",
        fixed_order=jnp.asarray([2, 0, 4]),
    )

    @eqx.filter_jit
    def selected(current_policy: Any) -> Any:
        resolved = current_policy.resolve(
            points,
            state=_state(),
            residual=residual,
        )
        return resolved.selected_indices

    assert jnp.array_equal(selected(policy), jnp.asarray([2, 0, 4]))


@pytest.mark.parametrize("kind", ["lanczos", "conjugate-gradient", "gauss-seidel"])
def test_bounded_iterative_actions_match_dense_reference(kind: str) -> None:
    points = jnp.linspace(0.0, 1.0, 9)[:, None]
    state = _state()
    residual = np.asarray([1.0, -0.2, 0.3, 0.1, -0.4, 0.2, 0.6, -0.1, 0.8])
    covariance = np.array(state.kernel.matrix(points, points), copy=True)
    covariance += np.diag(
        np.broadcast_to(np.asarray(state.noise_scale**2 + state.jitter), (9,))
    )
    if kind == "lanczos":
        policy = phx.uq.LanczosGaussianProcessActionPolicy(jnp.ones(9), max_actions=4)
        vector = np.ones(9) / 3.0
        previous = np.zeros(9)
        beta = 0.0
        reference = []
        for _ in range(4):
            reference.append(vector.copy())
            candidate = covariance @ vector - beta * previous
            for column in reference:
                candidate -= np.vdot(column, candidate).real * column
            beta = np.linalg.norm(candidate)
            previous, vector = vector, candidate / beta
    elif kind == "conjugate-gradient":
        policy = phx.uq.ConjugateGradientGaussianProcessActionPolicy(4)
        remainder = residual.copy()
        direction = residual.copy()
        reference = []
        for _ in range(4):
            reference.append(direction / np.linalg.norm(direction))
            product = covariance @ direction
            squared = np.dot(remainder, remainder)
            updated = remainder - squared / np.dot(direction, product) * product
            direction = updated + np.dot(updated, updated) / squared * direction
            remainder = updated
    else:
        policy = phx.uq.GaussSeidelGaussianProcessActionPolicy(
            4, ordering="largest-residual"
        )
        remainder = residual.copy()
        used = np.zeros(9, dtype=np.bool_)
        reference = []
        reference_history = [np.linalg.norm(remainder)]
        for _ in range(4):
            index = np.argmax(np.where(used, -np.inf, np.abs(remainder)))
            column = np.zeros(9)
            column[index] = 1.0
            reference.append(column)
            remainder -= (
                remainder[index] / covariance[index, index] * covariance[:, index]
            )
            used[index] = True
            reference_history.append(np.linalg.norm(remainder))
    resolved = policy.resolve(
        points,
        state=state,
        residual=jnp.asarray(residual),
        max_workspace_bytes=80,
        checkpoint_kernel_blocks=False,
    )
    actual = phx.linalg.materialize(
        resolved.operator,
        phx.linalg.MaterializationPolicy(max_entries=36, max_bytes=288),
    )
    np.testing.assert_allclose(
        actual, np.stack(reference, axis=1), atol=2e-10, rtol=2e-10
    )
    if kind == "gauss-seidel":
        np.testing.assert_allclose(
            resolved.residual_history, reference_history, atol=2e-10, rtol=2e-10
        )
    with pytest.raises(ValueError, match="one kernel row"):
        policy.resolve(
            points, state=state, residual=jnp.asarray(residual), max_workspace_bytes=79
        )


@pytest.mark.parametrize("fault", ["dropped-row", "wrong-pivot"])
def test_iterative_dense_oracle_detects_planted_fault(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    from phydrax.uq import _gp_computation_backend as backend

    if fault == "dropped-row":
        original = backend._kernel_matrix_action

        def dropped(*args: Any, **kwargs: Any) -> Any:
            action, batch, workspace, entries = original(*args, **kwargs)
            return action.at[-1].set(0.0), batch, workspace, entries

        monkeypatch.setattr(backend, "_kernel_matrix_action", dropped)
        kind = "lanczos"
    else:
        diagonal = backend._GaussianProcessCovarianceOperator.diagonal

        def wrong_pivot(operator: Any) -> Any:
            return 2.0 * diagonal(operator)

        monkeypatch.setattr(
            backend._GaussianProcessCovarianceOperator, "diagonal", wrong_pivot
        )
        kind = "gauss-seidel"
    with pytest.raises(AssertionError):
        test_bounded_iterative_actions_match_dense_reference(kind)
