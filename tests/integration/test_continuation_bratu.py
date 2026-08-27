#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def test_weighted_bratu_branch_crosses_the_discrete_fold():
    size = 7
    spacing = 1.0 / (size - 1)
    weights = (
        jnp.full((size,), spacing, dtype=jnp.float64)
        .at[0]
        .set(spacing / 2.0)
        .at[-1]
        .set(spacing / 2.0)
    )
    laplacian = (
        2.0 * jnp.eye(size, dtype=jnp.float64)
        - jnp.eye(size, k=1, dtype=jnp.float64)
        - jnp.eye(size, k=-1, dtype=jnp.float64)
    ) / spacing**2

    def residual(state, parameter, args):
        del args
        value = laplacian @ state - parameter * jnp.exp(state)
        return value.at[0].set(state[0]).at[-1].set(state[-1])

    state_space = phx.linalg.ArraySpace(
        (size,),
        dtype=jnp.float64,
        pairing=phx.linalg.DiagonalPairing(weights),
        space_id="bratu-weighted-state",
    )
    residual_space = phx.linalg.ArraySpace(
        (size,),
        dtype=jnp.float64,
        space_id="bratu-residual",
    )
    problem = phx.continuation.ParameterContinuationProblem(
        residual,
        state_space=state_space,
        residual_space=residual_space,
        problem_id="bratu-finite-difference",
    )
    result = phx.continuation.continue_branch(
        problem,
        jnp.zeros((size,), dtype=jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
        num_steps=14,
        method=phx.continuation.PseudoArclengthContinuation(
            initial_step=0.25,
            maximum_step=0.45,
            tangent_update="bordered",
        ),
    )

    assert result.status == phx.continuation.ContinuationStatus.SUCCESS
    assert result.fold_brackets
    maximum_parameter = max(float(point.coordinate) for point in result.points)
    assert 3.3 < maximum_parameter < 3.55
    assert min(float(point.tangent_coordinate) for point in result.points) < 0.0
    assert max(float(point.residual_norm) for point in result.points) <= 1e-8
    assert result.branch.geometry.execution_state_space.space_id == state_space.space_id
