#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Free-draining overdamped FIB diffusion with keyed pathwise noise."""

import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    space = phx.linalg.ArraySpace((4096,))
    mobility = phx.linalg.FunctionLinearOperator(
        lambda value: value,
        source=space,
        target=space,
        transpose_action=lambda value: value,
        properties=phx.linalg.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
            },
        ),
        operator_id="free-draining-mobility",
    )
    method = phx.solver.FIBOverdampedPlan(
        space,
        lambda _position: mobility,
        temperature=1.0,
        boltzmann_constant=1.0,
    )
    step_size = 1.0e-2
    key = phx.solver.StochasticReplayKey(
        jnp.asarray(2026),
        jnp.asarray(0),
        jnp.asarray(0),
        jnp.asarray(0),
    )
    result = method.step(
        jnp.zeros(space.shape),
        jnp.zeros(space.shape),
        step_size,
        key,
    )
    observed = jnp.mean(result.brownian_increment**2)
    expected = 2.0 * step_size
    relative_error = jnp.abs(observed - expected) / expected
    if not bool(result.accepted) or float(relative_error) > 0.1:
        raise RuntimeError("Free FIB diffusion failed its covariance qualification.")
    print(
        {
            "observed_variance": float(observed),
            "expected_variance": expected,
            "relative_error": float(relative_error),
        }
    )


if __name__ == "__main__":
    main()
