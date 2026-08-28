#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp

import phydrax as phx


class OffsetTransform(phx.solver.AbstractAcceptedStepTransform):
    offset: float = eqx.field(static=True)
    transform_id: str = eqx.field(static=True)

    def __init__(self, offset):
        self.offset = float(offset)
        self.transform_id = f"transform:offset:{offset}"

    def apply(self, step_index, time, previous_state, candidate_state, args, /):
        del step_index, time, previous_state, args
        transformed = candidate_state + self.offset
        return phx.solver.AcceptedStepTransformResult(
            transformed,
            jnp.asarray(True),
            jnp.asarray(True),
            jnp.sqrt(jnp.sum((transformed - candidate_state) ** 2)),
        )


def test_fixed_step_ssprk_solves_and_saves_requested_stride():
    method = phx.solver.SSPRK33FixedStepMethod(lambda time, state, args: -state)
    problem = phx.solver.FixedStepProblem(
        method,
        jnp.asarray([1.0]),
        t0=0.0,
        t1=0.1,
        step_size=0.01,
    )
    solution = phx.solver.solve_fixed_step(problem, save_every=2)

    assert solution.successful
    assert solution.times.shape == (6,)
    assert solution.states.shape == (6, 1)
    assert jnp.all(solution.valid)
    assert jnp.allclose(solution.states[-1, 0], jnp.exp(-0.1), rtol=2e-6)


def test_fixed_step_composes_accepted_step_transforms():
    transform = phx.solver.CompositeAcceptedStepTransform(
        (OffsetTransform(0.1), OffsetTransform(-0.05))
    )
    method = phx.solver.SSPRK54FixedStepMethod(
        lambda time, state, args: jnp.zeros_like(state), transform=transform
    )
    solution = phx.solver.solve_fixed_step(
        phx.solver.FixedStepProblem(
            method,
            jnp.asarray([0.0]),
            t0=0.0,
            t1=0.03,
            step_size=0.01,
        )
    )

    assert solution.successful
    assert jnp.all(solution.transform_applied)
    assert jnp.allclose(solution.states[-1], jnp.asarray([0.15]))
