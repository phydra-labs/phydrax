#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class LocalImplicitSourcePlan(StrictModule, NonTrainableState):
    """Batched point-local backward-Euler source solve with fixed Newton work."""

    source: Callable[[Array, Any], ArrayLike] = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: Callable[[Array, Any], ArrayLike],
        /,
        *,
        iterations: int = 6,
        tolerance: float = 1e-9,
    ):
        if not callable(source):
            raise TypeError("source must be callable.")
        iterations_ = int(iterations)
        tolerance_ = float(tolerance)
        if iterations_ <= 0 or not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Newton iterations/tolerance must be positive.")
        self.source = source
        self.iterations = iterations_
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "local-implicit-source",
                "source": repr(source),
                "iterations": iterations_,
                "tolerance": tolerance_,
            }
        )

    def step(
        self,
        state: ArrayLike,
        step_size: ArrayLike,
        args: Any = None,
        /,
    ) -> Array:
        values = jnp.asarray(state)
        dt = jnp.asarray(step_size)
        if values.ndim < 2 or dt.shape != ():
            raise ValueError(
                "Local implicit state must end in a channel axis and dt scalar."
            )
        channels = int(values.shape[-1])
        points = values.reshape((-1, channels))

        def solve_point(initial: Array) -> Array:
            def residual(candidate: Array) -> Array:
                source = jnp.asarray(self.source(candidate, args))
                if source.shape != candidate.shape:
                    raise ValueError("Local source must preserve channel shape.")
                return candidate - initial - dt * source

            def iteration(_: int, candidate: Array) -> Array:
                value = residual(candidate)
                jacobian = jax.jacfwd(residual)(candidate)
                correction = jnp.linalg.solve(jacobian, value)
                return candidate - correction

            result = jax.lax.fori_loop(0, self.iterations, iteration, initial)
            result = eqx.error_if(
                result,
                jnp.linalg.norm(residual(result)) > self.tolerance,
                "Local implicit source Newton solve did not converge.",
            )
            return result

        return jax.vmap(solve_point)(points).reshape(values.shape)


class StrangSplitPlan(StrictModule, NonTrainableState):
    """Explicit composition of transport and source evolution operators."""

    transport_step: Callable[[Array, Array, Array, Any], ArrayLike] = eqx.field(
        static=True
    )
    source_step: Callable[[Array, Array, Array, Any], ArrayLike] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transport_step: Callable[[Array, Array, Array, Any], ArrayLike],
        source_step: Callable[[Array, Array, Array, Any], ArrayLike],
        /,
    ):
        if not callable(transport_step) or not callable(source_step):
            raise TypeError("transport_step and source_step must be callable.")
        self.transport_step = transport_step
        self.source_step = source_step
        self.plan_id = canonical_fingerprint(
            {
                "kind": "strang-split",
                "transport": repr(transport_step),
                "source": repr(source_step),
            }
        )

    def step(
        self,
        time: Array,
        state: Array,
        step_size: ArrayLike,
        args: Any = None,
    ) -> Array:
        dt = jnp.asarray(step_size)
        first = jnp.asarray(self.source_step(time, state, 0.5 * dt, args))
        transported = jnp.asarray(self.transport_step(time, first, dt, args))
        result = jnp.asarray(
            self.source_step(jnp.asarray(time) + dt, transported, 0.5 * dt, args)
        )
        if result.shape != state.shape:
            raise ValueError("Strang split stages must preserve state shape.")
        return result


__all__ = ["LocalImplicitSourcePlan", "StrangSplitPlan"]
