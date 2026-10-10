#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
import math
from collections.abc import Callable
from typing import Any, final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier, positive_finite_float


class AbstractStepSizeController(StrictModule):
    """Accepted-step size control for `ProductionRunPlan`.

    The controller state is an array-only PyTree carried by the compiled production
    scan and persisted in every production checkpoint. `propose` returns the positive
    nominal size of the next step before end-time and output-schedule clamping.
    `update` is evaluated on device for every attempted step, and the runtime keeps
    its result only for accepted steps. `minimum_step` and `maximum_step` bound every
    proposal; `maximum_step` also bounds the preflight step-count estimate.
    """

    controller_id: eqx.AbstractVar[str]
    minimum_step: eqx.AbstractVar[float]
    maximum_step: eqx.AbstractVar[float]

    @abc.abstractmethod
    def initial_state(self, step_size: Array, /) -> PyTree[Array]:
        """Return the controller state whose first proposal is `step_size`."""

        raise NotImplementedError

    @abc.abstractmethod
    def propose(self, controller_state: PyTree[Array], time: Array, /) -> Array:
        """Return the positive scalar nominal step proposed at `time`."""

        raise NotImplementedError

    @abc.abstractmethod
    def update(
        self,
        controller_state: PyTree[Array],
        previous_state: PyTree[Array],
        accepted_state: PyTree[Array],
        accepted_step: Array,
        retry_count: Array,
        args: Any,
        /,
    ) -> PyTree[Array]:
        """Return the controller state after one accepted step.

        `retry_count` is the number of rejected retry attempts that preceded the
        accepted attempt. The result must keep the PyTree structure, shapes, and
        dtypes of `controller_state`.
        """

        raise NotImplementedError


@final
class IncrementLimitedStepController(AbstractStepSizeController, NonTrainableState):
    """Scale the step so that every physical increment approaches its limit.

    `increments(previous, accepted, step, args)` returns a one-dimensional array of
    nonnegative increment magnitudes of the accepted step, aligned with `limits`.
    With `r = safety * min_i(limit_i / max(|increment_i|, tiny))`, the next proposal
    is `clip(step * clip(r, shrink, growth), minimum_step, maximum_step)`, where
    `step` is the accepted step size and `tiny` is the smallest normal number of
    its dtype. The controller state is the last proposed step.
    """

    increments: Callable[[PyTree[Array], PyTree[Array], Array, Any], Array]
    limits: Array
    minimum_step: float = eqx.field(static=True)
    maximum_step: float = eqx.field(static=True)
    growth: float = eqx.field(static=True)
    shrink: float = eqx.field(static=True)
    safety: float = eqx.field(static=True)
    controller_id: str = eqx.field(static=True)

    def __init__(
        self,
        increments: Callable[[PyTree[Array], PyTree[Array], Array, Any], Array],
        limits: ArrayLike,
        /,
        *,
        minimum_step: float,
        maximum_step: float,
        growth: float = 2.0,
        shrink: float = 0.25,
        safety: float = 0.9,
        controller_id: str,
    ) -> None:
        if not callable(increments):
            raise TypeError("Step-controller increments must be callable.")
        identifier = canonical_identifier(controller_id, "controller_id")
        host_limits = np.asarray(limits, dtype=np.float64)
        if (
            host_limits.ndim != 1
            or host_limits.size == 0
            or not np.all(np.isfinite(host_limits))
            or np.any(host_limits <= 0.0)
        ):
            raise ValueError(
                "Step-controller limits must be a nonempty one-dimensional array of "
                "finite positive values."
            )
        minimum = positive_finite_float(minimum_step, "minimum_step")
        maximum = positive_finite_float(maximum_step, "maximum_step")
        if minimum > maximum:
            raise ValueError("Step-controller minimum_step cannot exceed maximum_step.")
        growth_ = float(growth)
        shrink_ = float(shrink)
        safety_ = float(safety)
        if not math.isfinite(growth_) or growth_ < 1.0:
            raise ValueError("Step-controller growth must be finite and at least one.")
        if not math.isfinite(shrink_) or not 0.0 < shrink_ <= 1.0:
            raise ValueError("Step-controller shrink must lie in (0, 1].")
        if not math.isfinite(safety_) or not 0.0 < safety_ <= 1.0:
            raise ValueError("Step-controller safety must lie in (0, 1].")
        self.increments = increments
        self.limits = jnp.asarray(host_limits)
        self.minimum_step = minimum
        self.maximum_step = maximum
        self.growth = growth_
        self.shrink = shrink_
        self.safety = safety_
        self.controller_id = canonical_fingerprint(
            {
                "kind": "increment-limited-step-controller",
                "increments": identifier,
                "limits": host_limits.tolist(),
                "minimum_step": minimum,
                "maximum_step": maximum,
                "growth": growth_,
                "shrink": shrink_,
                "safety": safety_,
            }
        )

    def initial_state(self, step_size: Array, /) -> Array:
        step = jnp.asarray(step_size)
        if step.shape != () or not jnp.issubdtype(step.dtype, jnp.floating):
            raise TypeError("Step-controller initial step must be a real scalar array.")
        return step

    def propose(self, controller_state: Array, time: Array, /) -> Array:
        del time
        return controller_state

    def update(
        self,
        controller_state: Array,
        previous_state: PyTree[Array],
        accepted_state: PyTree[Array],
        accepted_step: Array,
        retry_count: Array,
        args: Any,
        /,
    ) -> Array:
        del controller_state, retry_count
        step = jnp.asarray(accepted_step)
        increments = jnp.asarray(
            self.increments(previous_state, accepted_state, step, args)
        )
        if increments.shape != self.limits.shape:
            raise ValueError(
                "Step-controller increments must be one-dimensional and match limits."
            )
        dtype = step.dtype
        magnitude = jnp.maximum(
            jnp.abs(increments.astype(dtype)), jnp.asarray(jnp.finfo(dtype).tiny, dtype)
        )
        ratio = self.safety * jnp.min(self.limits.astype(dtype) / magnitude)
        factor = jnp.clip(ratio, self.shrink, self.growth)
        return jnp.clip(step * factor, self.minimum_step, self.maximum_step)


__all__ = ["AbstractStepSizeController", "IncrementLimitedStepController"]
