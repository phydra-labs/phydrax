#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....solver.maxwell import (
    CompatibleMaxwellRunResult,
    PreparedCompatibleMaxwell,
    solve_compatible_maxwell,
)


class GaussianDerivativeWaveform(StrictModule, NonTrainableState):
    center_frequency_Hz: float = eqx.field(static=True)
    delay_s: float = eqx.field(static=True)
    amplitude: float = eqx.field(static=True)

    def __init__(
        self, center_frequency_Hz: float, delay_s: float, amplitude: float = 1.0, /
    ):
        frequency, delay, amplitude_ = (
            float(center_frequency_Hz),
            float(delay_s),
            float(amplitude),
        )
        if (
            not np.isfinite(frequency)
            or frequency <= 0
            or not np.isfinite(delay)
            or delay < 0
            or not np.isfinite(amplitude_)
        ):
            raise ValueError("GPR waveform frequency/delay/amplitude are invalid.")
        self.center_frequency_Hz, self.delay_s, self.amplitude = (
            frequency,
            delay,
            amplitude_,
        )

    def __call__(self, time_s: ArrayLike, args: object = None, /) -> Array:
        del args
        time = jnp.asarray(time_s)
        argument = jnp.pi * self.center_frequency_Hz * (time - self.delay_s)
        return self.amplitude * (-2.0 * argument) * jnp.exp(-(argument**2))


class GPRResult(StrictModule):
    run: CompatibleMaxwellRunResult
    finite: Array
    passive: Array
    maximum_stable_step_s: Array
    plan_id: str = eqx.field(static=True)


class DispersiveFullWaveGPRPlan(StrictModule, NonTrainableState):
    """Full displacement-current GPR over compatible Maxwell ADE plus CPML."""

    runtime: PreparedCompatibleMaxwell
    step_size_s: Array
    step_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        runtime: PreparedCompatibleMaxwell,
        step_size_s: ArrayLike,
        step_count: int,
        /,
    ):
        if not isinstance(runtime, PreparedCompatibleMaxwell):
            raise TypeError("GPR requires a prepared compatible Maxwell runtime.")
        if not runtime.capabilities.dispersive or not runtime.capabilities.passive:
            raise ValueError("Full-wave GPR requires passive dispersive ADE material.")
        if runtime.pml is None or not runtime.capabilities.pml:
            raise ValueError("Full-wave GPR requires prepared Maxwell CPML.")
        if not runtime.observers or not runtime.sources:
            raise ValueError("GPR runtime requires explicit sources and observers.")
        if any(source.envelope is None for source in runtime.sources):
            raise ValueError("GPR sources require explicit real transient envelopes.")
        step = jnp.asarray(step_size_s)
        steps = int(step_count)
        if step.shape != () or steps <= 0:
            raise ValueError("GPR timestep/count are invalid.")
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step <= 0) | (step > runtime.stable_dt),
            "GPR timestep must be finite, positive, and within Maxwell CFL evidence.",
        )
        self.runtime, self.step_size_s, self.step_count = runtime, step, steps
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dispersive-full-wave-gpr",
                "runtime": runtime.prepared_id,
                "step_size_s": float(np.asarray(step)),
                "step_count": steps,
                "material": type(runtime.constitutive).__name__,
                "boundary": type(runtime.pml).__name__,
            }
        )

    def simulate(
        self,
        args: object = None,
        /,
        *,
        initial_state=None,
        start_time_s: ArrayLike = 0.0,
    ) -> GPRResult:
        start = jnp.asarray(start_time_s)
        if start.shape != () or not bool(jnp.isfinite(start)):
            raise ValueError("GPR start time must be a finite scalar.")
        source_samples = tuple(
            source.sample(start, args) for source in self.runtime.sources
        )
        if any(
            jnp.iscomplexobj(sample.electric_current)
            or jnp.iscomplexobj(sample.magnetic_current)
            for sample in source_samples
        ):
            raise ValueError("GPR transient source envelopes must produce real forcing.")
        state = (
            self.runtime.initialize()
            if initial_state is None
            else self.runtime._state(initial_state)
        )
        run = solve_compatible_maxwell(
            self.runtime,
            state,
            start_time_s,
            self.step_size_s,
            self.step_count,
            args,
        )
        leaves = tuple(
            leaf
            for leaf in jax.tree_util.tree_leaves((run.final_state, run.diagnostics))
            if isinstance(leaf, jax.Array)
        )
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))
        diagnostics = run.diagnostics
        passive = (
            self.runtime.capabilities.passive
            & (self.step_size_s <= diagnostics.stable_step)
            & finite
        )
        return GPRResult(
            run,
            jnp.asarray(finite),
            jnp.asarray(passive),
            self.runtime.stable_dt,
            self.plan_id,
        )


__all__ = [
    "DispersiveFullWaveGPRPlan",
    "GPRResult",
    "GaussianDerivativeWaveform",
]
