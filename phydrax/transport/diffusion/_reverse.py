#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite, prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._score_field import StateTimeScoreField
from ..._strict import StrictModule
from ...domain import DomainFunction
from ...stochastic._gaussian_diffusion import (
    AbstractGaussianDiffusion,
    DiffusionTerminalReference,
    TerminalReferenceRelationship,
)
from ...stochastic._wiener import WienerRealization


def _sample_shape(value, /) -> tuple[int, ...]:
    shape = tuple(int(size) for size in value)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("Reverse diffusion requires a non-empty positive sample_shape.")
    return shape


class _ReverseScoreDrift(eqx.Module):
    process: AbstractGaussianDiffusion
    score: StateTimeScoreField

    def __call__(self, reverse_time: Array, state: Array, score_key: Array, /) -> Array:
        forward_time = self.process.terminal_time - reverse_time
        scale = self.process.diffusion_scale(forward_time).astype(state.dtype)
        score = self.score(state, forward_time, key=score_key)
        return -self.process.drift(forward_time, state) + scale**2 * score


class _ReverseDiffusionCoefficient(eqx.Module):
    process: AbstractGaussianDiffusion

    def __call__(self, reverse_time: Array, state: Array, args: Any, /) -> Array:
        del args
        forward_time = self.process.terminal_time - reverse_time
        scale = self.process.diffusion_scale(forward_time).astype(state.dtype)
        return jnp.broadcast_to(scale, state.shape)


class ReverseDiffusionRealization(StrictModule):
    """Terminal states, score key, and global Wiener paths for exact replay."""

    terminal_states: Array
    score_key: Array
    wiener: WienerRealization
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    score_id: str = eqx.field(static=True)
    terminal_reference_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)

    def __init__(
        self,
        terminal_states: ArrayLike,
        score_key: Key[Array, ""],
        wiener: WienerRealization,
        /,
        *,
        process_id: str,
        score_id: str,
        terminal_reference_id: str,
        realization_id: str,
    ):
        if not isinstance(wiener, WienerRealization):
            raise TypeError("wiener must be a WienerRealization.")
        samples = tuple(wiener.sample_shape)
        events = tuple(wiener.noise_shape)
        states = jnp.asarray(terminal_states)
        expected = samples + events
        if states.shape != expected:
            raise ValueError(
                f"terminal_states must have shape {expected}; got {states.shape}."
            )
        for owner, value in (
            ("process_id", process_id),
            ("score_id", score_id),
            ("terminal_reference_id", terminal_reference_id),
            ("realization_id", realization_id),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{owner} must be a non-empty string.")
        self.terminal_states = states
        self.score_key = jnp.asarray(score_key)
        self.wiener = wiener
        self.sample_shape = samples
        self.state_shape = events
        self.process_id = process_id
        self.score_id = score_id
        self.terminal_reference_id = terminal_reference_id
        self.realization_id = realization_id


class ReverseDiffusionResult(StrictModule):
    """Reverse-diffusion terminal states and canonical differential solution evidence."""
    terminal_states: Array
    solution: Any
    residual_signal_scale: Array
    process_id: str = eqx.field(static=True)
    score_id: str = eqx.field(static=True)
    terminal_reference_id: str = eqx.field(static=True)
    terminal_relationship: TerminalReferenceRelationship = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)

    @property
    def sample_shape(self) -> tuple[int, ...]:
        return self.solution.sample_shape

    @property
    def state_shape(self) -> tuple[int, ...]:
        return tuple(self.terminal_states.shape[len(self.sample_shape) :])

    @property
    def final_states(self) -> Array:
        return jnp.take(self.solution.states, -1, axis=len(self.sample_shape))

    @property
    def valid(self) -> Array:
        return self.solution.successful

    @property
    def successful(self) -> Array:
        return jnp.all(self.valid)

    @property
    def num_samples(self) -> int:
        return prod(self.sample_shape)

    def to_stochastic_trajectory(
        self,
        /,
        *,
        realization_axes=None,
        state_axes=("state",),
    ):
        return self.solution.to_stochastic_trajectory(
            initial_state=self.terminal_states,
            initial_time=0.0,
            realization_axes=realization_axes,
            state_axes=state_axes,
            approximation_id=self.terminal_relationship,
            metadata={
                "process_id": self.process_id,
                "score_id": self.score_id,
                "terminal_reference_id": self.terminal_reference_id,
                "reverse_diffusion_realization_id": self.realization_id,
                "transport_id": self.transport_id,
            },
        )


class ReverseDiffusion(StrictModule):
    """Sample a learned score law through the canonical reverse-time Itô SDE."""

    process: AbstractGaussianDiffusion
    score: StateTimeScoreField
    terminal_reference: DiffusionTerminalReference
    solver: Any
    stepsize_controller: Any
    adjoint: Any
    precision: Any
    dt0: float = eqx.field(static=True)
    wiener_tolerance: float = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    score_id: str = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)

    def __init__(
        self,
        process: AbstractGaussianDiffusion,
        score: DomainFunction,
        terminal_reference: DiffusionTerminalReference,
        /,
        *,
        score_id: str,
        dt0: float,
        wiener_tolerance: float,
        state_label: str = "x",
        time_label: str = "t",
        solver: Any = None,
        stepsize_controller: Any = None,
        adjoint: Any = None,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        max_steps: int = 4096,
        precision: Any = None,
        transport_id: str | None = None,
    ):
        if not isinstance(process, AbstractGaussianDiffusion):
            raise TypeError("process must implement AbstractGaussianDiffusion.")
        if not isinstance(terminal_reference, DiffusionTerminalReference):
            raise TypeError("terminal_reference must be a DiffusionTerminalReference.")
        if terminal_reference.process_id != process.process_id:
            raise ValueError("Terminal reference and diffusion process IDs must match.")
        if tuple(terminal_reference.law.event_shape) != process.state_shape:
            raise ValueError("Terminal-reference event shape must match the process state.")
        if not isinstance(score_id, str) or not score_id:
            raise ValueError("score_id must be a non-empty string.")
        step = float(dt0)
        tolerance = float(wiener_tolerance)
        if not isfinite(step) or step <= 0.0:
            raise ValueError("dt0 must be finite and positive.")
        if not isfinite(tolerance) or tolerance <= 0.0 or tolerance >= step:
            raise ValueError("wiener_tolerance must be positive and strictly below dt0.")
        relative = float(rtol)
        absolute = float(atol)
        if not isfinite(relative) or not isfinite(absolute) or relative <= 0.0 or absolute <= 0.0:
            raise ValueError("Reverse-diffusion tolerances must be finite and positive.")
        limit = int(max_steps)
        if limit <= 0:
            raise ValueError("max_steps must be positive.")
        if precision is not None:
            from ...solver._temporal_precision import TemporalPrecisionPolicy

            if not isinstance(precision, TemporalPrecisionPolicy):
                raise TypeError("precision must be a TemporalPrecisionPolicy or None.")
        score_field = StateTimeScoreField(
            score,
            state_label=state_label,
            time_label=time_label,
        )
        resolved_id = transport_id or canonical_fingerprint(
            {
                "kind": "reverse-score-diffusion",
                "process_id": process.process_id,
                "score_id": score_id,
                "terminal_reference_id": terminal_reference.reference_id,
                "dt0": step,
                "wiener_tolerance": tolerance,
                "rtol": relative,
                "atol": absolute,
                "max_steps": limit,
            }
        )
        if not isinstance(resolved_id, str) or not resolved_id:
            raise ValueError("transport_id must be a non-empty string or None.")
        self.process = process
        self.score = score_field
        self.terminal_reference = terminal_reference
        self.solver = solver
        self.stepsize_controller = stepsize_controller
        self.adjoint = adjoint
        self.precision = precision
        self.dt0 = step
        self.wiener_tolerance = tolerance
        self.rtol = relative
        self.atol = absolute
        self.max_steps = limit
        self.score_id = score_id
        self.transport_id = resolved_id

    def realize(
        self,
        key: Key[Array, ""],
        sample_shape,
        /,
    ) -> ReverseDiffusionRealization:
        samples = _sample_shape(sample_shape)
        count = prod(samples)
        terminal_key = jr.fold_in(key, 0x7465726D)
        score_key = jr.fold_in(key, 0x73636F72)
        wiener_key = jr.fold_in(key, 0x7769656E)
        indices = jnp.arange(count, dtype=jnp.uint32)
        keys = jax.vmap(lambda index: jr.fold_in(terminal_key, index))(indices)
        flat_terminal = jax.vmap(self.terminal_reference.law.sample)(keys)
        terminal = flat_terminal.reshape(samples + self.process.state_shape)
        support = self.terminal_reference.law.contains(terminal)
        terminal = eqx.error_if(
            terminal,
            jnp.any(~support),
            "Terminal reference generated a state outside its declared support.",
        )
        wiener = WienerRealization.independent(
            wiener_key,
            self.process.state_shape,
            support=(0.0, self.process.terminal_time),
            sample_shape=samples,
            tolerance=self.wiener_tolerance,
            levy_area="brownian",
            noise_id=self.process.process_id,
            label="reverse-score-diffusion",
        )
        realization_id = canonical_fingerprint(
            {
                "kind": "reverse-diffusion-realization",
                "terminal_states": array_tree_fingerprint(terminal),
                "wiener_realization_id": wiener.realization_id,
                "score_key": array_tree_fingerprint(score_key),
                "process_id": self.process.process_id,
                "score_id": self.score_id,
                "terminal_reference_id": self.terminal_reference.reference_id,
            }
        )
        return ReverseDiffusionRealization(
            terminal,
            score_key,
            wiener,
            process_id=self.process.process_id,
            score_id=self.score_id,
            terminal_reference_id=self.terminal_reference.reference_id,
            realization_id=realization_id,
        )

    def solve(
        self,
        realization: ReverseDiffusionRealization,
        /,
        *,
        save_times: ArrayLike | None = None,
    ) -> ReverseDiffusionResult:
        if not isinstance(realization, ReverseDiffusionRealization):
            raise TypeError("realization must be a ReverseDiffusionRealization.")
        if realization.process_id != self.process.process_id or realization.score_id != self.score_id:
            raise ValueError("Reverse-diffusion realization does not match this transport.")
        if realization.terminal_reference_id != self.terminal_reference.reference_id:
            raise ValueError("Reverse-diffusion terminal reference does not match.")
        times = (
            jnp.asarray([self.process.terminal_time], dtype=realization.terminal_states.dtype)
            if save_times is None
            else jnp.asarray(save_times, dtype=realization.terminal_states.dtype)
        )
        if times.ndim != 1 or int(times.shape[0]) <= 0:
            raise ValueError("save_times must be a non-empty vector.")
        if bool(
            jnp.any(~jnp.isfinite(times))
            | jnp.any(times <= 0.0)
            | jnp.any(times > self.process.terminal_time)
            | jnp.any(jnp.diff(times) <= 0.0)
        ):
            raise ValueError(
                "Reverse-diffusion save_times must increase within (0, terminal_time]."
            )
        from ...solver._differential import DifferentialProblem, WienerTerm
        from ...solver._diffrax_backend import solve_diffrax_ensemble

        drift = _ReverseScoreDrift(self.process, self.score)
        coefficient = _ReverseDiffusionCoefficient(self.process)
        term = WienerTerm(
            "reverse-diffusion-noise",
            coefficient,
            self.process.state_shape,
            structure="additive",
            basis_id=self.process.process_id,
            representation="diagonal",
        )
        problem = DifferentialProblem(
            drift,
            jnp.zeros(self.process.state_shape, dtype=realization.terminal_states.dtype),
            t0=0.0,
            t1=self.process.terminal_time,
            args=realization.score_key,
            wiener_terms=(term,),
            interpretation="ito",
            problem_id=canonical_fingerprint(
                {
                    "kind": "reverse-diffusion-problem",
                    "transport_id": self.transport_id,
                    "realization_id": realization.realization_id,
                }
            ),
        )
        solution = solve_diffrax_ensemble(
            problem,
            save_times=times,
            realization=realization.wiener,
            initial_states=realization.terminal_states,
            solver=self.solver,
            stepsize_controller=self.stepsize_controller,
            adjoint=self.adjoint,
            dt0=self.dt0,
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
            throw=False,
            precision=self.precision,
        )
        return ReverseDiffusionResult(
            terminal_states=realization.terminal_states,
            solution=solution,
            residual_signal_scale=self.terminal_reference.residual_signal_scale,
            process_id=self.process.process_id,
            score_id=self.score_id,
            terminal_reference_id=self.terminal_reference.reference_id,
            terminal_relationship=self.terminal_reference.relationship,
            realization_id=realization.realization_id,
            transport_id=self.transport_id,
        )

    def sample_with_diagnostics(
        self,
        key: Key[Array, ""],
        sample_shape,
        /,
        *,
        save_times: ArrayLike | None = None,
    ) -> ReverseDiffusionResult:
        realization = self.realize(key, sample_shape)
        return self.solve(realization, save_times=save_times)

    def sample(
        self,
        key: Key[Array, ""],
        sample_shape,
        /,
    ) -> Array:
        result = self.sample_with_diagnostics(key, sample_shape)
        return eqx.error_if(
            result.final_states,
            ~result.successful,
            "Reverse diffusion failed for at least one sample.",
        )


__all__ = [
    "ReverseDiffusion",
    "ReverseDiffusionRealization",
    "ReverseDiffusionResult",
]
