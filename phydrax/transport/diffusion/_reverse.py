#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite, prod
from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array, core as jax_core
from jax.typing import ArrayLike

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
from ...typing import checked, PRNGKey
from ._guidance import ScoreContext
from ._probability_flow import bound_score_context, score_context_arguments


if TYPE_CHECKING:
    from ...stochastic._trajectory import StochasticTrajectory


_TERMINAL_TAG = 0x7465726D
_SCORE_TAG = 0x73636F72
_WIENER_TAG = 0x7769656E
_WIENER_LABEL = "reverse-score-diffusion"


def _sample_shape(value: Sequence[int], /) -> tuple[int, ...]:
    shape = tuple(value)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("Reverse diffusion requires a non-empty positive sample_shape.")
    return shape


def _host_save_times(
    values: ArrayLike | None, terminal_time: float, /
) -> tuple[float, ...]:
    """Static solve layout: increasing save times in ``(0, terminal_time]``."""
    if values is None:
        return (float(terminal_time),)
    if any(isinstance(leaf, jax_core.Tracer) for leaf in jax.tree.leaves(values)):
        raise TypeError(
            "Reverse-diffusion save_times are static solve layout and cannot be traced."
        )
    times = np.asarray(values, dtype=np.float64)
    if times.ndim != 1 or times.shape[0] <= 0:
        raise ValueError("Reverse-diffusion save_times must be a non-empty vector.")
    if (
        not np.all(np.isfinite(times))
        or np.any(times <= 0.0)
        or np.any(times > terminal_time)
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError(
            "Reverse-diffusion save_times must increase within (0, terminal_time]."
        )
    return tuple(float(value) for value in times)


def _scalar_key(key: PRNGKey, /) -> None:
    try:
        data = jr.key_data(key)
    except (TypeError, ValueError) as error:
        raise TypeError("Reverse diffusion realizes from one JAX PRNG key.") from error
    if data.shape != (2,):
        raise ValueError("Reverse diffusion realizes from one scalar JAX PRNG key.")


def _realization_identity(law_id: str, root_key: Array, /) -> str:
    data = jr.key_data(root_key)
    if isinstance(data, jax_core.Tracer):
        raise ValueError(
            "A traced reverse-diffusion realization has no host identity; bind its "
            "law_id and root key data instead."
        )
    return canonical_fingerprint(
        {
            "kind": "reverse-diffusion-realization",
            "law_id": law_id,
            "root_key": array_tree_fingerprint(data),
        }
    )


class _ReverseScoreDrift(StrictModule):
    process: AbstractGaussianDiffusion
    score: StateTimeScoreField

    def __call__(
        self, reverse_time: Array, state: Array, args: tuple[Array, ScoreContext], /
    ) -> Array:
        score_key, context = args
        forward_time = self.process.terminal_time - reverse_time
        scale = self.process.diffusion_scale(forward_time).astype(state.dtype)
        score = self.score(
            state,
            forward_time,
            key=score_key,
            context=score_context_arguments(self.score, context),
        )
        return -self.process.drift(forward_time, state) + scale**2 * score


class _ReverseDiffusionCoefficient(StrictModule):
    process: AbstractGaussianDiffusion

    def __call__(self, reverse_time: Array, state: Array, args: Any, /) -> Array:
        del args
        forward_time = self.process.terminal_time - reverse_time
        scale = self.process.diffusion_scale(forward_time).astype(state.dtype)
        return jnp.broadcast_to(scale, state.shape)


class ReverseDiffusionRealization(StrictModule):
    """Key-addressed terminal states, score key and global Wiener paths for exact replay.

    Every array is derived from ``root_key`` by the realization law ``law_id``
    (process, score, terminal reference, Wiener path law, sample shape and the
    fixed fold-in tags).  The complete identity is ``(law_id, root key data)``:
    `realization_id` materializes it on the host, while traced code carries the
    realization itself, so realizing works under ``jax.jit`` and ``jax.vmap``
    and a new key never retraces.
    """

    root_key: Array
    terminal_states: Array
    score_key: Array
    wiener: WienerRealization
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    state_shape: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)
    score_id: str = eqx.field(static=True)
    terminal_reference_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(self, prepared: PreparedReverseDiffusion, key: PRNGKey, /) -> None:
        if not isinstance(prepared, PreparedReverseDiffusion):
            raise TypeError("prepared must be a PreparedReverseDiffusion.")
        _scalar_key(key)
        transport = prepared.transport
        process = transport.process
        reference = transport.terminal_reference
        samples = prepared.sample_shape
        terminal_key = jr.fold_in(key, _TERMINAL_TAG)
        score_key = jr.fold_in(key, _SCORE_TAG)
        wiener_key = jr.fold_in(key, _WIENER_TAG)
        indices = jnp.arange(prod(samples), dtype=jnp.uint32)
        keys = jax.vmap(lambda index: jr.fold_in(terminal_key, index))(indices)
        flat_terminal = jax.vmap(reference.law.sample)(keys)
        terminal = flat_terminal.reshape(samples + process.state_shape)
        terminal = eqx.error_if(
            terminal,
            jnp.any(~reference.law.contains(terminal)),
            "Terminal reference generated a state outside its declared support.",
        )
        wiener = WienerRealization.independent(
            wiener_key,
            process.state_shape,
            support=(0.0, process.terminal_time),
            sample_shape=samples,
            tolerance=transport.wiener_tolerance,
            levy_area="brownian",
            noise_id=process.process_id,
            label=_WIENER_LABEL,
        )
        self.root_key = key
        self.terminal_states = terminal
        self.score_key = score_key
        self.wiener = wiener
        self.sample_shape = samples
        self.state_shape = process.state_shape
        self.process_id = process.process_id
        self.score_id = transport.score_id
        self.terminal_reference_id = reference.reference_id
        self.law_id = prepared.law_id

    @property
    def realization_id(self) -> str:
        """Host identity of ``(law_id, root key data)``; refuses traced keys."""
        return _realization_identity(self.law_id, self.root_key)


class ReverseDiffusionResult(StrictModule):
    """Reverse-diffusion terminal states and canonical differential solution evidence.

    ``root_key`` with ``realization_law_id`` is the solved realization's complete
    identity (host `realization_id`); ``prepared_id`` names the fixed solve layout
    and ``context_id`` the structure of the score context.
    """

    terminal_states: Array
    solution: Any
    residual_signal_scale: Array
    root_key: Array
    process_id: str = eqx.field(static=True)
    score_id: str = eqx.field(static=True)
    terminal_reference_id: str = eqx.field(static=True)
    terminal_relationship: TerminalReferenceRelationship = eqx.field(static=True)
    realization_law_id: str = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    context_id: str = eqx.field(static=True)

    @property
    def realization_id(self) -> str:
        return _realization_identity(self.realization_law_id, self.root_key)

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
        realization_axes: Sequence[str] | None = None,
        state_axes: Sequence[str] = ("state",),
    ) -> StochasticTrajectory:
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
                "reverse_diffusion_realization_law_id": self.realization_law_id,
                "prepared_id": self.prepared_id,
                "context_id": self.context_id,
                "transport_id": self.transport_id,
            },
        )


class PreparedReverseDiffusion(StrictModule):
    """Immutable solve layout of one reverse diffusion over a fixed sample shape.

    Preparation validates the static save times on the host once and fixes the
    realization law ``law_id``, the differential ``problem_id`` and
    ``prepared_id``; none of them depends on a key.  `realize` and `solve` are
    numerical operations without array hashing or host synchronization, so they
    compose with ``jax.jit``, ``jax.vmap`` and bounded lane maps.  The score
    context of a solve is dynamic data passed with the score key as
    differential-problem arguments; the realization binding (root key, terminal
    states, Wiener paths) stays distinct from the problem identity.
    """

    transport: ReverseDiffusion
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    save_times: tuple[float, ...] = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        transport: ReverseDiffusion,
        sample_shape: Sequence[int],
        /,
        *,
        save_times: ArrayLike | None = None,
    ) -> None:
        if not isinstance(transport, ReverseDiffusion):
            raise TypeError("transport must be a ReverseDiffusion.")
        samples = _sample_shape(sample_shape)
        process = transport.process
        times = _host_save_times(save_times, process.terminal_time)
        law_id = canonical_fingerprint(
            {
                "kind": "reverse-diffusion-realization-law",
                "process_id": process.process_id,
                "score_id": transport.score_id,
                "terminal_reference_id": transport.terminal_reference.reference_id,
                "state_shape": list(process.state_shape),
                "sample_shape": list(samples),
                "wiener": {
                    "support": [0.0, process.terminal_time],
                    "tolerance": transport.wiener_tolerance,
                    "levy_area": "brownian",
                    "noise_id": process.process_id,
                    "label": _WIENER_LABEL,
                },
                "derivation": {
                    "terminal": _TERMINAL_TAG,
                    "score": _SCORE_TAG,
                    "wiener": _WIENER_TAG,
                    "terminal_paths": "fold_in(terminal_key, flat_sample_index)",
                },
            }
        )
        problem_id = canonical_fingerprint(
            {
                "kind": "reverse-diffusion-problem",
                "transport_id": transport.transport_id,
                "sample_shape": list(samples),
                "context_labels": list(transport.score.context_labels),
            }
        )
        self.transport = transport
        self.sample_shape = samples
        self.save_times = times
        self.law_id = law_id
        self.problem_id = problem_id
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-reverse-diffusion",
                "transport_id": transport.transport_id,
                "law_id": law_id,
                "problem_id": problem_id,
                "save_times": list(times),
            }
        )

    def realize(self, key: PRNGKey, /) -> ReverseDiffusionRealization:
        """Terminal states, score key and Wiener paths derived from ``key``."""
        return ReverseDiffusionRealization(self, key)

    def solve(
        self,
        realization: ReverseDiffusionRealization,
        /,
        *,
        context: ScoreContext | None = None,
    ) -> ReverseDiffusionResult:
        """Advance the realization's terminal states under the score context."""
        if not isinstance(realization, ReverseDiffusionRealization):
            raise TypeError("realization must be a ReverseDiffusionRealization.")
        transport = self.transport
        process = transport.process
        reference = transport.terminal_reference
        if (
            realization.process_id != process.process_id
            or realization.score_id != transport.score_id
        ):
            raise ValueError(
                "Reverse-diffusion realization does not match this transport."
            )
        if realization.terminal_reference_id != reference.reference_id:
            raise ValueError("Reverse-diffusion terminal reference does not match.")
        if realization.sample_shape != self.sample_shape:
            raise ValueError(
                "Reverse-diffusion realization does not match the prepared sample shape."
            )
        resolved = bound_score_context(transport.score, context)
        from ...solver._differential import DifferentialProblem, WienerTerm
        from ...solver._diffrax_backend import solve_diffrax_ensemble

        dtype = realization.terminal_states.dtype
        term = WienerTerm(
            "reverse-diffusion-noise",
            _ReverseDiffusionCoefficient(process),
            process.state_shape,
            structure="additive",
            basis_id=process.process_id,
            representation="diagonal",
        )
        problem = DifferentialProblem(
            _ReverseScoreDrift(process, transport.score),
            jnp.zeros(process.state_shape, dtype=dtype),
            t0=0.0,
            t1=process.terminal_time,
            args=(realization.score_key, resolved),
            wiener_terms=(term,),
            interpretation="ito",
            problem_id=self.problem_id,
        )
        solution = solve_diffrax_ensemble(
            problem,
            save_times=jnp.asarray(self.save_times, dtype=dtype),
            realization=realization.wiener,
            initial_states=realization.terminal_states,
            solver=transport.solver,
            stepsize_controller=transport.stepsize_controller,
            adjoint=transport.adjoint,
            dt0=transport.dt0,
            rtol=transport.rtol,
            atol=transport.atol,
            max_steps=transport.max_steps,
            throw=False,
            precision=transport.precision,
        )
        return ReverseDiffusionResult(
            terminal_states=realization.terminal_states,
            solution=solution,
            residual_signal_scale=reference.residual_signal_scale,
            root_key=realization.root_key,
            process_id=process.process_id,
            score_id=transport.score_id,
            terminal_reference_id=reference.reference_id,
            terminal_relationship=reference.relationship,
            realization_law_id=realization.law_id,
            transport_id=transport.transport_id,
            prepared_id=self.prepared_id,
            context_id=resolved.context_id,
        )

    def sample_with_diagnostics(
        self, key: PRNGKey, /, *, context: ScoreContext | None = None
    ) -> ReverseDiffusionResult:
        return self.solve(self.realize(key), context=context)


class ReverseDiffusion(StrictModule):
    """Sample a learned score law through the canonical reverse-time Itô SDE.

    ``context_labels`` declares the named score context (a `ScoreContext` shared
    by every sample of one realization); `prepare` fixes the solve layout once.
    """

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

    @checked
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
        context_labels: Sequence[str] = (),
        solver: Any = None,
        stepsize_controller: Any = None,
        adjoint: Any = None,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        max_steps: int = 4096,
        precision: Any = None,
        transport_id: str | None = None,
    ) -> None:
        if terminal_reference.process_id != process.process_id:
            raise ValueError("Terminal reference and diffusion process IDs must match.")
        if tuple(terminal_reference.law.event_shape) != process.state_shape:
            raise ValueError(
                "Terminal-reference event shape must match the process state."
            )
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
        if (
            not isfinite(relative)
            or not isfinite(absolute)
            or relative <= 0.0
            or absolute <= 0.0
        ):
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
            context_labels=context_labels,
        )
        resolved_id = transport_id or canonical_fingerprint(
            {
                "kind": "reverse-score-diffusion",
                "process_id": process.process_id,
                "score_id": score_id,
                "terminal_reference_id": terminal_reference.reference_id,
                "context_labels": list(score_field.context_labels),
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

    def prepare(
        self,
        sample_shape: Sequence[int],
        /,
        *,
        save_times: ArrayLike | None = None,
    ) -> PreparedReverseDiffusion:
        """Fixed solve layout of ``sample_shape`` samples (host-validated save times)."""
        return PreparedReverseDiffusion(self, sample_shape, save_times=save_times)

    def realize(
        self,
        key: PRNGKey,
        sample_shape: Sequence[int],
        /,
    ) -> ReverseDiffusionRealization:
        return self.prepare(sample_shape).realize(key)

    @checked
    def solve(
        self,
        realization: ReverseDiffusionRealization,
        /,
        *,
        context: ScoreContext | None = None,
        save_times: ArrayLike | None = None,
    ) -> ReverseDiffusionResult:
        return self.prepare(realization.sample_shape, save_times=save_times).solve(
            realization, context=context
        )

    def sample_with_diagnostics(
        self,
        key: PRNGKey,
        sample_shape: Sequence[int],
        /,
        *,
        context: ScoreContext | None = None,
        save_times: ArrayLike | None = None,
    ) -> ReverseDiffusionResult:
        return self.prepare(sample_shape, save_times=save_times).sample_with_diagnostics(
            key, context=context
        )

    def sample(
        self,
        key: PRNGKey,
        sample_shape: Sequence[int],
        /,
        *,
        context: ScoreContext | None = None,
    ) -> Array:
        result = self.sample_with_diagnostics(key, sample_shape, context=context)
        return eqx.error_if(
            result.final_states,
            ~result.successful,
            "Reverse diffusion failed for at least one sample.",
        )


__all__ = [
    "PreparedReverseDiffusion",
    "ReverseDiffusion",
    "ReverseDiffusionRealization",
    "ReverseDiffusionResult",
]
