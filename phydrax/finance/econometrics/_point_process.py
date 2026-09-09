#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...optim import minimize, NewtonTrustRegion, OptimizationTermination
from ...stochastic._point_process import (
    evaluate_hawkes_likelihood,
    ExponentialHawkesProcess,
    HawkesLikelihoodPlan,
    HawkesLikelihoodResult,
    PointProcessObservation,
    prepare_hawkes_likelihood,
)
from ..core import FinanceEvidenceBinding, PhysicalLaw
from ._datasets import PreparedMarketEventStream


class MarketPointProcessDefinition(StrictModule):
    """Market-event observation window and same-time likelihood convention."""

    start_time_ns: int = eqx.field(static=True)
    end_time_ns: int = eqx.field(static=True)
    tie_policy: Literal["simultaneous", "ordered"] = eqx.field(static=True)
    require_stable: bool = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        start_time_ns: int,
        end_time_ns: int,
        /,
        *,
        tie_policy: Literal["simultaneous", "ordered"] = "simultaneous",
        require_stable: bool = True,
    ):
        start = int(start_time_ns)
        end = int(end_time_ns)
        if end <= start:
            raise ValueError("end_time_ns must exceed start_time_ns.")
        if tie_policy not in ("simultaneous", "ordered"):
            raise ValueError("tie_policy must be simultaneous or ordered.")
        self.start_time_ns = start
        self.end_time_ns = end
        self.tie_policy = tie_policy
        self.require_stable = bool(require_stable)
        self.definition_id = canonical_fingerprint(
            {
                "kind": "market-point-process-definition",
                "start_time_ns": start,
                "end_time_ns": end,
                "tie_policy": tie_policy,
                "require_stable": bool(require_stable),
            }
        )


class HawkesDefinition(StrictModule):
    """Initial exponential-Hawkes parameters and fixed optimization budget."""

    baseline: Array
    excitation: Array
    decay: Array
    maximum_steps: int = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        baseline: ArrayLike,
        excitation: ArrayLike,
        decay: ArrayLike,
        /,
        *,
        maximum_steps: int = 128,
    ):
        process = ExponentialHawkesProcess(baseline, excitation, decay)
        steps = int(maximum_steps)
        if steps < 1:
            raise ValueError("maximum_steps must be positive.")
        self.baseline = process.baseline
        self.excitation = process.excitation
        self.decay = process.decay
        self.maximum_steps = steps
        self.definition_id = canonical_fingerprint(
            {
                "kind": "finance-hawkes-definition",
                "baseline": tuple(float(value) for value in process.baseline.tolist()),
                "excitation": tuple(
                    tuple(float(value) for value in row)
                    for row in process.excitation.tolist()
                ),
                "decay": tuple(
                    tuple(float(value) for value in row) for row in process.decay.tolist()
                ),
                "maximum_steps": steps,
            }
        )


class PointProcessPlan(StrictModule):
    observation: PointProcessObservation
    likelihood_plan: HawkesLikelihoodPlan = eqx.field(static=True)
    initial_process: ExponentialHawkesProcess
    stream_id: str = eqx.field(static=True)
    market_definition_id: str = eqx.field(static=True)
    hawkes_definition_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PointProcessFit(StrictModule):
    process: ExponentialHawkesProcess
    likelihood: HawkesLikelihoodResult
    initial_likelihood: HawkesLikelihoodResult
    optimizer_status: Array
    optimizer_iterations: Array
    gradient_norm: Array
    evidence: FinanceEvidenceBinding = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)


class PointProcessResult(StrictModule):
    log_likelihood: Array
    compensator: Array
    event_intensity: Array
    spectral_radius: Array
    stable: Array
    status: Array
    fit: PointProcessFit


def _evidence(stream_id: str, model_id: str, route: str) -> FinanceEvidenceBinding:
    return FinanceEvidenceBinding(
        (
            canonical_fingerprint(
                {"kind": "point-process-data-evidence", "stream": stream_id}
            ),
        ),
        (
            canonical_fingerprint(
                {"kind": "point-process-model-evidence", "model": model_id}
            ),
        ),
        (
            canonical_fingerprint(
                {"kind": "point-process-numerical-evidence", "route": route}
            ),
        ),
        (
            canonical_fingerprint(
                {"kind": "point-process-use-evidence", "trade_emission": False}
            ),
        ),
    )


def prepare_point_process(
    stream: PreparedMarketEventStream,
    market_definition: MarketPointProcessDefinition,
    hawkes_definition: HawkesDefinition,
    law: PhysicalLaw,
    /,
) -> PointProcessPlan:
    """Map admissible market clocks/channels into a generic event-time observation."""

    if not isinstance(stream, PreparedMarketEventStream):
        raise TypeError("stream must be a PreparedMarketEventStream.")
    if stream.overflow:
        raise ValueError("overflowed market streams cannot define a complete likelihood.")
    if not isinstance(market_definition, MarketPointProcessDefinition):
        raise TypeError("market_definition must be a MarketPointProcessDefinition.")
    if not isinstance(hawkes_definition, HawkesDefinition):
        raise TypeError("hawkes_definition must be a HawkesDefinition.")
    if not isinstance(law, PhysicalLaw):
        raise TypeError("market point-process inference requires a PhysicalLaw.")
    if hawkes_definition.baseline.shape[0] != len(stream.quote_key_ids):
        raise ValueError("Hawkes channel count must match market stream channels.")
    active = np.asarray(stream.valid_mask, dtype=bool)
    admissible = (
        active
        & (np.asarray(stream.event_times_ns) >= market_definition.start_time_ns)
        & (np.asarray(stream.event_times_ns) < market_definition.end_time_ns)
    )
    indices = np.flatnonzero(admissible)
    order = sorted(
        indices.tolist(),
        key=lambda index: (
            int(np.asarray(stream.event_times_ns)[index]),
            int(np.asarray(stream.available_times_ns)[index]),
            stream.observation_ids[index],
        ),
    )
    capacity = stream.capacity
    scale = 1.0e9
    times = np.zeros((capacity,), dtype=float)
    channels = np.zeros((capacity,), dtype=np.int32)
    valid = np.zeros((capacity,), dtype=bool)
    for output, index in enumerate(order):
        times[output] = (
            int(np.asarray(stream.event_times_ns)[index])
            - market_definition.start_time_ns
        ) / scale
        channels[output] = int(np.asarray(stream.channels)[index])
        valid[output] = True
    observation = PointProcessObservation(
        times,
        channels,
        valid,
        start_time=0.0,
        end_time=(market_definition.end_time_ns - market_definition.start_time_ns)
        / scale,
        channel_count=len(stream.quote_key_ids),
    )
    initial = ExponentialHawkesProcess(
        hawkes_definition.baseline,
        hawkes_definition.excitation,
        hawkes_definition.decay,
    )
    likelihood_plan = prepare_hawkes_likelihood(
        observation,
        initial,
        tie_policy=market_definition.tie_policy,
        require_stable=market_definition.require_stable,
    )
    plan_id = canonical_fingerprint(
        {
            "kind": "finance-point-process-plan",
            "stream": stream.stream_id,
            "market_definition": market_definition.definition_id,
            "hawkes_definition": hawkes_definition.definition_id,
            "law": law.law_id,
            "events": order,
        }
    )
    return PointProcessPlan(
        observation=observation,
        likelihood_plan=likelihood_plan,
        initial_process=initial,
        stream_id=stream.stream_id,
        market_definition_id=market_definition.definition_id,
        hawkes_definition_id=hawkes_definition.definition_id,
        law_id=law.law_id,
        plan_id=plan_id,
    )


def _softplus_inverse(value: Array) -> Array:
    return jnp.log(jnp.expm1(jnp.maximum(value, jnp.finfo(value.dtype).eps)))


def _decode_parameters(raw: Array, channels: int) -> ExponentialHawkesProcess:
    baseline_raw = raw[:channels]
    excitation_raw = raw[channels : channels + channels * channels].reshape(
        (channels, channels)
    )
    decay_raw = raw[channels + channels * channels :].reshape((channels, channels))
    baseline = jax.nn.softplus(baseline_raw) + jnp.finfo(raw.dtype).tiny
    decay = jax.nn.softplus(decay_raw) + jnp.finfo(raw.dtype).eps
    branching = jax.nn.softplus(excitation_raw)
    radius = jnp.max(jnp.abs(jnp.linalg.eigvals(branching))).real
    branching = branching * jnp.minimum(
        0.999 / jnp.maximum(radius, jnp.finfo(raw.dtype).eps), 1.0
    )
    return ExponentialHawkesProcess(baseline, branching * decay, decay)


def fit_hawkes(
    plan: PointProcessPlan,
    definition: HawkesDefinition,
    law: PhysicalLaw,
    /,
) -> PointProcessFit:
    """Fit stable exponential-Hawkes parameters by exact compensator QMLE."""

    if not isinstance(plan, PointProcessPlan):
        raise TypeError("plan must be a PointProcessPlan.")
    if not isinstance(definition, HawkesDefinition):
        raise TypeError("definition must be a HawkesDefinition.")
    if not isinstance(law, PhysicalLaw) or law.law_id != plan.law_id:
        raise ValueError("fit law must match the prepared point-process plan.")
    if definition.definition_id != plan.hawkes_definition_id:
        raise ValueError("Hawkes definition does not match the prepared plan.")
    channels = plan.initial_process.channel_count
    initial_branching = definition.excitation / definition.decay
    raw = jnp.concatenate(
        (
            _softplus_inverse(definition.baseline),
            _softplus_inverse(initial_branching).reshape(-1),
            _softplus_inverse(definition.decay).reshape(-1),
        )
    )
    initial_likelihood = evaluate_hawkes_likelihood(
        plan.observation,
        plan.initial_process,
        plan=plan.likelihood_plan,
    )

    def objective(parameters, _):
        process = _decode_parameters(parameters, channels)
        result = evaluate_hawkes_likelihood(
            plan.observation,
            process,
            plan=plan.likelihood_plan,
        )
        return -result.log_likelihood

    optimization = minimize(
        objective,
        raw,
        method=NewtonTrustRegion(),
        termination=OptimizationTermination(maximum_steps=definition.maximum_steps),
    )
    process = _decode_parameters(optimization.parameters, channels)
    likelihood = evaluate_hawkes_likelihood(
        plan.observation,
        process,
        plan=plan.likelihood_plan,
    )
    return PointProcessFit(
        process=process,
        likelihood=likelihood,
        initial_likelihood=initial_likelihood,
        optimizer_status=optimization.status,
        optimizer_iterations=optimization.diagnostics.iterations,
        gradient_norm=optimization.diagnostics.final_optimality_norm,
        evidence=_evidence(plan.stream_id, definition.definition_id, "exact-hawkes-qmle"),
        plan_id=plan.plan_id,
        law_id=law.law_id,
    )


def score_market_point_process(fit: PointProcessFit, /) -> PointProcessResult:
    if not isinstance(fit, PointProcessFit):
        raise TypeError("fit must be a PointProcessFit.")
    likelihood = fit.likelihood
    return PointProcessResult(
        log_likelihood=likelihood.log_likelihood,
        compensator=likelihood.compensator,
        event_intensity=likelihood.event_intensity,
        spectral_radius=likelihood.spectral_radius,
        stable=likelihood.stable,
        status=likelihood.status,
        fit=fit,
    )


__all__ = [
    "HawkesDefinition",
    "MarketPointProcessDefinition",
    "PointProcessFit",
    "PointProcessPlan",
    "PointProcessResult",
    "fit_hawkes",
    "prepare_point_process",
    "score_market_point_process",
]
