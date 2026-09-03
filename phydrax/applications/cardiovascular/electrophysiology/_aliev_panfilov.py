#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Dimensional-time Aliev--Panfilov phenomenological reaction dynamics.

The two state coordinates are phenomenological activation and recovery variables.
They are not membrane voltage, ionic concentrations, or physical current density.
Time is measured in milliseconds, so every returned rate is per millisecond.
"""

from __future__ import annotations

from enum import IntFlag
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class AlievPanfilovStatus(IntFlag):
    """Fail-closed status for one phenomenological reaction evaluation."""

    SUCCESS = 0
    NONFINITE_STATE = 1
    RECOVERY_DENOMINATOR_SINGULAR = 2
    NONFINITE_SOURCE = 4
    NONFINITE_RATE = 8


class AlievPanfilovParameters(StrictModule, NonTrainableState):
    """Exact seven-coefficient Aliev--Panfilov convention with ``tau`` in ms.

    The implemented equations are

    ``du/dt = (k*u*(u-a)*(1-u) - u*r) / tau + s`` and
    ``dr/dt = (epsilon0 + mu1*r/(u+mu2)) *
              (-r - k*u*(u-b-1)) / tau``.

    ``u`` and ``r`` are dimensionless phenomenological coordinates and ``s`` is
    an externally supplied activation rate in ``ms^-1``.  The parameter identity
    includes every coefficient and the pinned equation convention.
    """

    a: float = eqx.field(static=True)
    b: float = eqx.field(static=True)
    k: float = eqx.field(static=True)
    epsilon0: float = eqx.field(static=True)
    mu1: float = eqx.field(static=True)
    mu2: float = eqx.field(static=True)
    tau: float = eqx.field(static=True)
    singularity_tolerance: float = eqx.field(static=True)
    parameter_id: str = eqx.field(static=True)

    def __init__(
        self,
        a: float,
        b: float,
        k: float,
        epsilon0: float,
        mu1: float,
        mu2: float,
        tau: float,
        /,
        *,
        singularity_tolerance: float = 1.0e-7,
    ):
        coefficients = {
            "a": float(a),
            "b": float(b),
            "k": float(k),
            "epsilon0": float(epsilon0),
            "mu1": float(mu1),
            "mu2": float(mu2),
            "tau_ms": float(tau),
        }
        if any(not isfinite(value) for value in coefficients.values()):
            raise ValueError("All Aliev--Panfilov coefficients must be finite.")
        if coefficients["a"] < 0.0 or coefficients["b"] < 0.0:
            raise ValueError("Aliev--Panfilov a and b must be nonnegative.")
        if coefficients["k"] <= 0.0:
            raise ValueError("Aliev--Panfilov k must be positive.")
        if coefficients["epsilon0"] < 0.0 or coefficients["mu1"] < 0.0:
            raise ValueError("Aliev--Panfilov epsilon0 and mu1 must be nonnegative.")
        if coefficients["mu2"] <= 0.0 or coefficients["tau_ms"] <= 0.0:
            raise ValueError("Aliev--Panfilov mu2 and tau must be positive.")
        tolerance = float(singularity_tolerance)
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("singularity_tolerance must be finite and positive.")
        self.a = coefficients["a"]
        self.b = coefficients["b"]
        self.k = coefficients["k"]
        self.epsilon0 = coefficients["epsilon0"]
        self.mu1 = coefficients["mu1"]
        self.mu2 = coefficients["mu2"]
        self.tau = coefficients["tau_ms"]
        self.singularity_tolerance = tolerance
        self.parameter_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-aliev-panfilov-dimensional-time",
                "equations": (
                    "du_dt=(k*u*(u-a)*(1-u)-u*r)/tau_ms+source_per_ms;"
                    "dr_dt=(epsilon0+mu1*r/(u+mu2))*"
                    "(-r-k*u*(u-b-1))/tau_ms"
                ),
                "coefficients": coefficients,
                "singularity_tolerance": tolerance,
                "state_units": ["1", "1"],
                "time_unit": "ms",
            }
        )


class AlievPanfilovState(StrictModule):
    """Fixed-shape phenomenological activation and recovery coordinates."""

    activation: Array
    recovery: Array

    def __init__(self, activation: ArrayLike, recovery: ArrayLike, /):
        activation_ = jnp.asarray(activation)
        recovery_ = jnp.asarray(recovery, dtype=activation_.dtype)
        if activation_.shape != recovery_.shape or activation_.size == 0:
            raise ValueError(
                "activation and recovery must have identical nonempty shapes."
            )
        if not jnp.issubdtype(activation_.dtype, jnp.inexact):
            activation_ = activation_.astype(float)
            recovery_ = recovery_.astype(activation_.dtype)
        self.activation = activation_
        self.recovery = recovery_


class AlievPanfilovRates(StrictModule):
    """Candidate activation and recovery rates in ``ms^-1``."""

    activation_per_ms: Array
    recovery_per_ms: Array


class AlievPanfilovEvidence(StrictModule):
    """Finiteness and recovery-denominator singularity evidence."""

    minimum_abs_recovery_denominator: Array
    singular_count: Array
    state_finite: Array
    source_finite: Array
    rates_finite: Array
    status: Array
    successful: Array


class AlievPanfilovCandidate(StrictModule):
    """Uncommitted dimensional rates and their fail-closed evidence."""

    rates: AlievPanfilovRates
    evidence: AlievPanfilovEvidence


def evaluate_aliev_panfilov(
    parameters: AlievPanfilovParameters,
    state: AlievPanfilovState,
    /,
    *,
    activation_source_per_ms: ArrayLike = 0.0,
) -> AlievPanfilovCandidate:
    """Evaluate exact model rates without assigning physical ionic meaning."""

    if not isinstance(parameters, AlievPanfilovParameters):
        raise TypeError("parameters must be AlievPanfilovParameters.")
    if not isinstance(state, AlievPanfilovState):
        raise TypeError("state must be AlievPanfilovState.")
    source = jnp.asarray(activation_source_per_ms, dtype=state.activation.dtype)
    source = jnp.broadcast_to(source, state.activation.shape)
    denominator = state.activation + parameters.mu2
    denominator_abs = jnp.abs(denominator)
    nonsingular = denominator_abs > parameters.singularity_tolerance
    safe_denominator = jnp.where(nonsingular, denominator, jnp.ones_like(denominator))
    activation_rate = (
        parameters.k
        * state.activation
        * (state.activation - parameters.a)
        * (1.0 - state.activation)
        - state.activation * state.recovery
    ) / parameters.tau + source
    recovery_rate = (
        (parameters.epsilon0 + parameters.mu1 * state.recovery / safe_denominator)
        * (
            -state.recovery
            - parameters.k * state.activation * (state.activation - parameters.b - 1.0)
        )
        / parameters.tau
    )

    state_finite = jnp.all(jnp.isfinite(state.activation)) & jnp.all(
        jnp.isfinite(state.recovery)
    )
    source_finite = jnp.all(jnp.isfinite(source))
    rates_finite = jnp.all(jnp.isfinite(activation_rate)) & jnp.all(
        jnp.isfinite(recovery_rate)
    )
    singular_count = jnp.sum(~nonsingular, dtype=jnp.int32)
    minimum_denominator = jnp.min(denominator_abs)
    status = jnp.asarray(int(AlievPanfilovStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        state_finite,
        status,
        jnp.bitwise_or(status, int(AlievPanfilovStatus.NONFINITE_STATE)),
    )
    status = jnp.where(
        singular_count == 0,
        status,
        jnp.bitwise_or(status, int(AlievPanfilovStatus.RECOVERY_DENOMINATOR_SINGULAR)),
    )
    status = jnp.where(
        source_finite,
        status,
        jnp.bitwise_or(status, int(AlievPanfilovStatus.NONFINITE_SOURCE)),
    )
    status = jnp.where(
        rates_finite,
        status,
        jnp.bitwise_or(status, int(AlievPanfilovStatus.NONFINITE_RATE)),
    )
    successful = status == int(AlievPanfilovStatus.SUCCESS)
    safe_activation_rate = jnp.where(
        successful, activation_rate, jnp.zeros_like(activation_rate)
    )
    safe_recovery_rate = jnp.where(
        successful, recovery_rate, jnp.zeros_like(recovery_rate)
    )
    return AlievPanfilovCandidate(
        AlievPanfilovRates(safe_activation_rate, safe_recovery_rate),
        AlievPanfilovEvidence(
            minimum_denominator,
            singular_count,
            state_finite,
            source_finite,
            rates_finite,
            status,
            successful,
        ),
    )


def commit_aliev_panfilov_rates(
    candidate: AlievPanfilovCandidate,
    accepted_rates: AlievPanfilovRates,
    /,
) -> AlievPanfilovRates:
    """Commit valid rates or preserve the caller's previously accepted rates."""

    if not isinstance(candidate, AlievPanfilovCandidate):
        raise TypeError("candidate must be AlievPanfilovCandidate.")
    if not isinstance(accepted_rates, AlievPanfilovRates):
        raise TypeError("accepted_rates must be AlievPanfilovRates.")
    return jax.lax.cond(
        candidate.evidence.successful,
        lambda _: candidate.rates,
        lambda _: accepted_rates,
        operand=None,
    )


__all__ = [
    "AlievPanfilovCandidate",
    "AlievPanfilovEvidence",
    "AlievPanfilovParameters",
    "AlievPanfilovRates",
    "AlievPanfilovState",
    "AlievPanfilovStatus",
    "commit_aliev_panfilov_rates",
    "evaluate_aliev_panfilov",
]
