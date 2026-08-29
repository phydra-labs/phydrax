#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._strict import StrictModule


class CombinatorialStatus(IntEnum):
    """Portable status for one linear combinatorial solve."""

    OPTIMAL = 0
    FEASIBLE = 1
    INFEASIBLE = 2
    UNBOUNDED = 3
    MAXIMUM_STEPS_REACHED = 4
    NONFINITE_INPUT = 5
    NUMERICAL_FAILURE = 6
    CERTIFICATION_FAILED = 7


_STATUS_MESSAGES = {
    CombinatorialStatus.OPTIMAL: "optimal solution certified",
    CombinatorialStatus.FEASIBLE: "feasible solution without an optimality certificate",
    CombinatorialStatus.INFEASIBLE: "the declared feasible set is empty",
    CombinatorialStatus.UNBOUNDED: "the objective is unbounded below",
    CombinatorialStatus.MAXIMUM_STEPS_REACHED: "maximum combinatorial steps reached",
    CombinatorialStatus.NONFINITE_INPUT: "objective costs contain non-finite values",
    CombinatorialStatus.NUMERICAL_FAILURE: "native combinatorial computation failed",
    CombinatorialStatus.CERTIFICATION_FAILED: "independent solution certificate failed",
}


def combinatorial_status_message(status: int | CombinatorialStatus, /) -> str:
    """Return the stable message for one combinatorial status."""

    return _STATUS_MESSAGES[CombinatorialStatus(int(status))]


class CombinatorialMethodCapabilities(StrictModule):
    """Static behavior guaranteed by one combinatorial method."""

    exact: bool = eqx.field(static=True)
    jax_native: bool = eqx.field(static=True)
    jit: bool = eqx.field(static=True)
    batched: bool = eqx.field(static=True)
    signed_costs: bool = eqx.field(static=True)
    deterministic_ties: bool = eqx.field(static=True)
    optimality_certificate: bool = eqx.field(static=True)
    prepared_refresh: bool = eqx.field(static=True)
    warm_start: bool = eqx.field(static=True)
    surrogate_pullback: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        exact: bool,
        jax_native: bool,
        jit: bool,
        batched: bool,
        signed_costs: bool,
        deterministic_ties: bool,
        optimality_certificate: bool,
        prepared_refresh: bool = False,
        warm_start: bool = False,
        surrogate_pullback: bool = False,
    ):
        self.exact = bool(exact)
        self.jax_native = bool(jax_native)
        self.jit = bool(jit)
        self.batched = bool(batched)
        self.signed_costs = bool(signed_costs)
        self.deterministic_ties = bool(deterministic_ties)
        self.optimality_certificate = bool(optimality_certificate)
        self.prepared_refresh = bool(prepared_refresh)
        self.warm_start = bool(warm_start)
        self.surrogate_pullback = bool(surrogate_pullback)


class CombinatorialCertification(StrictModule):
    """Scale-aware tolerances for independent final certificates."""

    absolute: float = eqx.field(static=True)
    relative: float = eqx.field(static=True)

    def __init__(self, *, absolute: float = 1e-6, relative: float = 1e-6):
        absolute_ = float(absolute)
        relative_ = float(relative)
        if not isfinite(absolute_) or absolute_ < 0.0:
            raise ValueError(
                "absolute certification tolerance must be finite and non-negative."
            )
        if not isfinite(relative_) or relative_ < 0.0:
            raise ValueError(
                "relative certification tolerance must be finite and non-negative."
            )
        self.absolute = absolute_
        self.relative = relative_

    def threshold(self, *scales: Any) -> Array:
        """Return the tolerance at the largest declared objective scale."""

        scale = jnp.asarray(1.0)
        for value in scales:
            array = jnp.asarray(value)
            scale = jnp.maximum(scale.astype(array.dtype), jnp.abs(array))
        return self.absolute + self.relative * scale


class CombinatorialFeasibility(StrictModule):
    """Independent feasibility verdict and non-negative violation residual."""

    feasible: Array
    residual: Array


class CombinatorialCertificate(StrictModule):
    """Independent feasibility, objective, and optimality evidence."""

    finite: Array
    feasible: Array
    objective_consistent: Array
    optimality_proven: Array
    primal_residual: Array
    dual_residual: Array
    absolute_gap: Array
    relative_gap: Array
    tie_margin: Array
    dual_available: Array
    gap_available: Array
    tie_available: Array


class CombinatorialProvenance(StrictModule):
    """Static identity and guarantees of one native combinatorial execution."""

    problem_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    implementation: str = eqx.field(static=True)
    tie_policy: str = eqx.field(static=True)
    certificate_kind: str = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    signed_costs: bool = eqx.field(static=True)
    configuration: tuple[tuple[str, str], ...] = eqx.field(static=True)


class CombinatorialResult(StrictModule):
    """Discrete decision, objective features, certificate, and provenance."""

    decision: PyTree[Any]
    features: PyTree[Array]
    objective_value: Array
    status: Array
    valid: Array
    certificate: CombinatorialCertificate
    iterations: Array
    work: Array
    provenance: CombinatorialProvenance
    batch_shape: tuple[int, ...] = eqx.field(static=True)

    @property
    def success(self) -> Array:
        """Return per-instance certified optimality."""

        return (
            self.valid
            & (self.status == int(CombinatorialStatus.OPTIMAL))
            & self.certificate.optimality_proven
        )

    @property
    def optimal(self) -> Array:
        """Alias the certified-success contract."""

        return self.success

    @property
    def all_success(self) -> Array:
        """Return whether every batch member is certified optimal."""

        return jnp.all(self.success)


__all__ = [
    "CombinatorialCertificate",
    "CombinatorialCertification",
    "CombinatorialFeasibility",
    "CombinatorialMethodCapabilities",
    "CombinatorialProvenance",
    "CombinatorialResult",
    "CombinatorialStatus",
    "combinatorial_status_message",
]
