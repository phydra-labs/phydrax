#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact unreleased profiles and observations for the stable core portfolio."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .._fingerprint import canonical_fingerprint
from ._registry import CapabilityProfile, SupportTuple


_CORE_SPECIFICATIONS = (
    (
        "core.axes-units",
        {
            "axis_identity": "semantic-key-and-slot",
            "unit_conversion": "exact-rational-multiplicative",
            "precision": "float64",
        },
        ("identity", "algebra", "jit-derivative", "failure"),
    ),
    (
        "core.sparse-ein",
        {
            "sparse_routes": "masked-fixed-capacity",
            "contraction": "opt-einsum-jax",
            "precision": "float64",
        },
        ("forward", "transpose-adjoint", "masked-nonfinite", "jit-derivative"),
    ),
    (
        "core.linear-algebra",
        {
            "operator": "dense-explicit",
            "problem": "square-nonsingular",
            "method": "dense-lu",
            "precision": "float64",
        },
        ("original-residual", "backward-error", "adjoint", "derivative", "failure"),
    ),
    (
        "core.nonlinear",
        {
            "problem": "smooth-square-root",
            "method": "newton-krylov",
            "precision": "float64",
        },
        ("physical-residual", "globalization", "implicit-derivative", "failure"),
    ),
    (
        "core.optimization",
        {
            "problem": "smooth-unconstrained-quadratic",
            "method": "newton-krylov",
            "precision": "float64",
        },
        ("stationarity", "objective", "implicit-derivative", "failure"),
    ),
    (
        "core.integration",
        {
            "domain": "interval",
            "method": "gauss-legendre",
            "precision": "float64",
        },
        ("polynomial-exactness", "measure", "derivative", "failure"),
    ),
    (
        "core.time-integration",
        {
            "problem": "separable-harmonic-oscillator",
            "method": "stormer-verlet-fixed-step",
            "precision": "float64",
        },
        ("observed-order", "terminal-error", "replay", "failure"),
    ),
    (
        "core.discretization",
        {
            "grid": "one-dimensional-uniform-cells",
            "method": "finite-volume",
            "precision": "float64",
        },
        ("topology", "measure", "conservation", "failure"),
    ),
    (
        "core.uncertainty-quantification",
        {
            "diagnostic": "effective-sample-size",
            "distribution": "uniform-log-weights",
            "precision": "float64",
        },
        ("normalization", "effective-sample-size", "finite-diagnostics", "failure"),
    ),
    (
        "core.execution-lifecycle",
        {
            "execution": "transactional-local",
            "checkpoint": "content-addressed",
            "precision": "float64",
        },
        ("atomic-acceptance", "rollback", "checkpoint-roundtrip", "identity"),
    ),
)


def core_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Return exact evidence-free profiles for the first stable-core campaign."""

    return tuple(
        CapabilityProfile(
            f"{capability}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(capability, attributes),),
            required_gates=gates,
        )
        for capability, attributes, gates in _CORE_SPECIFICATIONS
    )


@dataclass(frozen=True, slots=True)
class CoreQualificationObservation:
    """One deterministic observation over every declared gate of one profile."""

    profile_id: str
    gate_results: tuple[tuple[str, bool], ...]
    metrics: tuple[tuple[str, float], ...]
    observation_id: str

    def __init__(
        self,
        profile: CapabilityProfile,
        gate_results: Mapping[str, bool],
        metrics: Mapping[str, float],
        /,
    ):
        if not isinstance(profile, CapabilityProfile):
            raise TypeError("profile must be CapabilityProfile.")
        gates = tuple(
            sorted((str(name), bool(value)) for name, value in gate_results.items())
        )
        if tuple(name for name, _ in gates) != tuple(sorted(profile.required_gates)):
            raise ValueError(
                "Core observations must report every required gate exactly once."
            )
        metrics_ = tuple(
            sorted((str(name), float(value)) for name, value in metrics.items())
        )
        if not metrics_ or any(
            not name or not np.isfinite(value) for name, value in metrics_
        ):
            raise ValueError("Core qualification metrics must be named and finite.")
        record = {
            "kind": "core-qualification-observation",
            "profile_id": profile.profile_id,
            "gate_results": dict(gates),
            "metrics": dict(metrics_),
        }
        object.__setattr__(self, "profile_id", profile.profile_id)
        object.__setattr__(self, "gate_results", gates)
        object.__setattr__(self, "metrics", metrics_)
        object.__setattr__(self, "observation_id", canonical_fingerprint(record))

    @property
    def passed(self) -> bool:
        return all(value for _, value in self.gate_results)

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "core-qualification-observation",
            "profile_id": self.profile_id,
            "gate_results": dict(self.gate_results),
            "metrics": dict(self.metrics),
            "passed": self.passed,
            "observation_id": self.observation_id,
        }


def core_portfolio_observation(
    observations: Sequence[CoreQualificationObservation],
    /,
) -> dict[str, object]:
    """Aggregate each exact core profile once without creating release evidence."""

    values = tuple(observations)
    expected = {profile.profile_id for profile in core_candidate_profiles()}
    observed = {value.profile_id for value in values}
    if len(values) != len(observed) or observed != expected:
        raise ValueError(
            "Core portfolio observations must cover every exact profile once."
        )
    ordered = tuple(sorted(values, key=lambda value: value.profile_id))
    payload: dict[str, object] = {
        "kind": "core-qualification-portfolio",
        "release_claim": False,
        "observations": [value.to_record() for value in ordered],
        "passed": all(value.passed for value in ordered),
    }
    return {**payload, "portfolio_id": canonical_fingerprint(payload)}


__all__ = [
    "CoreQualificationObservation",
    "core_candidate_profiles",
    "core_portfolio_observation",
]
