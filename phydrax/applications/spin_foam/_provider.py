#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pinned process boundary for convention-matched finite EPRL references."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal

from ..._fingerprint import canonical_fingerprint
from ...interchange.energy_runtime import (
    EnergyRunResult,
    EnergyRuntimeError,
    PinnedExecutable,
    run_energy_command,
)
from ._eprl import EPRLVertexPlan


@dataclass(frozen=True, slots=True)
class ExternalSpinFoamProvider:
    executable: PinnedExecutable
    protocol_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.executable, PinnedExecutable):
            raise TypeError("executable must be PinnedExecutable.")
        if not self.protocol_id.strip():
            raise ValueError("protocol_id must be non-empty.")


SpinFoamProviderStatus = Literal[
    "complete",
    "provider-failed",
    "invalid-output",
    "semantic-mismatch",
]


@dataclass(frozen=True, slots=True)
class SpinFoamProviderResult:
    status: SpinFoamProviderStatus
    amplitude_real: str
    amplitude_imaginary: str
    absolute_error: str
    run: EnergyRunResult | None
    plan_id: str
    provider_id: str
    result_id: str
    error: str = ""
    claim: str = "external-finite-cutoff-eprl-reference-not-quantum-gravity-validation"

    @property
    def amplitude(self) -> complex:
        return complex(
            float(Decimal(self.amplitude_real)), float(Decimal(self.amplitude_imaginary))
        )


def _input_record(plan: EPRLVertexPlan, /) -> dict[str, object]:
    return {
        "plan_id": plan.plan_id,
        "boundary_twice_spins": list(plan.boundary_twice_spins),
        "boundary_twice_intertwiners": list(plan.boundary_twice_intertwiners),
        "immirzi_parameter": repr(plan.immirzi_parameter),
        "delta_l": plan.delta_l,
        "face_amplitude": plan.face_amplitude,
        "edge_amplitude": plan.edge_amplitude,
        "coherent_phase_convention": plan.coherent_phase_convention,
        "normal_frame_id": plan.normal_frame_id,
        "quadrature_id": plan.quadrature_id,
        "precision_bits": plan.precision_bits,
    }


def _result(
    status: SpinFoamProviderStatus,
    plan: EPRLVertexPlan,
    provider: ExternalSpinFoamProvider,
    run: EnergyRunResult | None,
    /,
    *,
    amplitude_real: str = "0",
    amplitude_imaginary: str = "0",
    absolute_error: str = "Infinity",
    error: str = "",
) -> SpinFoamProviderResult:
    identifier = canonical_fingerprint(
        {
            "kind": "external-spin-foam-provider-result",
            "status": status,
            "plan": plan.plan_id,
            "provider": provider.executable.sha256,
            "protocol": provider.protocol_id,
            "run": None if run is None else run.artifact.artifact_id,
            "amplitude": (amplitude_real, amplitude_imaginary),
            "absolute_error": absolute_error,
            "error": error,
        }
    )
    return SpinFoamProviderResult(
        status,
        amplitude_real,
        amplitude_imaginary,
        absolute_error,
        run,
        plan.plan_id,
        canonical_fingerprint(
            {
                "executable": provider.executable.sha256,
                "protocol": provider.protocol_id,
            }
        ),
        identifier,
        error,
    )


def execute_spin_foam_provider(
    provider: ExternalSpinFoamProvider,
    plan: EPRLVertexPlan,
    /,
    *,
    timeout: float = 3600.0,
    maximum_output_bytes: int = 64 * 1024 * 1024,
) -> SpinFoamProviderResult:
    if not isinstance(provider, ExternalSpinFoamProvider):
        raise TypeError("provider must be ExternalSpinFoamProvider.")
    if not isinstance(plan, EPRLVertexPlan):
        raise TypeError("plan must be EPRLVertexPlan.")
    payload = (json.dumps(_input_record(plan), sort_keys=True) + "\n").encode("ascii")
    try:
        run = run_energy_command(
            provider.executable,
            ("--input=eprl-input.json", "--output=eprl-output.json"),
            inputs={"eprl-input.json": payload},
            outputs=("eprl-output.json",),
            timeout=timeout,
            max_output_bytes=maximum_output_bytes,
        )
    except EnergyRuntimeError as failure:
        return _result(
            "provider-failed",
            plan,
            provider,
            failure.result,
            error=str(failure),
        )
    try:
        record = json.loads(run.output("eprl-output.json"))
    except (UnicodeDecodeError, json.JSONDecodeError) as failure:
        return _result("invalid-output", plan, provider, run, error=str(failure))
    required = {
        "plan_id",
        "delta_l",
        "precision_bits",
        "amplitude_real",
        "amplitude_imaginary",
        "absolute_error",
    }
    if not isinstance(record, dict) or set(record) != required:
        return _result(
            "invalid-output",
            plan,
            provider,
            run,
            error="Spin-foam output fields do not match the protocol.",
        )
    if (
        record["plan_id"] != plan.plan_id
        or int(record["delta_l"]) != plan.delta_l
        or int(record["precision_bits"]) != plan.precision_bits
    ):
        return _result(
            "semantic-mismatch",
            plan,
            provider,
            run,
            error="Spin-foam output convention/cutoff identity differs.",
        )
    decimals = tuple(
        str(record[key])
        for key in ("amplitude_real", "amplitude_imaginary", "absolute_error")
    )
    try:
        parsed = tuple(Decimal(value) for value in decimals)
    except InvalidOperation as failure:
        return _result("invalid-output", plan, provider, run, error=str(failure))
    if any(not value.is_finite() for value in parsed) or parsed[2] < 0:
        return _result(
            "invalid-output",
            plan,
            provider,
            run,
            error="Spin-foam decimals must be finite with nonnegative error.",
        )
    return _result(
        "complete",
        plan,
        provider,
        run,
        amplitude_real=decimals[0],
        amplitude_imaginary=decimals[1],
        absolute_error=decimals[2],
    )


__all__ = [
    "ExternalSpinFoamProvider",
    "SpinFoamProviderResult",
    "SpinFoamProviderStatus",
    "execute_spin_foam_provider",
]
