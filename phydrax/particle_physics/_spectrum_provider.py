#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pinned host-only external spectrum execution with strict SLHA output parsing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from .._fingerprint import canonical_fingerprint
from ..interchange.energy_runtime import (
    EnergyRunResult,
    EnergyRuntimeError,
    PinnedExecutable,
    run_energy_command,
)
from ._capabilities import HEPProviderBinding
from ._spectrum import (
    SpectrumApproximationProfile,
    SpectrumCalculationResult,
    SpectrumDiagnostics,
    SpectrumStatus,
)


if TYPE_CHECKING:
    from ..interchange.hep._slha import SLHADocument


@dataclass(frozen=True, slots=True)
class ExternalSpectrumProvider:
    executable: PinnedExecutable
    capability: HEPProviderBinding

    def __post_init__(self) -> None:
        if not isinstance(self.executable, PinnedExecutable):
            raise TypeError("executable must be PinnedExecutable.")
        if not isinstance(self.capability, HEPProviderBinding):
            raise TypeError("capability must be HEPProviderBinding.")
        if self.executable.version != self.capability.provider_release:
            raise ValueError("Spectrum executable and capability releases must match.")
        if not self.capability.supports("particle-spectrum.external"):
            raise ValueError("HEP capability does not admit external particle spectra.")


@dataclass(frozen=True, slots=True)
class SpectrumProviderPlan:
    input_filename: str
    output_filename: str
    arguments: tuple[str, ...]
    timeout_seconds: float
    maximum_output_bytes: int
    environment: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.input_filename
            or not self.output_filename
            or self.input_filename == self.output_filename
            or any(
                "\x00" in value for value in (self.input_filename, self.output_filename)
            )
        ):
            raise ValueError("Spectrum provider filenames are invalid.")
        if not np.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0:
            raise ValueError("Spectrum provider timeout must be positive and finite.")
        if self.maximum_output_bytes < 1:
            raise ValueError("Spectrum provider output limit must be positive.")
        if len({key for key, _ in self.environment}) != len(self.environment):
            raise ValueError("Spectrum provider environment keys must be unique.")


SpectrumProviderStatus = Literal[
    "complete",
    "provider-failed",
    "invalid-slha",
]


@dataclass(frozen=True, slots=True)
class SpectrumProviderExecution:
    status: SpectrumProviderStatus
    run: EnergyRunResult | None
    document: SLHADocument | None
    provider_id: str
    input_artifact_id: str
    output_artifact_id: str
    error: str
    execution_id: str


def _execution(
    status: SpectrumProviderStatus,
    provider: ExternalSpectrumProvider,
    input_artifact_id: str,
    run: EnergyRunResult | None,
    document: SLHADocument | None,
    error: str,
    /,
) -> SpectrumProviderExecution:
    output_id = "none" if document is None else document.source_id
    execution_id = canonical_fingerprint(
        {
            "kind": "external-spectrum-provider-execution",
            "status": status,
            "provider": provider.capability.binding_id,
            "executable": provider.executable.sha256,
            "input": input_artifact_id,
            "output": output_id,
            "run_artifact": None if run is None else run.artifact.artifact_id,
            "error": error,
        }
    )
    return SpectrumProviderExecution(
        status=status,
        run=run,
        document=document,
        provider_id=provider.capability.binding_id,
        input_artifact_id=input_artifact_id,
        output_artifact_id=output_id,
        error=error,
        execution_id=execution_id,
    )


def execute_spectrum_provider(
    provider: ExternalSpectrumProvider,
    plan: SpectrumProviderPlan,
    input_slha: bytes,
    /,
) -> SpectrumProviderExecution:
    if not isinstance(provider, ExternalSpectrumProvider):
        raise TypeError("provider must be ExternalSpectrumProvider.")
    if not isinstance(plan, SpectrumProviderPlan):
        raise TypeError("plan must be SpectrumProviderPlan.")
    if not isinstance(input_slha, bytes):
        raise TypeError("input_slha must be exact bytes.")
    from ..interchange.hep._slha import parse_slha

    input_document = parse_slha(input_slha)
    arguments = tuple(
        value.replace("{input}", plan.input_filename).replace(
            "{output}", plan.output_filename
        )
        for value in plan.arguments
    )
    run: EnergyRunResult | None = None
    try:
        run = run_energy_command(
            provider.executable,
            arguments,
            inputs={plan.input_filename: input_slha},
            outputs=(plan.output_filename,),
            timeout=plan.timeout_seconds,
            max_output_bytes=plan.maximum_output_bytes,
            environment=dict(plan.environment),
        )
    except EnergyRuntimeError as failure:
        return _execution(
            "provider-failed",
            provider,
            input_document.source_id,
            failure.result,
            None,
            str(failure),
        )
    output = run.output(plan.output_filename)
    try:
        document = parse_slha(output)
    except (UnicodeDecodeError, ValueError) as failure:
        return _execution(
            "invalid-slha",
            provider,
            input_document.source_id,
            run,
            None,
            str(failure),
        )
    return _execution(
        "complete",
        provider,
        input_document.source_id,
        run,
        document,
        "",
    )


def spectrum_result_from_provider(
    execution: SpectrumProviderExecution,
    approximation: SpectrumApproximationProfile,
    running_scales: ArrayLike,
    running_parameters: ArrayLike,
    /,
    *,
    residual_norm: ArrayLike = 0.0,
    electroweak_minimum: bool = True,
    perturbative: bool = True,
    running_tachyons: bool = False,
    pole_tachyons: bool = False,
    warning_ids: Sequence[str] = (),
) -> SpectrumCalculationResult:
    if not isinstance(execution, SpectrumProviderExecution):
        raise TypeError("execution must be SpectrumProviderExecution.")
    if not isinstance(approximation, SpectrumApproximationProfile):
        raise TypeError("approximation must be SpectrumApproximationProfile.")
    if execution.document is None or execution.status != "complete":
        status = (
            SpectrumStatus.PROVIDER_FAILED
            if execution.status == "provider-failed"
            else SpectrumStatus.INVALID_INPUT
        )
        raise ValueError(
            f"Cannot construct a spectrum result from provider status {status.name}."
        )
    from ..interchange.hep._slha import spectrum_observables_from_slha

    observables = spectrum_observables_from_slha(execution.document)
    finite = bool(jnp.all(jnp.isfinite(observables.values)))
    warning_values = tuple(warning_ids)
    diagnostics = SpectrumDiagnostics(
        SpectrumStatus.SUCCESS if finite else SpectrumStatus.NONFINITE_OUTPUT,
        provider_available=True,
        finite=finite,
        root_found=True,
        electroweak_minimum=electroweak_minimum,
        perturbative=perturbative,
        running_tachyons=running_tachyons,
        pole_tachyons=pole_tachyons,
        approximation_warning=bool(warning_values),
        residual_norm=residual_norm,
        warning_ids=warning_values,
    )
    return SpectrumCalculationResult(
        observables,
        running_scales,
        running_parameters,
        diagnostics,
        approximation,
        provider_id=execution.provider_id,
        input_artifact_id=execution.input_artifact_id,
        output_artifact_id=execution.output_artifact_id,
    )


__all__ = [
    "ExternalSpectrumProvider",
    "SpectrumProviderExecution",
    "SpectrumProviderPlan",
    "SpectrumProviderStatus",
    "execute_spectrum_provider",
    "spectrum_result_from_provider",
]
