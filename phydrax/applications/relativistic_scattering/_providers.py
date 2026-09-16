#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..._external_runtime import (
    EnergyRunResult,
    PinnedExecutable,
    run_energy_command,
)
from ...particle_physics import HEPProviderBinding


@dataclass(frozen=True, slots=True)
class ExternalHEPProvider:
    """Host-only composition of a pinned executable and admitted HEP capabilities."""

    executable: PinnedExecutable
    capability: HEPProviderBinding

    def __post_init__(self) -> None:
        if not isinstance(self.executable, PinnedExecutable):
            raise TypeError("executable must be PinnedExecutable.")
        if not isinstance(self.capability, HEPProviderBinding):
            raise TypeError("capability must be HEPProviderBinding.")
        if self.executable.version != self.capability.provider_release:
            raise ValueError("Executable and capability provider releases must match.")


def run_external_hep_provider(
    provider: ExternalHEPProvider,
    args: Sequence[str],
    /,
    *,
    required_capability: str,
    inputs: Mapping[str, bytes],
    outputs: Sequence[str],
    stdin: bytes = b"",
    timeout: float = 120.0,
    max_output_bytes: int = 1 << 30,
    environment: Mapping[str, str] | None = None,
) -> EnergyRunResult:
    """Run one admitted external capability in the existing bounded host runtime."""
    if not isinstance(provider, ExternalHEPProvider):
        raise TypeError("provider must be ExternalHEPProvider.")
    capability = str(required_capability).strip()
    if not capability or not provider.capability.supports(capability):
        raise ValueError("The provider does not advertise the required capability.")
    return run_energy_command(
        provider.executable,
        args,
        inputs=inputs,
        outputs=outputs,
        stdin=stdin,
        timeout=timeout,
        max_output_bytes=max_output_bytes,
        environment=environment,
    )


__all__ = ["ExternalHEPProvider", "run_external_hep_provider"]
