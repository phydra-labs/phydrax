#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Owner-local qualification declaration for distributed spectral execution."""

from __future__ import annotations

from ...qualification._catalog import CapabilityDeclaration, CapabilityDisposition
from ...qualification._registry import CapabilityProfile, SupportTuple, SupportValue


DISTRIBUTED_SPECTRAL_CAPABILITY = "distributed-spectral-execution"
DISTRIBUTED_SPECTRAL_PROFILE_NAME = "distributed-spectral-execution.profile"
DISTRIBUTED_SPECTRAL_ROUTES = (
    "slab-roundtrip",
    "pencil-roundtrip",
    "padded-dealias",
    "channel-horizontal",
    "global-reductions",
    "scale-resource",
    "multi-host",
)
DISTRIBUTED_SPECTRAL_REQUIRED_GATES = (
    "forward-reference",
    "inverse-reference",
    "directional-jvp",
    "hilbert-adjoint",
    "precision-policy",
    "payload-admission",
    "stage-identity",
    "process-qualified-topology",
    "declared-resource",
    "compiler-memory",
    "no-host-gather",
    "physical-multi-device",
    "physical-multi-host",
    "channel-execution",
    "channel-horizontal-layout",
    "channel-horizontal-partition",
    "channel-atomic-zero-mode",
)
DISTRIBUTED_SPECTRAL_NONCLAIMS = (
    "no-dynamic-scheduling",
    "no-external-fft-provider",
    "no-forced-or-simulated-device-evidence",
    "no-implicit-host-gather",
    "no-les-or-pic-evidence-inheritance",
    "no-rank-one-slab",
    "no-r2c-or-c2r",
    "no-same-host-multi-host-evidence",
    "no-uneven-shards",
    "no-unlisted-mixed-transform-sequences",
    "not-release-authorized",
)

_COMMON_SUPPORT: dict[str, SupportValue] = {
    "backend": "jax",
    "decomposition": "regular-divisible-slab-and-pencil",
    "execution": "global-arrays-without-host-gather",
    "precision": "declared-transform-storage-accumulation-policy",
    "resources": "exact-admitted-payload-shapes-including-state",
    "topology": "process-qualified-physical-jax-mesh",
}


def _support_attributes(route: str, /) -> dict[str, SupportValue]:
    attributes = {**_COMMON_SUPPORT, "route": route}
    if route == "channel-horizontal":
        attributes["differentiation"] = "channel-action-derivative-not-claimed"
        attributes["geometry"] = "current-fourier-chebyshev-fourier-channel"
        attributes["transform"] = "execute-channel-horizontal-action"
    else:
        attributes["differentiation"] = "directional-jvp-and-hilbert-adjoint"
        attributes["geometry"] = "periodic-cartesian"
        attributes["transform"] = "full-complex-c2c"
    return attributes


def distributed_spectral_support_tuples() -> tuple[SupportTuple, ...]:
    """Return the exact evidence-free route support owned by this substrate."""

    return tuple(
        SupportTuple(
            DISTRIBUTED_SPECTRAL_CAPABILITY,
            _support_attributes(route),
        )
        for route in DISTRIBUTED_SPECTRAL_ROUTES
    )


def distributed_spectral_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Return the single unreleased profile without borrowing consumer evidence."""

    return (
        CapabilityProfile(
            DISTRIBUTED_SPECTRAL_PROFILE_NAME,
            "phydrax",
            "candidate",
            distributed_spectral_support_tuples(),
            required_gates=DISTRIBUTED_SPECTRAL_REQUIRED_GATES,
            released=False,
        ),
    )


def distributed_spectral_candidate_declarations() -> tuple[CapabilityDeclaration, ...]:
    """Declare the exact candidate and its fixed nonclaims at its scientific owner."""

    return (
        CapabilityDeclaration(
            DISTRIBUTED_SPECTRAL_CAPABILITY,
            "phydrax.discretization.spectral",
            CapabilityDisposition.CANDIDATE,
            domain_maturity="unreleased-candidate",
            profiles=distributed_spectral_candidate_profiles(),
            intended_uses=("bounded-distributed-spectral-engineering-evaluation",),
            nonclaims=DISTRIBUTED_SPECTRAL_NONCLAIMS,
        ),
    )


__all__ = [
    "DISTRIBUTED_SPECTRAL_CAPABILITY",
    "DISTRIBUTED_SPECTRAL_NONCLAIMS",
    "DISTRIBUTED_SPECTRAL_PROFILE_NAME",
    "DISTRIBUTED_SPECTRAL_REQUIRED_GATES",
    "DISTRIBUTED_SPECTRAL_ROUTES",
    "distributed_spectral_candidate_declarations",
    "distributed_spectral_candidate_profiles",
    "distributed_spectral_support_tuples",
]
