#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Evidence-free oneAPI candidate scope; upstream wheels are not owned builds."""

from __future__ import annotations

from typing import get_args, Literal

from ..qualification._catalog import CapabilityDeclaration, CapabilityDisposition
from ..qualification._registry import CapabilityProfile, SupportTuple


type OneapiWorkload = Literal[
    "pde-mlp",
    "exact-integer-addressing",
    "compiled-control-flow",
    "iterative-linear",
    "real-fourier",
]


_ONEAPI_CAPABILITY = "intel-oneapi-execution"
_ONEAPI_GATES = (
    "installed-wheel",
    "upstream-runtime-identity",
    "real-device",
    "float32-explicit",
    "exact-integer-addressing",
    "compiled-errors",
    "forward-reference",
    "derivative-reference",
    "resource-evidence",
)


def oneapi_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Declare bounded float32 cases, not a release or hardware qualification."""
    return (
        CapabilityProfile(
            "intel-oneapi-execution.profile",
            "jax-oneapi",
            "candidate",
            tuple(
                SupportTuple(
                    _ONEAPI_CAPABILITY,
                    {
                        "platform": "oneapi",
                        "process_count": 1,
                        "execution_device_count": 1,
                        "workload": workload,
                        "precision": "float32",
                        "integer_addressing": "exact-int64-uint64",
                        "x64_enabled": True,
                    },
                )
                for workload in get_args(OneapiWorkload.__value__)
            ),
            required_gates=_ONEAPI_GATES,
        ),
    )


def oneapi_candidate_declarations() -> tuple[CapabilityDeclaration, ...]:
    return (
        CapabilityDeclaration(
            _ONEAPI_CAPABILITY,
            "phydrax.backends",
            CapabilityDisposition.CANDIDATE,
            domain_maturity="unreleased-candidate",
            public_symbols=(
                "phydrax.backends.assess_oneapi_environment",
                "phydrax.backends.oneapi_environment",
            ),
            profiles=oneapi_candidate_profiles(),
            documentation=("docs/api/backends.md",),
            intended_uses=("bounded-single-device-scientific-evaluation",),
            nonclaims=(
                "no-float64-or-complex-coordinate-qualification",
                "no-multi-device-or-multi-host-qualification",
                "no-oneapi-mpi-provider",
                "no-cuda-or-pallas-kernel-parity",
                "no-implicit-cpu-numerical-fallback",
                "no-owned-native-companion",
                "mixed-jax-jaxlib-stack-not-intel-matched-release",
                "no-hardware-qualification-from-installation",
                "not-release-authorized",
            ),
        ),
    )
