#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Evidence-free candidate scope; native build and hardware evidence is separate."""

from __future__ import annotations

from typing import get_args, Literal

from ..qualification._catalog import CapabilityDeclaration, CapabilityDisposition
from ..qualification._registry import CapabilityProfile, SupportTuple


type MetalWorkload = Literal[
    "pde-mlp",
    "prepared-neighbors",
    "ordered-sparse",
    "native-mace",
    "fourier",
    "iterative-linear",
    "temporal",
]


_METAL_CAPABILITY = "apple-metal-execution"
_METAL_GATES = (
    "installed-wheel",
    "native-build-identity",
    "real-device",
    "precision-refusal",
    "exact-integer-addressing",
    "compiled-errors",
    "forward-reference",
    "derivative-reference",
    "ordered-replay",
    "resource-evidence",
    "native-lifecycle",
    "phase-separated-performance",
)


def metal_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    return (
        CapabilityProfile(
            "apple-metal-execution.profile",
            "phydrax-metal",
            "candidate",
            tuple(
                SupportTuple(
                    _METAL_CAPABILITY,
                    {
                        "platform": "mps",
                        "process_count": 1,
                        "device_count": 1,
                        "workload": workload,
                        "precision": "complex64" if workload == "fourier" else "float32",
                        "integer_addressing": "exact-int64-uint64",
                        "x64_enabled": True,
                    },
                )
                for workload in get_args(MetalWorkload.__value__)
            ),
            required_gates=_METAL_GATES,
        ),
    )


def metal_candidate_declarations() -> tuple[CapabilityDeclaration, ...]:
    return (
        CapabilityDeclaration(
            _METAL_CAPABILITY,
            "phydrax.backends",
            CapabilityDisposition.CANDIDATE,
            domain_maturity="unreleased-candidate",
            public_symbols=(
                "phydrax.backends.assess_metal_environment",
                "phydrax.backends.metal_environment",
            ),
            profiles=metal_candidate_profiles(),
            documentation=("docs/api/backends.md",),
            intended_uses=("bounded-single-device-scientific-evaluation",),
            nonclaims=(
                "no-float64-or-complex128",
                "no-cuda-native-provider-parity",
                "no-pallas-metal-kernels",
                "no-multi-device-or-multi-host",
                "no-implicit-cpu-numerical-fallback",
                "no-hardware-qualification-from-installation",
                "not-release-authorized",
            ),
        ),
    )
