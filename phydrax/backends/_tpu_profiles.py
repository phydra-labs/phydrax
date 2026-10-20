#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Evidence-free TPU candidate scope; libtpu retains upstream runtime identity."""

from __future__ import annotations

from typing import get_args, Literal

from ..qualification._catalog import CapabilityDeclaration, CapabilityDisposition
from ..qualification._registry import CapabilityProfile, SupportTuple


type TpuWorkload = Literal[
    "pde-mlp",
    "exact-integer-addressing",
    "compiled-control-flow",
    "iterative-linear",
    "real-fourier",
]


_TPU_CAPABILITY = "google-tpu-execution"
_TPU_GATES = (
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


def tpu_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Declare float32 cases under explicit TPU matrix-precision semantics."""
    return (
        CapabilityProfile(
            "google-tpu-execution.profile",
            "jax-tpu",
            "candidate",
            tuple(
                SupportTuple(
                    _TPU_CAPABILITY,
                    {
                        "platform": "tpu",
                        "process_count": 1,
                        "execution_device_count": 1,
                        "workload": workload,
                        "precision": "float32",
                        "matmul_precision": "highest",
                        "integer_addressing": "exact-int64-uint64",
                        "x64_enabled": True,
                    },
                )
                for workload in get_args(TpuWorkload.__value__)
            ),
            required_gates=_TPU_GATES,
        ),
    )


def tpu_candidate_declarations() -> tuple[CapabilityDeclaration, ...]:
    return (
        CapabilityDeclaration(
            _TPU_CAPABILITY,
            "phydrax.backends",
            CapabilityDisposition.CANDIDATE,
            domain_maturity="unreleased-candidate",
            public_symbols=(
                "phydrax.backends.assess_tpu_environment",
                "phydrax.backends.tpu_environment",
            ),
            profiles=tpu_candidate_profiles(),
            documentation=("docs/api/backends.md",),
            intended_uses=("bounded-single-device-scientific-evaluation",),
            nonclaims=(
                "no-native-float64-or-complex128",
                "no-emulated-float64-or-complex-coordinate-qualification",
                "highest-matmul-is-six-bfloat16-passes-not-native-ieee-float32",
                "no-multi-device-or-multi-host-qualification",
                "no-cuda-provider-or-pallas-particle-fmm-parity",
                "no-implicit-cpu-numerical-fallback",
                "no-owned-native-companion",
                "no-hardware-qualification-from-installation",
                "not-release-authorized",
            ),
        ),
    )
