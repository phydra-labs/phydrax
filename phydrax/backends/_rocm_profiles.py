# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Evidence-free ROCm candidates; installation never certifies hardware execution."""

from __future__ import annotations

from ..qualification._catalog import CapabilityDeclaration, CapabilityDisposition
from ..qualification._registry import CapabilityProfile, SupportTuple


_ROCM_CAPABILITY = "amd-rocm-execution"
_ROCM_GATES = (
    "installed-wheel",
    "upstream-runtime-identity",
    "owned-rocm-runtime",
    "all-architecture-assets",
    "automatic-runtime-setup",
    "real-device",
    "explicit-coordinate-dtype",
    "exact-integer-addressing",
    "compiled-errors",
    "forward-reference",
    "derivative-reference",
    "linear-failure-status",
    "resource-evidence",
)
# Each pair corresponds to a bounded installed-wheel consumer contract. Wide
# integer and callback gates remain mandatory for the selected runtime.
_ROCM_WORKLOAD_PRECISIONS = (
    ("pde-mlp", "float32"),
    ("exact-integer-addressing", "float32"),
    ("compiled-control-flow", "float32"),
    ("iterative-linear", "float32"),
    ("real-fourier", "float32"),
    ("iterative-linear", "float64"),
    ("iterative-linear", "complex128"),
    ("fourier-resampling", "float64"),
    ("fourier-resampling", "complex128"),
)
# Model names are examples of ISA coverage, never runtime admission allowlists.
# gfx1250 is published upstream, but build/sanity testing is not release readiness.
_ROCM_ARCHITECTURE_CANDIDATES = (
    (
        "gfx1250",
        "MI400",
        "MI450,MI450X,MI455X",
        "build-and-sanity-tested-not-release-ready",
    ),
    ("gfx942", "MI300", "MI300A,MI300X,MI325X", "rocm-10.1-supported"),
    ("gfx950", "MI300", "MI350P,MI350X,MI355X", "rocm-10.1-supported"),
)


def rocm_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Declare ROCm10 candidates without claiming passing hardware evidence."""
    return (
        CapabilityProfile(
            "amd-rocm10-execution.profile",
            "jax-rocm10",
            "candidate",
            tuple(
                SupportTuple(
                    _ROCM_CAPABILITY,
                    {
                        "platform": "rocm",
                        "vendor": "amd",
                        "install_selector": "rocm10",
                        "rocm_major": 10,
                        "plugin_version": "0.12.0",
                        "rocm_userspace_version": "10.1.0",
                        "architecture": architecture,
                        "device_family": family,
                        "example_models": models,
                        "model_names_are_allowlist": False,
                        "upstream_readiness": readiness,
                        "process_count": 1,
                        "execution_device_count": 1,
                        "workload": workload,
                        "precision": precision,
                        "integer_addressing": "exact-int64-uint64",
                        "x64_enabled": True,
                    },
                )
                for architecture, family, models, readiness in _ROCM_ARCHITECTURE_CANDIDATES
                for workload, precision in _ROCM_WORKLOAD_PRECISIONS
            ),
            required_gates=_ROCM_GATES,
        ),
    )


def rocm_candidate_declarations() -> tuple[CapabilityDeclaration, ...]:
    return (
        CapabilityDeclaration(
            _ROCM_CAPABILITY,
            "phydrax.backends",
            CapabilityDisposition.CANDIDATE,
            domain_maturity="unreleased-candidate",
            public_symbols=(
                "phydrax.backends.assess_rocm_environment",
                "phydrax.backends.rocm_environment",
            ),
            profiles=rocm_candidate_profiles(),
            documentation=("docs/api/backends.md",),
            intended_uses=("bounded-single-device-scientific-evaluation",),
            nonclaims=(
                "no-multi-device-or-multi-host-qualification",
                "no-rocm-mpi4jax-provider",
                "no-cuda-or-pallas-kernel-parity",
                "no-compiled-mace-fmm-cudss-amgx-or-quda-qualification",
                "no-implicit-cpu-numerical-fallback",
                "no-owned-upstream-build-identity",
                "no-hardware-qualification-from-installation",
                "no-mi350-hardware-qualification-from-static-gfx950-assets",
                "no-mi400-hardware-qualification-from-static-gfx1250-assets",
                "no-gfx1250-upstream-release-readiness",
                "no-unmapped-future-mi400-sku-coverage",
                "not-release-authorized",
            ),
        ),
    )
