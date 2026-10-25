#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Evidence-free CUDA library candidates, not installed or qualified hardware.

Scientific owners retain their native defaults. These declarations admit no
release evidence: availability, a successful build, and a device-local smoke
result cannot qualify a distributed topology or an installed artifact.
"""

from __future__ import annotations

from typing import get_args, Literal

from ..qualification._catalog import CapabilityDeclaration, CapabilityDisposition
from ..qualification._registry import CapabilityProfile, SupportTuple


type CudaTensorProvider = Literal["cutensor", "cutensornet"]
type CudaTensorPrecision = Literal["float32", "float64", "complex64", "complex128"]
type CudaRealPrecision = Literal["float32", "float64"]
type CudaTransformPrecision = Literal["complex64", "complex128"]
type CudaLocalMemoryMode = Literal["device", "hybrid"]
type CudaSameHostDeviceCount = Literal[2, 4, 8]


_CUDA_GATES = (
    "installed-wheel",
    "installed-native-artifact",
    "native-build-and-vendor-library-identity",
    "real-device",
    "real-device-topology",
    "precision-and-placement-refusal",
    "scientific-identity-and-forward-reference",
    "compiled-failure-propagation",
    "resource-evidence-and-refusal",
    "native-lifecycle-release-and-rebuild",
    "phase-separated-performance",
)
_COLLECTIVE_GATES = (
    "site-mpi-cuda-and-thread-multiple",
    "all-rank-device-mapping-and-peer-integrity",
    "staged-artifact-and-identical-collective-schedule",
    "all-rank-failure-and-supervised-retirement",
)
_COMMON_NONCLAIMS = (
    "available-is-not-qualified",
    "no-hardware-qualification-from-installation-or-build",
    "no-multi-host",
    "no-implicit-provider-selection-or-native-default-change",
    "no-implicit-host-transfer-or-native-numerical-fallback",
    "not-release-authorized",
)


def cuda_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    """Declare bounded candidate cases without manufacturing hardware evidence."""
    return (
        CapabilityProfile(
            "cuda-prepared-tensor.profile",
            "phydrax-cuda",
            "candidate",
            tuple(
                SupportTuple(
                    "cuda-prepared-tensor",
                    {
                        "platform": "cuda",
                        "host_platform": host,
                        "runtime": runtime,
                        "provider": provider,
                        "precision": precision,
                        "placement": "device-local",
                        "process_count": 1,
                        "device_count": 1,
                        "path": "native-or-explicit-imported-ssa",
                        "derivatives": "native-owned-jvp-vjp-through-order-two",
                        "batching": "bounded-serial-device-local-lanes",
                        "numerical_cache": False,
                        "vendor_host_bound": "unknown",
                        "resident_bound": "not-certified-all-process",
                    },
                )
                for provider in get_args(CudaTensorProvider.__value__)
                for precision in get_args(CudaTensorPrecision.__value__)
                for host, runtime in (
                    (("linux", "jax-cuda13-pjrt"), ("win32", "phydrax-cuda13-pjrt"))
                    if provider == "cutensor"
                    else (("linux", "jax-cuda13-pjrt"),)
                )
            ),
            required_gates=(*_CUDA_GATES, "jvp-vjp-second-order-and-batching-reference"),
        ),
        CapabilityProfile(
            "cuda-tensor-path-planning.profile",
            "phydrax-cuda",
            "candidate",
            (
                SupportTuple(
                    "cuda-tensor-path-planning",
                    {
                        "platform": "cuda",
                        "host_platform": "linux",
                        "runtime": "jax-cuda13-pjrt",
                        "provider": "cutensornet",
                        "process_count": 1,
                        "device_count": 1,
                        "phase": "shape-only-host-preparation",
                        "objective": "flops",
                        "worker": "spawned-deadline-bounded",
                        "slicing": False,
                        "numerical_execution": False,
                        "vendor_host_bound": "unknown",
                    },
                ),
            ),
            required_gates=(
                *_CUDA_GATES,
                "planner-deadline-worker-cleanup-and-path-validation",
            ),
        ),
        CapabilityProfile(
            "cuda-tensor-host-slices.profile",
            "phydrax-cuda",
            "candidate",
            tuple(
                SupportTuple(
                    "cuda-tensor-host-slices",
                    {
                        "platform": "cuda",
                        "host_platform": host,
                        "runtime": runtime,
                        "provider": provider,
                        "precision": precision,
                        "process_count": 1,
                        "device_count": 1,
                        "placement": "device-local-residual-executor",
                        "orchestration": "explicit-host-slices-and-checkpoints",
                        "derivatives": "not-admitted-through-host-orchestration",
                        "vendor_host_bound": "unknown",
                        "resident_bound": "not-certified-all-process",
                    },
                )
                for provider in get_args(CudaTensorProvider.__value__)
                for precision in get_args(CudaTensorPrecision.__value__)
                for host, runtime in (
                    (("linux", "jax-cuda13-pjrt"), ("win32", "phydrax-cuda13-pjrt"))
                    if provider == "cutensor"
                    else (("linux", "jax-cuda13-pjrt"),)
                )
            ),
            required_gates=(
                *_CUDA_GATES,
                "slice-coverage-checkpoint-identity-and-merge-reference",
                "host-orchestration-ad-refusal",
            ),
        ),
        CapabilityProfile(
            "cuda-cuest-df-rhf.profile",
            "phydrax-cuda",
            "candidate",
            (
                SupportTuple(
                    "cuda-cuest-df-rhf",
                    {
                        "platform": "cuda",
                        "host_platform": "linux",
                        "runtime": "jax-cuda13-pjrt",
                        "provider": "cuest",
                        "precision": "float64",
                        "x64_enabled": True,
                        "placement": "device-local",
                        "process_count": 1,
                        "device_count": 1,
                        "method": "full-range-nonperiodic-df-rhf",
                        "occupations": "integer-ground-state",
                        "system": "unpadded-all-electron",
                        "basis": "explicit-orbital-and-auxiliary-gaussian",
                        "geometry": "explicit-bound-dynamic-recipe-and-content-key",
                        "forces": "contracted-analytic-first-nuclear",
                        "automatic_differentiation": False,
                        "hessian": False,
                        "vendor_host_bound": "unknown",
                        "resident_bound": "unknown-outside-owned-workspaces",
                    },
                ),
            ),
            required_gates=(
                *_CUDA_GATES,
                "actual-density-psd-factorization-reference",
                "df-rhf-energy-and-analytic-first-force-reference",
                "bound-geometry-content-identity-and-retirement",
                "unsupported-ad-hessian-and-model-refusal",
            ),
        ),
        CapabilityProfile(
            "cuda-cudss-local.profile",
            "phydrax-cuda",
            "candidate",
            tuple(
                SupportTuple(
                    "cuda-cudss-local",
                    {
                        "platform": "cuda",
                        "host_platform": host,
                        "runtime": runtime,
                        "provider": "cudss",
                        "precision": precision,
                        "execution": "local",
                        "process_count": 1,
                        "device_count": 1,
                        "memory_mode": memory_mode,
                        "memory_evidence": "provider_estimate",
                        "certified_bounds": False,
                        "vendor_host_bound": "unknown",
                        "resident_bound": "unknown",
                        "hybrid_device_limit_scope": "post-analysis-factorization-and-solve-only",
                        "derivatives": "implicit-linear-solve",
                    },
                )
                for precision in get_args(CudaRealPrecision.__value__)
                for memory_mode in get_args(CudaLocalMemoryMode.__value__)
                for host, runtime in (
                    ("linux", "jax-cuda13-pjrt"),
                    ("win32", "phydrax-cuda13-pjrt"),
                )
            ),
            required_gates=(
                *_CUDA_GATES,
                "true-residual-inertia-and-implicit-derivative-reference",
                "hybrid-limit-scope-and-unknown-estimate-preservation",
            ),
        ),
        CapabilityProfile(
            "cuda-cudss-mg.profile",
            "phydrax-cuda",
            "candidate",
            tuple(
                SupportTuple(
                    "cuda-cudss-mg",
                    {
                        "platform": "cuda",
                        "host_platform": "linux",
                        "runtime": "jax-cuda13-pjrt",
                        "provider": "cudss",
                        "precision": precision,
                        "execution": "mg",
                        "host_count": 1,
                        "process_count": 1,
                        "device_count": count,
                        "placement": "full-input-output-on-first-reserved-device",
                        "reservation": "ordered-vendor-allocations-not-os-exclusive",
                        "memory_mode": "device",
                        "memory_evidence": "provider_estimate",
                        "certified_bounds": False,
                        "vendor_host_bound": "unknown-nonadditive-per-device-estimates",
                        "resident_bound": "unknown",
                        "derivatives": "implicit-linear-solve",
                    },
                )
                for precision in get_args(CudaRealPrecision.__value__)
                for count in get_args(CudaSameHostDeviceCount.__value__)
            ),
            required_gates=(
                *_CUDA_GATES,
                "ordered-reservation-and-peer-integrity",
                "mg-true-residual-inertia-and-implicit-derivative-reference",
            ),
        ),
        CapabilityProfile(
            "cuda-cudss-mgmn.profile",
            "phydrax-cuda",
            "candidate",
            tuple(
                SupportTuple(
                    "cuda-cudss-mgmn",
                    {
                        "platform": "cuda",
                        "host_platform": "linux",
                        "runtime": "jax-cuda13-pjrt",
                        "provider": "cudss",
                        "precision": precision,
                        "execution": "mgmn",
                        "host_count": 1,
                        "process_count": count,
                        "device_count": count,
                        "devices_per_rank": 1,
                        "placement": "rank-owned-distributed-csr",
                        "invocation": "staged-artifact-only-collective-lease",
                        "memory_mode": "device",
                        "memory_evidence": "provider_estimate",
                        "certified_bounds": False,
                        "vendor_host_bound": "unknown",
                        "resident_bound": "unknown",
                        "derivatives": "collective-implicit-linear-solve",
                    },
                )
                for precision in get_args(CudaRealPrecision.__value__)
                for count in get_args(CudaSameHostDeviceCount.__value__)
            ),
            required_gates=(
                *_CUDA_GATES,
                *_COLLECTIVE_GATES,
                "exact-site-communication-library-provenance",
                "owned-row-residual-and-collective-implicit-ad-reference",
                "zero-cotangent-participation-and-symbolic-activity-refusal",
            ),
        ),
        CapabilityProfile(
            "cuda-cudecomp-direct-slabs.profile",
            "phydrax-cuda",
            "candidate",
            tuple(
                SupportTuple(
                    "cuda-cudecomp-direct-slabs",
                    {
                        "platform": "cuda",
                        "host_platform": "linux",
                        "runtime": "jax-cuda13-pjrt",
                        "provider": "cudecomp",
                        "transform_precision": precision,
                        "host_count": 1,
                        "process_count": count,
                        "device_count": count,
                        "devices_per_rank": 1,
                        "schedule": "direct-three-dimensional-slab",
                        "mesh_rank": 1,
                        "packing": "explicit-leading-component-serial-3d",
                        "invocation": "staged-artifact-only-collective-lease",
                        "derivatives": "linear-permutation-inverse-transpose",
                        "vendor_host_bound": "unknown",
                        "resident_bound": "not-certified-all-process",
                    },
                )
                for precision in get_args(CudaTransformPrecision.__value__)
                for count in get_args(CudaSameHostDeviceCount.__value__)
            ),
            required_gates=(
                *_CUDA_GATES,
                *_COLLECTIVE_GATES,
                "vendor-pencil-endpoint-and-ordered-device-proof",
                "direct-slab-roundtrip-spectral-and-adjoint-reference",
                "external-scratch-and-packing-admission",
            ),
        ),
    )


def _declaration(
    profile: CapabilityProfile,
    owner: str,
    public_symbols: tuple[str, ...],
    documentation: tuple[str, ...],
    intended_uses: tuple[str, ...],
    nonclaims: tuple[str, ...],
    /,
) -> CapabilityDeclaration:
    return CapabilityDeclaration(
        profile.capability,
        owner,
        CapabilityDisposition.CANDIDATE,
        domain_maturity="unreleased-candidate",
        public_symbols=public_symbols,
        profiles=(profile,),
        documentation=documentation,
        intended_uses=intended_uses,
        nonclaims=(*_COMMON_NONCLAIMS, *nonclaims),
    )


def cuda_candidate_declarations() -> tuple[CapabilityDeclaration, ...]:
    """Keep optional-provider claims with their canonical scientific owners."""
    tensor, planner, slices, cuest, local, mg, mgmn, slabs = cuda_candidate_profiles()
    return (
        _declaration(
            tensor,
            "phydrax.tensor_network",
            (
                "phydrax.tensor_network.ContractionExecutionPolicy",
                "phydrax.tensor_network.prepare_contraction",
                "phydrax.tensor_network.execute_contraction",
                "phydrax.tensor_network.refresh_contraction",
            ),
            ("docs/api/tensor_network.md",),
            ("explicit-device-local-prepared-tensor-contractions",),
            (
                "ordinary-ein-contract-unchanged",
                "no-vendor-autodiff-or-derivatives-beyond-order-two",
                "no-differentiable-host-slice-or-checkpoint-orchestration",
                "no-certified-vendor-host-or-total-resident-memory-bound",
            ),
        ),
        _declaration(
            planner,
            "phydrax.tensor_network",
            (
                "phydrax.tensor_network.ContractionPathPlanner",
                "phydrax.tensor_network.plan_contraction",
            ),
            ("docs/api/tensor_network.md",),
            ("bounded-shape-only-cutensornet-path-search",),
            (
                "planner-is-not-the-numerical-executor",
                "no-planner-selected-slicing",
                "no-optimality-or-energy-objective-certificate",
                "no-certified-vendor-host-memory-bound",
            ),
        ),
        _declaration(
            slices,
            "phydrax.tensor_network",
            (
                "phydrax.tensor_network.PreparedSlicedContraction",
                "phydrax.tensor_network.prepare_sliced_contraction",
                "phydrax.tensor_network.execute_sliced_contraction",
                "phydrax.tensor_network.merge_slice_checkpoints",
            ),
            ("docs/api/tensor_network.md",),
            ("explicit-host-orchestrated-device-local-slices",),
            (
                "no-host-orchestration-autodiff",
                "no-implicit-distributed-slice-launch",
                "no-certified-vendor-host-or-total-resident-memory-bound",
            ),
        ),
        _declaration(
            cuest,
            "phydrax.operators.quantum.gaussian",
            (
                "phydrax.operators.quantum.gaussian.CuestIntegralPlan",
                "phydrax.operators.quantum.gaussian.CuestGeometry",
                "phydrax.operators.quantum.gaussian.MolecularIntegralPlan",
                "phydrax.chemistry.MolecularHartreeFockPlan",
                "phydrax.chemistry.MolecularRHFProvider",
            ),
            ("docs/api/backends.md",),
            ("explicit-full-range-df-rhf-energy-and-analytic-first-forces",),
            (
                "no-arbitrary-jvp-vjp-basis-derivatives-or-hessian",
                "no-vendor-scf",
                "no-periodic-dft-open-shell-or-range-separated-model-claim",
                "fitting-cutoff-is-not-metric-rank-or-energy-error-certificate",
                "owned-workspace-budget-is-not-total-resident-memory-bound",
                "runtime-cache-ids-and-weakrefs-are-not-portable-scientific-identity",
            ),
        ),
        _declaration(
            local,
            "phydrax.linalg",
            (
                "phydrax.linalg.SparseLDLT",
                "phydrax.linalg.CudssExecutionPolicy",
                "phydrax.linalg.CudssMemoryPolicy",
                "phydrax.linalg.CudssMemoryEvidence",
            ),
            ("docs/api/backends.md",),
            ("explicit-local-device-or-hybrid-sparse-ldlt",),
            (
                "provider-estimate-is-not-certified-all-phase-memory-bound",
                "no-enforceable-host-limit-or-certified-resident-bound",
                "hybrid-device-limit-does-not-bound-analysis",
                "analysis-estimates-use-one-rhs-and-exclude-retired-factors-and-caller-buffers",
            ),
        ),
        _declaration(
            mg,
            "phydrax.linalg",
            (),
            ("docs/api/backends.md",),
            ("explicit-single-process-ordered-multi-gpu-sparse-ldlt",),
            (
                "no-hybrid-plus-mg",
                "no-sharded-input-or-matrix-batches",
                "no-exclusive-os-gpu-ownership-claim",
                "provider-estimate-is-not-certified-all-phase-memory-bound",
                "mg-host-estimates-are-not-additive",
                "no-topology-qualification-from-device-local-success",
            ),
        ),
        _declaration(
            mgmn,
            "phydrax.linalg",
            (
                "phydrax.linalg.DistributedCSRStorage",
                "phydrax.linalg.DistributedCSRLinearOperator",
            ),
            ("docs/api/backends.md", "docs/api/execution.md"),
            ("same-host-one-gpu-per-rank-owned-distributed-sparse-ldlt",),
            (
                "no-hybrid-plus-mgmn",
                "no-cross-rank-transformation-schedule-divergence",
                "no-topology-qualification-from-build-or-mpi-query",
                "provider-estimate-is-not-certified-all-phase-memory-bound",
                "no-certified-host-or-total-resident-memory-bound",
            ),
        ),
        _declaration(
            slabs,
            "phydrax.discretization.spectral",
            (
                "phydrax.discretization.spectral.DistributedSpectralExecutionPlan",
                "phydrax.discretization.spectral.CudecompTransposeEvidence",
            ),
            ("docs/api/backends.md", "docs/api/execution.md"),
            ("same-host-one-gpu-per-rank-direct-three-dimensional-slab-transposes",),
            (
                "native-pencil-transposes-remain-native",
                "no-cudecomp-pencil-or-hidden-slab-fallback",
                "no-topology-qualification-from-build-or-mpi-query",
                "no-certified-vendor-host-or-total-resident-memory-bound",
            ),
        ),
    )
