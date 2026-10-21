from __future__ import annotations

from collections.abc import Callable

import pytest

from phydrax.backends._oneapi_profiles import (
    oneapi_candidate_declarations,
    oneapi_candidate_profiles,
)
from phydrax.backends._rocm_profiles import (
    rocm_candidate_declarations,
    rocm_candidate_profiles,
)
from phydrax.backends._tpu_profiles import (
    tpu_candidate_declarations,
    tpu_candidate_profiles,
)
from phydrax.backends.distributed import (
    INTEL_JAX_PROFILE,
    JAX_NATIVE_DISTRIBUTION,
    MPI4JAX_CAPABILITIES,
    TPU_JAX_PROFILE,
    vendor_execution_profiles,
)
from phydrax.backends.spatial import (
    PALLAS_SPATIAL_CAPABILITIES,
    PALLAS_TPU_SPATIAL_CAPABILITIES,
)
from phydrax.qualification._catalog import (
    CapabilityDeclaration,
    CapabilityDisposition,
    EvidenceState,
)
from phydrax.qualification._registry import CapabilityProfile


def test_intel_profile_uses_actual_upstream_platform_without_distributed_claims() -> None:
    assert INTEL_JAX_PROFILE.provider_id == "jax-oneapi"
    assert INTEL_JAX_PROFILE.platform == "oneapi"
    assert INTEL_JAX_PROFILE.vendor == "intel"
    assert INTEL_JAX_PROFILE.experimental
    INTEL_JAX_PROFILE.require(scope="single_device", dtype="float32")
    for scope in ("multi_device", "multi_host"):
        with pytest.raises(ValueError, match="scope"):
            INTEL_JAX_PROFILE.require(scope=scope, dtype="float32")
    assert "oneapi" not in MPI4JAX_CAPABILITIES.distribution.platforms
    assert "xpu" not in MPI4JAX_CAPABILITIES.distribution.platforms
    assert "plugin" not in JAX_NATIVE_DISTRIBUTION.platforms
    assert all(profile.platform != "xpu" for profile in vendor_execution_profiles())


@pytest.mark.parametrize("dtype", ("float64", "complex64", "complex128"))
def test_tpu_admission_does_not_claim_fp64_or_complex(dtype: str) -> None:
    assert not TPU_JAX_PROFILE.experimental
    with pytest.raises(ValueError, match="dtype"):
        TPU_JAX_PROFILE.require(scope="single_device", dtype=dtype)
    TPU_JAX_PROFILE.require(scope="single_device", dtype="float32")
    TPU_JAX_PROFILE.require(scope="multi_device", dtype="bfloat16")


@pytest.mark.parametrize(
    ("profiles", "declarations", "platform", "provider"),
    (
        (
            oneapi_candidate_profiles,
            oneapi_candidate_declarations,
            "oneapi",
            "jax-oneapi",
        ),
        (tpu_candidate_profiles, tpu_candidate_declarations, "tpu", "jax-tpu"),
    ),
)
def test_accelerator_candidates_carry_scope_without_fabricated_evidence(
    profiles: Callable[[], tuple[CapabilityProfile, ...]],
    declarations: Callable[[], tuple[CapabilityDeclaration, ...]],
    platform: str,
    provider: str,
) -> None:
    (profile,) = profiles()
    (declaration,) = declarations()
    assert profile.provider == provider
    assert profile.version == "candidate"
    assert profile.release_evidence == ()
    assert declaration.disposition is CapabilityDisposition.CANDIDATE
    assert all(item.state is not EvidenceState.PASSED for item in declaration.evidence)
    assert declaration.profiles[0].profile_id == profile.profile_id
    assert {dict(item.attributes)["workload"] for item in profile.support_tuples} == {
        "pde-mlp",
        "exact-integer-addressing",
        "compiled-control-flow",
        "iterative-linear",
        "real-fourier",
    }
    for support in profile.support_tuples:
        attributes = dict(support.attributes)
        assert attributes["platform"] == platform
        assert attributes["precision"] == "float32"
        assert attributes["execution_device_count"] == attributes["process_count"] == 1
        assert "device_count" not in attributes
        assert attributes["integer_addressing"] == "exact-int64-uint64"
        assert attributes["x64_enabled"] is True
        if platform == "tpu":
            assert attributes["matmul_precision"] == "highest"
    assert {"installed-wheel", "upstream-runtime-identity", "real-device"} <= set(
        profile.required_gates
    )
    assert "not-release-authorized" in declaration.nonclaims


def test_pallas_tpu_metadata_preserves_distance_leaves_without_fmm_or_fp64() -> None:
    assert "tpu" not in PALLAS_SPATIAL_CAPABILITIES.distribution.platforms
    assert PALLAS_SPATIAL_CAPABILITIES.supports("particle_fmm.p2p")
    assert PALLAS_TPU_SPATIAL_CAPABILITIES.distribution.platforms == ("tpu",)
    assert PALLAS_TPU_SPATIAL_CAPABILITIES.coordinate_dtypes == ("float32",)
    for capability in (
        "spatial.distance_squared",
        "spatial.morton_knn_leaf",
        "spatial.morton_radius_leaf",
    ):
        assert PALLAS_TPU_SPATIAL_CAPABILITIES.supports(capability)
    assert not PALLAS_TPU_SPATIAL_CAPABILITIES.supports("particle_fmm.p2p")


def test_rocm_candidates_keep_precision_and_hardware_scopes_explicit() -> None:
    profiles = rocm_candidate_profiles()
    (declaration,) = rocm_candidate_declarations()
    assert len(profiles) == 1
    assert declaration.disposition is CapabilityDisposition.CANDIDATE
    assert declaration.profiles == profiles
    assert all(item.state is not EvidenceState.PASSED for item in declaration.evidence)
    for profile, major, userspace in zip(profiles, (10,), ("10.1.0",), strict=True):
        assert profile.provider == f"jax-rocm{major}"
        assert profile.version == "candidate"
        assert profile.release_evidence == ()
        scopes = [dict(support.attributes) for support in profile.support_tuples]
        assert {(scope["workload"], scope["precision"]) for scope in scopes} == {
            ("pde-mlp", "float32"),
            ("exact-integer-addressing", "float32"),
            ("compiled-control-flow", "float32"),
            ("iterative-linear", "float32"),
            ("real-fourier", "float32"),
            ("iterative-linear", "float64"),
            ("iterative-linear", "complex128"),
            ("fourier-resampling", "float64"),
            ("fourier-resampling", "complex128"),
        }
        assert len(scopes) == 27
        architecture_scope = {scope["architecture"]: scope for scope in scopes}
        assert architecture_scope["gfx942"]["example_models"] == "MI300A,MI300X,MI325X"
        assert architecture_scope["gfx950"]["example_models"] == "MI350P,MI350X,MI355X"
        assert architecture_scope["gfx1250"]["example_models"] == "MI450,MI450X,MI455X"
        assert architecture_scope["gfx942"]["device_family"] == "MI300"
        assert architecture_scope["gfx950"]["device_family"] == "MI300"
        assert architecture_scope["gfx1250"]["device_family"] == "MI400"
        assert architecture_scope["gfx1250"]["upstream_readiness"] == (
            "build-and-sanity-tested-not-release-ready"
        )
        for scope in scopes:
            assert scope["platform"] == "rocm"
            assert scope["vendor"] == "amd"
            assert scope["install_selector"] == f"rocm{major}"
            assert scope["rocm_major"] == major
            assert scope["plugin_version"] == "0.12.0"
            assert scope["rocm_userspace_version"] == userspace
            assert scope["process_count"] == scope["execution_device_count"] == 1
            assert scope["x64_enabled"] is True
            assert scope["integer_addressing"] == "exact-int64-uint64"
            assert scope["model_names_are_allowlist"] is False
        assert {
            "installed-wheel",
            "upstream-runtime-identity",
            "owned-rocm-runtime",
            "all-architecture-assets",
            "automatic-runtime-setup",
            "real-device",
            "explicit-coordinate-dtype",
            "linear-failure-status",
            "forward-reference",
            "derivative-reference",
        } <= set(profile.required_gates)
    assert {
        "no-mi350-hardware-qualification-from-static-gfx950-assets",
        "no-mi400-hardware-qualification-from-static-gfx1250-assets",
        "no-gfx1250-upstream-release-readiness",
        "no-unmapped-future-mi400-sku-coverage",
        "no-hardware-qualification-from-installation",
        "no-multi-device-or-multi-host-qualification",
        "no-cuda-or-pallas-kernel-parity",
        "not-release-authorized",
    } <= set(declaration.nonclaims)
