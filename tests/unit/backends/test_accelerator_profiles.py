from __future__ import annotations

from collections.abc import Callable

import pytest

from phydrax.backends._oneapi_profiles import (
    oneapi_candidate_declarations,
    oneapi_candidate_profiles,
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
