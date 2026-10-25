from typing import assert_type

from phydrax.backends import (
    assess_cuda_environment,
    cuda_environment,
    CudaEnvironmentFacts,
    CudaEnvironmentIssue,
    CudaEnvironmentReport,
    CudaPluginInstallation,
)


def consumer_contract(facts: CudaEnvironmentFacts) -> None:
    assert_type(cuda_environment(), CudaEnvironmentReport)
    assert_type(assess_cuda_environment(facts), CudaEnvironmentReport)
    assert_type(facts.plugins, tuple[CudaPluginInstallation, ...])
    assert_type(
        CudaPluginInstallation(
            distribution="phydrax-cuda13-pjrt", version="0.2.2", cuda_major=13
        ),
        CudaPluginInstallation,
    )
    assert_type(
        CudaEnvironmentIssue(
            code="unsupported-cuda-major", severity="error", message="CUDA 13 required"
        ),
        CudaEnvironmentIssue,
    )
    CudaEnvironmentIssue(code="unknown", severity="error", message="invalid code")  # ty: ignore[invalid-argument-type]
