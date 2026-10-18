from typing import assert_type

from phydrax.backends import (
    assess_cuda_environment,
    cuda_environment,
    CudaEnvironmentFacts,
    CudaEnvironmentIssue,
    CudaEnvironmentReport,
)


def consumer_contract(facts: CudaEnvironmentFacts) -> None:
    assert_type(cuda_environment(), CudaEnvironmentReport)
    assert_type(assess_cuda_environment(facts), CudaEnvironmentReport)
    assert_type(
        CudaEnvironmentIssue(
            code="unsupported-cuda-major", severity="error", message="CUDA 13 required"
        ),
        CudaEnvironmentIssue,
    )
    CudaEnvironmentIssue(code="unknown", severity="error", message="invalid code")  # ty: ignore[invalid-argument-type]
