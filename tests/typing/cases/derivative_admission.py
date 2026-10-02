"""Canonical derivative admissions retain optional dynamic numerical evidence."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from typing_extensions import assert_type

from phydrax import (
    DerivativeAdmission,
    DerivativeContract,
    DerivativeSurface,
    DifferentiationRequest,
)
from phydrax.ml import FitResult
from phydrax.ml._contracts import OperationDerivativeContract


request = DifferentiationRequest((DerivativeSurface.FIT_FEATURES,))
contract = DerivativeContract.smooth((DerivativeSurface.FIT_FEATURES,))
admission = contract.admit(request)
assert_type(admission, DerivativeAdmission)
assert_type(admission.runtime_valid, Array | None)
assert_type(admission.runtime_status, Array | None)
resolved = admission.with_runtime_evidence(jnp.array([True]), jnp.array([0]))
assert_type(resolved, DerivativeAdmission)
assert_type(resolved.require_runtime(), DerivativeAdmission)
operation = OperationDerivativeContract(
    "project",
    contract,
    runtime_surfaces=(DerivativeSurface.FIT_FEATURES,),
    runtime_valid=jnp.array([True]),
    runtime_status=jnp.array([0]),
)
assert_type(operation.admit(request), DerivativeAdmission)
assert_type(operation.require(request), DerivativeAdmission)


def fitted_admission(result: FitResult, /) -> DerivativeAdmission:
    assert_type(
        result.derivative_admission(request, operation="project"), DerivativeAdmission
    )
    return result.require_derivative(request, operation="project")


admission.with_runtime_evidence(True, jnp.array([0]))  # ty: ignore[invalid-argument-type]
operation.admit("fit-features")  # ty: ignore[invalid-argument-type]
