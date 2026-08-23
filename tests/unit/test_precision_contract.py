#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax._numerics import log_normalize, LogWeightedAccumulator, weight_ess
from phydrax._precision import (
    complex_precision_dtype,
    precision_dtype_name,
    PrecisionEvidenceEnvelope,
    PrecisionRequest,
    PrecisionResolution,
    PrecisionResourceAssumptions,
)


def test_precision_contracts_are_strict_nested_and_content_addressed():
    request = PrecisionRequest(
        "test-domain",
        {"storage": "float32", "accumulation": "float64"},
    )
    resolution = PrecisionResolution(
        request,
        "test-provider",
        {"storage": "float32", "accumulation": "float64"},
    )
    child = PrecisionEvidenceEnvelope(
        resolution,
        {"storage": "float32", "accumulation": "float64"},
    )
    evidence = PrecisionEvidenceEnvelope(
        resolution,
        {"storage": "float32", "accumulation": "float64"},
        children={"child": child},
    )

    assert PrecisionRequest.from_dict(request.to_dict()) == request
    assert PrecisionResolution.from_dict(resolution.to_dict()) == resolution
    assert PrecisionEvidenceEnvelope.from_dict(evidence.to_dict()) == evidence

    corrupted = evidence.to_dict()
    corrupted["observed"]["storage"] = "float64"
    with pytest.raises(ValueError, match="identity mismatch"):
        PrecisionEvidenceEnvelope.from_dict(corrupted)


def test_precision_dtype_vocabulary_distinguishes_semantic_and_storage_names():
    assert precision_dtype_name(jnp.float64) == "float64"
    assert precision_dtype_name(jnp.complex64) == "complex64"
    assert complex_precision_dtype("bfloat16") == "complex64"
    assert complex_precision_dtype("float64") == "complex128"
    with pytest.raises(ValueError, match="Unsupported precision dtype"):
        precision_dtype_name(jnp.int32)


def test_precision_resource_assumptions_round_trip_without_execution_claims():
    assumptions = PrecisionResourceAssumptions(
        "test-domain",
        {"storage": "float32", "checkpoint": "float64"},
    )
    restored = PrecisionResourceAssumptions.from_dict(assumptions.to_dict())

    assert restored == assumptions
    assert assumptions.itemsize("storage") == 4
    assert assumptions.itemsize("checkpoint") == 8

    corrupted = assumptions.to_dict()
    corrupted["item_sizes"]["storage"] = 8
    with pytest.raises(ValueError, match="identity mismatch"):
        PrecisionResourceAssumptions.from_dict(corrupted)


def test_stable_reductions_respect_explicit_accumulation_dtype():
    log_weights = jnp.asarray([-1000.0, -1001.0, -1002.0], dtype=jnp.float32)
    normalized, log_sum, valid = log_normalize(
        log_weights,
        accumulation_dtype="float64",
    )
    ess = weight_ess(normalized, accumulation_dtype="float64")

    assert normalized.dtype == jnp.float64
    assert log_sum.dtype == jnp.float64
    assert ess.dtype == jnp.float64
    assert valid
    assert jnp.allclose(jnp.sum(normalized), 1.0)


def test_log_weighted_accumulator_widens_real_and_complex_values():
    real = LogWeightedAccumulator.from_values(
        jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        jnp.asarray([0.0, -1.0], dtype=jnp.float32),
        accumulation_dtype="float64",
    )
    complex_values = LogWeightedAccumulator.from_values(
        jnp.asarray([1.0 + 2.0j, 3.0 - 1.0j], dtype=jnp.complex64),
        jnp.asarray([0.0, -1.0], dtype=jnp.float32),
        accumulation_dtype="float64",
    )

    assert real.normalized_mean.dtype == jnp.float64
    assert complex_values.normalized_mean.dtype == jnp.complex128
    assert jax.jit(lambda value: value.normalized_mean)(real).dtype == jnp.float64
