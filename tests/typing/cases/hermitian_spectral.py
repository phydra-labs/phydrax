"""Hermitian functions and generic spectral-map results use public contracts."""

from typing import assert_type

import jax.numpy as jnp
from jax import Array

from phydrax.linalg import (
    hermitian_exp,
    hermitian_inverse_sqrt,
    hermitian_log,
    hermitian_spectral_map,
    hermitian_sqrt,
    HermitianFunctionResult,
    HermitianPrecisionPolicy,
    HermitianSpectralMapResult,
    HermitianSpectrum,
)


def principal_map(
    eigenvalues: Array, parameters: tuple[Array, Array]
) -> tuple[Array, dict[str, Array]]:
    scale, offset = parameters
    return jnp.stack((scale * eigenvalues, eigenvalues**3 + offset)), {
        "energy": jnp.sum(eigenvalues**2),
        "branch": jnp.asarray(1, dtype=jnp.int32),
    }


def hermitian_public_contract(matrix: Array, parameters: tuple[Array, Array]) -> None:
    precision = HermitianPrecisionPolicy()
    spectrum = HermitianSpectrum(matrix, precision=precision)
    assert_type(spectrum, HermitianSpectrum)
    for function in (
        hermitian_log,
        hermitian_exp,
        hermitian_sqrt,
        hermitian_inverse_sqrt,
    ):
        result = function(matrix, tolerance=1e-10, precision=precision)
        assert_type(result, HermitianFunctionResult)
        assert_type(result.value, Array)
        assert_type(result.spectrum, HermitianSpectrum)
        assert_type(result.valid, Array)
        assert_type(result.derivative_valid, Array)
        assert_type(result.function_id, str)
    result = HermitianFunctionResult(
        matrix,
        spectrum,
        valid=jnp.asarray(True),
        derivative_valid=jnp.asarray(True),
        function_id="typing-function",
    )
    assert_type(result, HermitianFunctionResult)
    mapped = hermitian_spectral_map(
        matrix, principal_map, parameters, function_id="typing-map", precision=precision
    )
    assert_type(mapped, HermitianSpectralMapResult[dict[str, Array]])
    assert_type(mapped.values, Array)
    assert_type(mapped.invariants, dict[str, Array])
    assert_type(mapped.spectrum, HermitianSpectrum)
    assert_type(mapped.valid, Array)
    assert_type(mapped.derivative_valid, Array)
    assert_type(mapped.function_id, str)
    constructed = HermitianSpectralMapResult[dict[str, Array]](
        mapped.values,
        mapped.invariants,
        spectrum,
        valid=mapped.valid,
        derivative_valid=mapped.derivative_valid,
        function_id="typing-map-result",
    )
    assert_type(constructed, HermitianSpectralMapResult[dict[str, Array]])
