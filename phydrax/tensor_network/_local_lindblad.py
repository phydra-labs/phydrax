#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import HermitianSpectrum


def _adjoint(value: Array, /) -> Array:
    return jnp.swapaxes(jnp.conj(value), -1, -2)


class LocalKrausPreparationEvidence(StrictModule):
    choi_hermiticity_residual: Array
    raw_minimum_choi_eigenvalue: Array
    numerical_cleanup_norm: Array
    completeness_residual: Array
    reconstruction_residual: Array
    valid: Array

    def __init__(
        self,
        *,
        choi_hermiticity_residual: ArrayLike,
        raw_minimum_choi_eigenvalue: ArrayLike,
        numerical_cleanup_norm: ArrayLike,
        completeness_residual: ArrayLike,
        reconstruction_residual: ArrayLike,
        tolerance: float,
    ):
        self.choi_hermiticity_residual = jnp.asarray(choi_hermiticity_residual)
        self.raw_minimum_choi_eigenvalue = jnp.asarray(raw_minimum_choi_eigenvalue)
        self.numerical_cleanup_norm = jnp.asarray(numerical_cleanup_norm)
        self.completeness_residual = jnp.asarray(completeness_residual)
        self.reconstruction_residual = jnp.asarray(reconstruction_residual)
        self.valid = (
            jnp.isfinite(self.raw_minimum_choi_eigenvalue)
            & (self.raw_minimum_choi_eigenvalue >= -tolerance)
            & (self.completeness_residual <= tolerance)
            & (self.reconstruction_residual <= 10.0 * tolerance)
        )


class PreparedLocalKrausChannel(StrictModule):
    kraus: Array
    superoperator: Array
    evidence: LocalKrausPreparationEvidence
    dimension: int
    step_size: Array

    def __init__(
        self,
        kraus: ArrayLike,
        superoperator: ArrayLike,
        evidence: LocalKrausPreparationEvidence,
        step_size: ArrayLike,
        /,
    ):
        values = jnp.asarray(kraus)
        self.kraus = values
        self.superoperator = jnp.asarray(superoperator)
        self.evidence = evidence
        self.dimension = values.shape[-1]
        self.step_size = jnp.asarray(step_size)


def prepare_local_lindblad_channel(
    hamiltonian: ArrayLike,
    jump_operators: ArrayLike,
    step_size: ArrayLike,
    /,
    *,
    tolerance: float = 1e-9,
) -> PreparedLocalKrausChannel:
    raw_hamiltonian = jnp.asarray(hamiltonian)
    raw_jumps = jnp.asarray(jump_operators)
    dtype = jnp.result_type(raw_hamiltonian, raw_jumps, jnp.complex64)
    hamiltonian_ = raw_hamiltonian.astype(dtype)
    jumps = raw_jumps.astype(dtype)
    step = jnp.asarray(step_size, dtype=hamiltonian_.real.dtype).reshape(())
    if hamiltonian_.ndim != 2 or hamiltonian_.shape[0] != hamiltonian_.shape[1]:
        raise ValueError("Local Hamiltonian must be square.")
    dimension = hamiltonian_.shape[0]
    if jumps.ndim != 3 or jumps.shape[1:] != (dimension, dimension):
        raise ValueError("Local jumps require shape (count,d,d).")
    if not bool(jnp.isfinite(step) & (step > 0.0)):
        raise ValueError("Local channel step_size must be finite and positive.")
    if not bool(
        jnp.all(jnp.isfinite(hamiltonian_))
        & jnp.all(jnp.isfinite(jumps))
        & jnp.allclose(hamiltonian_, _adjoint(hamiltonian_), rtol=1e-10, atol=1e-12)
    ):
        raise ValueError("Local Hamiltonian must be finite and Hermitian; jumps finite.")

    def generator(density):
        result = -1j * (hamiltonian_ @ density - density @ hamiltonian_)
        for jump in jumps:
            product = _adjoint(jump) @ jump
            result = (
                result
                + jump @ density @ _adjoint(jump)
                - 0.5 * (product @ density + density @ product)
            )
        return result

    size = dimension**2
    basis = jnp.eye(size, dtype=hamiltonian_.dtype).reshape((size, dimension, dimension))
    generator_matrix = jnp.swapaxes(
        jax.vmap(lambda density: generator(density).reshape(-1))(basis), -1, -2
    )
    superoperator = jsp.linalg.expm(step * generator_matrix)
    choi = jnp.zeros(
        (dimension, dimension, dimension, dimension), dtype=superoperator.dtype
    )
    for row in range(dimension):
        for column in range(dimension):
            matrix = (
                jnp.zeros((dimension, dimension), dtype=superoperator.dtype)
                .at[row, column]
                .set(1.0)
            )
            output = (superoperator @ matrix.reshape(-1)).reshape((dimension, dimension))
            choi = choi.at[row, :, column, :].set(output)
    flat = choi.reshape((size, size))
    hermiticity = jnp.max(jnp.abs(flat - _adjoint(flat)))
    hermitian = 0.5 * flat + 0.5 * _adjoint(flat)
    spectrum = HermitianSpectrum(hermitian)
    eigenvalues = spectrum.eigenvalues
    eigenvectors = spectrum.eigenvectors
    minimum = spectrum.minimum_eigenvalue
    if bool(jax.device_get(minimum < -tolerance)):
        raise ValueError("Finite local Lindblad channel has a nonpositive Choi matrix.")
    cleaned = jnp.maximum(eigenvalues, 0.0)
    cleanup = jnp.sqrt(jnp.sum(jnp.abs(cleaned - eigenvalues) ** 2))
    active = cleaned > tolerance
    kraus = jnp.stack(
        [
            jnp.sqrt(cleaned[index])
            * eigenvectors[:, index].reshape((dimension, dimension)).T
            for index in range(size)
        ]
    )
    kraus = jnp.where(active[:, None, None], kraus, 0.0)
    completeness = sum(_adjoint(operator) @ operator for operator in kraus)
    completeness_residual = jnp.linalg.norm(
        completeness - jnp.eye(dimension, dtype=kraus.dtype)
    )
    reconstructed = jnp.swapaxes(
        jax.vmap(
            lambda density: sum(
                operator @ density @ _adjoint(operator) for operator in kraus
            ).reshape(-1)
        )(basis),
        -1,
        -2,
    )
    reconstruction_residual = jnp.linalg.norm(reconstructed - superoperator)
    evidence = LocalKrausPreparationEvidence(
        choi_hermiticity_residual=hermiticity,
        raw_minimum_choi_eigenvalue=minimum,
        numerical_cleanup_norm=cleanup,
        completeness_residual=completeness_residual,
        reconstruction_residual=reconstruction_residual,
        tolerance=tolerance,
    )
    return PreparedLocalKrausChannel(kraus, superoperator, evidence, step)


__all__ = [
    "LocalKrausPreparationEvidence",
    "PreparedLocalKrausChannel",
    "prepare_local_lindblad_channel",
]
