#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._geometry_precision import GeometryPrecisionPolicy
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ..linalg import HermitianPrecisionPolicy, HermitianSpectrum


def canonical_commutation_matrix(mode_count: int, /, *, dtype=float) -> Array:
    modes = int(mode_count)
    if modes < 1:
        raise ValueError("mode_count must be positive.")
    block = jnp.asarray([[0.0, 1.0], [-1.0, 0.0]], dtype=dtype)
    return jnp.kron(jnp.eye(modes, dtype=dtype), block)


class BosonicGaussianState(StrictModule):
    mean: Array
    covariance: Array
    symplectic_form: Array
    uncertainty_margin: Array
    symmetry_residual: Array
    symplectic_eigenvalues: Array
    purity: Array
    valid: Array
    geometry_precision: GeometryPrecisionPolicy
    hermitian_precision: HermitianPrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    mode_count: int = eqx.field(static=True)
    hbar: float = eqx.field(static=True)

    def __init__(
        self,
        mean: ArrayLike,
        covariance: ArrayLike,
        /,
        *,
        hbar: float = 1.0,
        tolerance: float = 1e-9,
        geometry_precision: GeometryPrecisionPolicy | None = None,
        hermitian_precision: HermitianPrecisionPolicy | None = None,
    ):
        mean_ = jnp.asarray(mean)
        covariance_ = jnp.asarray(covariance, dtype=mean_.dtype)
        geometry_ = (
            GeometryPrecisionPolicy()
            if geometry_precision is None
            else geometry_precision
        )
        hermitian_ = (
            HermitianPrecisionPolicy()
            if hermitian_precision is None
            else hermitian_precision
        )
        if not isinstance(geometry_, GeometryPrecisionPolicy):
            raise TypeError("geometry_precision must be GeometryPrecisionPolicy or None.")
        if not isinstance(hermitian_, HermitianPrecisionPolicy):
            raise TypeError(
                "hermitian_precision must be HermitianPrecisionPolicy or None."
            )
        geometry_.validate_coordinates(mean_)
        if mean_.ndim != 1 or mean_.shape[0] % 2:
            raise ValueError("Gaussian mean must have even vector dimension.")
        if covariance_.shape != (mean_.shape[0], mean_.shape[0]):
            raise ValueError("Gaussian covariance shape must match the mean dimension.")
        modes = mean_.shape[0] // 2
        omega = canonical_commutation_matrix(modes, dtype=mean_.dtype)
        covariance_compute = geometry_.compute(covariance_)
        symmetric = 0.5 * (covariance_compute + covariance_compute.T)
        complex_dtype = jnp.complex64 if symmetric.dtype.itemsize <= 4 else jnp.complex128
        uncertainty = symmetric.astype(complex_dtype) + (0.5j * float(hbar) * omega)
        uncertainty_spectrum = HermitianSpectrum(
            uncertainty,
            tolerance=tolerance,
            precision=hermitian_,
        )
        margin = geometry_.decision(uncertainty_spectrum.minimum_eigenvalue)
        symplectic_matrix = hermitian_.factorization(1j * omega @ symmetric)
        eigenvalues = jnp.linalg.eigvals(symplectic_matrix)
        symplectic = geometry_.decision(jnp.sort(jnp.abs(eigenvalues))[::2])
        _, log_determinant = jnp.linalg.slogdet(
            hermitian_.factorization(2.0 * symmetric / float(hbar))
        )
        purity = geometry_.decision(jnp.exp(-0.5 * jnp.maximum(log_determinant, 0.0)))
        residual = geometry_.decision(
            jnp.max(
                jnp.abs(geometry_.accumulation(covariance_compute - covariance_compute.T))
            )
        )
        self.mean = geometry_.output(mean_)
        self.covariance = geometry_.output(symmetric)
        self.symplectic_form = geometry_.output(omega)
        self.uncertainty_margin = margin
        self.symmetry_residual = residual
        self.symplectic_eigenvalues = symplectic
        self.purity = purity
        self.valid = (
            jnp.all(jnp.isfinite(mean_))
            & jnp.all(jnp.isfinite(covariance_))
            & uncertainty_spectrum.valid
            & (residual <= tolerance)
            & (margin >= -tolerance)
        )
        self.geometry_precision = geometry_
        self.hermitian_precision = hermitian_
        self.precision_evidence = geometry_.evidence_for(
            mean_,
            children={"uncertainty-spectrum": uncertainty_spectrum.precision_evidence},
        )
        self.mode_count = modes
        self.hbar = float(hbar)


class BosonicGaussianChannel(StrictModule):
    x: Array
    y: Array
    displacement: Array
    cp_margin: Array
    valid: Array
    geometry_precision: GeometryPrecisionPolicy
    hermitian_precision: HermitianPrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    mode_count: int = eqx.field(static=True)
    hbar: float = eqx.field(static=True)
    channel_id: str = eqx.field(static=True)

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        displacement: ArrayLike,
        /,
        *,
        channel_id: str,
        hbar: float = 1.0,
        tolerance: float = 1e-9,
        geometry_precision: GeometryPrecisionPolicy | None = None,
        hermitian_precision: HermitianPrecisionPolicy | None = None,
    ):
        x_ = jnp.asarray(x)
        y_ = jnp.asarray(y, dtype=x_.dtype)
        displacement_ = jnp.asarray(displacement, dtype=x_.dtype)
        geometry_ = (
            GeometryPrecisionPolicy()
            if geometry_precision is None
            else geometry_precision
        )
        hermitian_ = (
            HermitianPrecisionPolicy()
            if hermitian_precision is None
            else hermitian_precision
        )
        if not isinstance(geometry_, GeometryPrecisionPolicy):
            raise TypeError("geometry_precision must be GeometryPrecisionPolicy or None.")
        if not isinstance(hermitian_, HermitianPrecisionPolicy):
            raise TypeError(
                "hermitian_precision must be HermitianPrecisionPolicy or None."
            )
        geometry_.validate_coordinates(x_)
        if x_.ndim != 2 or x_.shape[0] != x_.shape[1] or x_.shape[0] % 2:
            raise ValueError("Gaussian channel X must be an even square matrix.")
        if y_.shape != x_.shape or displacement_.shape != (x_.shape[0],):
            raise ValueError("Gaussian channel Y/displacement shapes do not match X.")
        modes = x_.shape[0] // 2
        omega = canonical_commutation_matrix(modes, dtype=x_.dtype)
        x_compute = geometry_.compute(x_)
        y_compute = geometry_.compute(y_)
        complex_dtype = jnp.complex64 if x_compute.dtype.itemsize <= 4 else jnp.complex128
        cp_matrix = 0.5 * (y_compute + y_compute.T).astype(complex_dtype) + 0.5j * float(
            hbar
        ) * (omega - x_compute @ omega @ x_compute.T)
        cp_spectrum = HermitianSpectrum(
            cp_matrix,
            tolerance=tolerance,
            precision=hermitian_,
        )
        margin = geometry_.decision(cp_spectrum.minimum_eigenvalue)
        self.x = geometry_.output(x_)
        self.y = geometry_.output(0.5 * (y_compute + y_compute.T))
        self.displacement = geometry_.output(displacement_)
        self.cp_margin = margin
        self.valid = (
            jnp.all(jnp.isfinite(cp_matrix)) & cp_spectrum.valid & (margin >= -tolerance)
        )
        self.geometry_precision = geometry_
        self.hermitian_precision = hermitian_
        self.precision_evidence = geometry_.evidence_for(
            x_,
            children={"cp-spectrum": cp_spectrum.precision_evidence},
        )
        self.mode_count = modes
        self.hbar = float(hbar)
        self.channel_id = str(channel_id)

    def apply(self, state: BosonicGaussianState, /) -> BosonicGaussianState:
        if state.mode_count != self.mode_count or state.hbar != self.hbar:
            raise ValueError(
                "Gaussian state and channel mode counts or hbar conventions differ."
            )
        return BosonicGaussianState(
            self.x @ state.mean + self.displacement,
            self.x @ state.covariance @ self.x.T + self.y,
            hbar=state.hbar,
            geometry_precision=self.geometry_precision,
            hermitian_precision=self.hermitian_precision,
        )

    def compose(self, after: BosonicGaussianChannel, /) -> BosonicGaussianChannel:
        if after.mode_count != self.mode_count or after.hbar != self.hbar:
            raise ValueError("Gaussian channel mode counts or hbar conventions differ.")
        return BosonicGaussianChannel(
            after.x @ self.x,
            after.x @ self.y @ after.x.T + after.y,
            after.x @ self.displacement + after.displacement,
            channel_id=f"{after.channel_id}∘{self.channel_id}",
            hbar=self.hbar,
            geometry_precision=after.geometry_precision,
            hermitian_precision=after.hermitian_precision,
        )


__all__ = [
    "BosonicGaussianChannel",
    "BosonicGaussianState",
    "canonical_commutation_matrix",
]
