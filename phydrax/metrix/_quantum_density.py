#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..linalg import (
    hermitian_exp,
    hermitian_inverse_sqrt,
    hermitian_sqrt,
    HermitianSpectrum,
    HermitianSylvesterOperator,
    TracelessHermitianSpace,
)
from ._manifold import _array_with_trailing_shape, _same_shape, AbstractRiemannianManifold


def _adjoint(value: Array, /) -> Array:
    return jnp.swapaxes(jnp.conj(value), -1, -2)


def faithful_density_from_cholesky(factor: ArrayLike, /) -> Array:
    value = jnp.asarray(factor)
    if value.ndim < 2 or value.shape[-2] != value.shape[-1]:
        raise ValueError("Density Cholesky factors must be square matrices.")
    positive = value @ _adjoint(value)
    return positive / jnp.trace(positive, axis1=-2, axis2=-1)[..., None, None]


def faithful_density_from_generator(generator: ArrayLike, /) -> Array:
    value = jnp.asarray(generator)
    if value.ndim < 2 or value.shape[-2] != value.shape[-1]:
        raise ValueError("Density generators must be square matrices.")
    hermitian = 0.5 * (value + _adjoint(value))
    dimension = value.shape[-1]
    trace = jnp.trace(hermitian, axis1=-2, axis2=-1) / float(dimension)
    centered = hermitian - trace[..., None, None] * jnp.eye(dimension, dtype=value.dtype)
    exponential = hermitian_exp(centered).value
    return exponential / jnp.trace(exponential, axis1=-2, axis2=-1)[..., None, None]


class FaithfulDensityReport(StrictModule):
    valid: Array
    hermiticity_residual: Array
    trace_residual: Array
    minimum_eigenvalue: Array
    rank: Array
    condition_number: Array

    def __init__(self, density: ArrayLike, /, *, tolerance: float):
        value = jnp.asarray(density)
        spectrum = HermitianSpectrum(value, tolerance=tolerance)
        trace_residual = jnp.abs(jnp.trace(value, axis1=-2, axis2=-1) - 1.0)
        self.hermiticity_residual = spectrum.hermiticity_residual
        self.trace_residual = trace_residual
        self.minimum_eigenvalue = spectrum.minimum_eigenvalue
        self.rank = spectrum.numerical_rank
        self.condition_number = spectrum.condition_number
        self.valid = (
            spectrum.valid
            & (trace_residual <= tolerance)
            & (spectrum.minimum_eigenvalue > tolerance)
        )


class SLDQuantumFisherGeometry(StrictModule):
    """SLD information metric with ``g_SLD(A,B)=tr(A L_B)``."""

    tolerance: float

    def __init__(self, *, tolerance: float = 1e-10):
        self.tolerance = float(tolerance)

    def sld(self, density: ArrayLike, tangent: ArrayLike, /) -> Array:
        operator = HermitianSylvesterOperator(density, tolerance=self.tolerance)
        result = operator.solve(2.0 * jnp.asarray(tangent))
        return result.value

    def inner(
        self,
        density: ArrayLike,
        left: ArrayLike,
        right: ArrayLike,
        /,
    ) -> Array:
        return jnp.real(jnp.trace(_adjoint(jnp.asarray(left)) @ self.sld(density, right)))

    def flat(self, density: ArrayLike, tangent: ArrayLike, /) -> Array:
        return self.sld(density, tangent)

    def sharp(self, density: ArrayLike, cotangent: ArrayLike, /) -> Array:
        rho = jnp.asarray(density)
        covector = 0.5 * (jnp.asarray(cotangent) + _adjoint(jnp.asarray(cotangent)))
        expectation = jnp.real(jnp.trace(rho @ covector))
        identity = jnp.eye(rho.shape[-1], dtype=rho.dtype)
        centered = covector - expectation * identity
        return 0.5 * (rho @ centered + centered @ rho)


class BuresDensityManifold(AbstractRiemannianManifold):
    """Full-rank trace-one density matrices with the Bures metric."""

    dimension: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    manifold_id: str = eqx.field(static=True)
    point_shape: tuple[int, int] = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    transport_method: str = eqx.field(static=True)
    transport_is_isometric: bool = eqx.field(static=True)
    transport_is_parallel: bool = eqx.field(static=True)
    sld_geometry: SLDQuantumFisherGeometry
    tangent_space: TracelessHermitianSpace

    def __init__(self, dimension: int, /, *, tolerance: float = 1e-9):
        dimension_ = int(dimension)
        if dimension_ < 2:
            raise ValueError("Density dimension must be at least two.")
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        self.dimension = dimension_
        self.tolerance = float(tolerance)
        self.manifold_id = f"manifold:density:bures:{dimension_}"
        self.point_shape = (dimension_, dimension_)
        self.retraction_method = "normalized-affine-exponential"
        self.transport_method = "traceless-hermitian-projection"
        self.transport_is_isometric = False
        self.transport_is_parallel = False
        self.sld_geometry = SLDQuantumFisherGeometry(tolerance=tolerance)
        self.tangent_space = TracelessHermitianSpace(dimension_)

    @property
    def scalar_field(self) -> str:
        return "complex"

    def _matrix(self, value: ArrayLike, name: str, /) -> Array:
        return _array_with_trailing_shape(value, self.point_shape, name)

    def contains(self, point: ArrayLike, /) -> Array:
        return FaithfulDensityReport(
            self._matrix(point, "density"), tolerance=self.tolerance
        ).valid

    def constraint_residual(self, point: ArrayLike, /) -> Array:
        report = FaithfulDensityReport(
            self._matrix(point, "density"), tolerance=self.tolerance
        )
        return jnp.maximum(
            jnp.maximum(report.hermiticity_residual, report.trace_residual),
            jnp.maximum(self.tolerance - report.minimum_eigenvalue, 0.0),
        )

    def project_tangent(self, point: ArrayLike, ambient_vector: ArrayLike, /) -> Array:
        self._matrix(point, "density")
        return self.tangent_space.project(ambient_vector)

    def egrad_to_rgrad(self, point: ArrayLike, ambient_cotangent: ArrayLike, /) -> Array:
        rho = self._matrix(point, "density")
        # Bures sharp is four times the SLD sharp under g_B = g_SLD / 4.
        return self.tangent_space.project(
            4.0 * self.sld_geometry.sharp(rho, ambient_cotangent)
        )

    def inner(
        self,
        point: ArrayLike,
        left_tangent: ArrayLike,
        right_tangent: ArrayLike,
        /,
    ) -> Array:
        rho = self._matrix(point, "density")
        left = self.project_tangent(rho, left_tangent)
        right = self.project_tangent(rho, right_tangent)
        return 0.25 * self.sld_geometry.inner(rho, left, right)

    def retract(self, point: ArrayLike, tangent_step: ArrayLike, /) -> Array:
        rho = self._matrix(point, "density")
        step = self.project_tangent(rho, tangent_step)
        root = hermitian_sqrt(rho, tolerance=self.tolerance).value
        inverse_root = hermitian_inverse_sqrt(rho, tolerance=self.tolerance).value
        local = inverse_root @ step @ inverse_root
        candidate = root @ hermitian_exp(local).value @ root
        candidate = 0.5 * (candidate + _adjoint(candidate))
        return candidate / jnp.trace(candidate)

    def transport(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        destination: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        self.project_tangent(point, tangent_step)
        target = self._matrix(destination, "density destination")
        vector = self._matrix(tangent, "density tangent")
        _same_shape(target, vector, "density tangent")
        return self.project_tangent(target, vector)


def density_fidelity(left: ArrayLike, right: ArrayLike, /) -> Array:
    rho = jnp.asarray(left)
    sigma = jnp.asarray(right)
    if rho.shape != sigma.shape or rho.ndim != 2:
        raise ValueError("Density fidelity requires equal square matrices.")
    root = hermitian_sqrt(rho).value
    middle = hermitian_sqrt(root @ sigma @ root).value
    return jnp.real(jnp.trace(middle)) ** 2


def bures_squared_distance(left: ArrayLike, right: ArrayLike, /) -> Array:
    fidelity = density_fidelity(left, right)
    return 2.0 * (1.0 - jnp.sqrt(jnp.clip(fidelity, 0.0, 1.0)))


def principal_purification(density: ArrayLike, /) -> Array:
    return hermitian_sqrt(density).value


class FixedRankDensityStratum(StrictModule):
    dimension: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)

    def __init__(self, dimension: int, rank: int, /, *, tolerance: float = 1e-9):
        dimension_ = int(dimension)
        rank_ = int(rank)
        if not 1 <= rank_ <= dimension_:
            raise ValueError("rank must lie in [1, dimension].")
        self.dimension = dimension_
        self.rank = rank_
        self.tolerance = float(tolerance)

    def from_factor(self, factor: ArrayLike, /) -> Array:
        value = jnp.asarray(factor)
        if value.shape != (self.dimension, self.rank):
            raise ValueError("Fixed-rank factor has the wrong shape.")
        density = value @ _adjoint(value)
        return density / jnp.trace(density)

    def rank_residual(self, density: ArrayLike, /) -> Array:
        spectrum = HermitianSpectrum(density, tolerance=self.tolerance)
        return jnp.abs(spectrum.numerical_rank - self.rank)


class UhlmannAlignment(StrictModule):
    unitary: Array
    overlap: Array
    residual: Array
    valid: Array

    def __init__(self, unitary: ArrayLike, overlap: ArrayLike, residual: ArrayLike, /):
        self.unitary = jnp.asarray(unitary)
        self.overlap = jnp.asarray(overlap)
        self.residual = jnp.asarray(residual)
        self.valid = jnp.all(jnp.isfinite(self.unitary)) & jnp.isfinite(self.residual)


def uhlmann_alignment(
    left_amplitude: ArrayLike, right_amplitude: ArrayLike, /
) -> UhlmannAlignment:
    left = jnp.asarray(left_amplitude)
    right = jnp.asarray(right_amplitude)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Purification amplitudes must be equal square matrices.")
    product = _adjoint(right) @ left
    u, _, vh = jnp.linalg.svd(product, full_matrices=False)
    unitary = u @ vh
    aligned = right @ unitary
    residual = jnp.linalg.norm(_adjoint(left) @ aligned - _adjoint(aligned) @ left)
    overlap = jnp.trace(_adjoint(left) @ aligned)
    return UhlmannAlignment(unitary, overlap, residual)


__all__ = [
    "BuresDensityManifold",
    "FaithfulDensityReport",
    "FixedRankDensityStratum",
    "SLDQuantumFisherGeometry",
    "UhlmannAlignment",
    "bures_squared_distance",
    "density_fidelity",
    "faithful_density_from_cholesky",
    "faithful_density_from_generator",
    "principal_purification",
    "uhlmann_alignment",
]
