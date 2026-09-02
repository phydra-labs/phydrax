#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._geometry_precision import GeometryPrecisionPolicy
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ..linalg import (
    DenseCholesky,
    DenseLinearOperator,
    FailurePolicy,
    hermitian_exp,
    hermitian_inverse_sqrt,
    hermitian_sqrt,
    HermitianPrecisionPolicy,
    HermitianSpectrum,
    HermitianSylvesterOperator,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    solve,
    TracelessHermitianSpace,
)
from ._manifold import _array_with_trailing_shape, _same_shape, AbstractRiemannianManifold


def _adjoint(value: Array, /) -> Array:
    return jnp.swapaxes(jnp.conj(value), -1, -2)


_FIXED_RANK_SOLVE = LinearSolvePolicy(
    DenseCholesky(),
    failure=FailurePolicy("status"),
)


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
    precision_evidence: PrecisionEvidenceEnvelope

    def __init__(
        self,
        density: ArrayLike,
        /,
        *,
        tolerance: float,
        precision: HermitianPrecisionPolicy | None = None,
    ):
        value = jnp.asarray(density)
        spectrum = HermitianSpectrum(
            value,
            tolerance=tolerance,
            precision=precision,
        )
        trace_residual = jnp.abs(jnp.trace(value, axis1=-2, axis2=-1) - 1.0)
        self.hermiticity_residual = spectrum.hermiticity_residual
        self.trace_residual = trace_residual
        self.minimum_eigenvalue = spectrum.minimum_eigenvalue
        self.rank = spectrum.numerical_rank
        self.condition_number = spectrum.condition_number
        self.precision_evidence = spectrum.precision_evidence
        self.valid = (
            spectrum.valid
            & (trace_residual <= tolerance)
            & (spectrum.minimum_eigenvalue > tolerance)
        )


class SLDQuantumFisherGeometry(StrictModule):
    """SLD information metric with ``g_SLD(A,B)=tr(A L_B)``."""

    tolerance: float
    precision: HermitianPrecisionPolicy
    geometry_precision: GeometryPrecisionPolicy

    def __init__(
        self,
        *,
        tolerance: float = 1e-10,
        precision: HermitianPrecisionPolicy | None = None,
        geometry_precision: GeometryPrecisionPolicy | None = None,
    ):
        precision_ = HermitianPrecisionPolicy() if precision is None else precision
        geometry_ = (
            GeometryPrecisionPolicy()
            if geometry_precision is None
            else geometry_precision
        )
        if not isinstance(precision_, HermitianPrecisionPolicy):
            raise TypeError("precision must be a HermitianPrecisionPolicy or None.")
        if not isinstance(geometry_, GeometryPrecisionPolicy):
            raise TypeError(
                "geometry_precision must be a GeometryPrecisionPolicy or None."
            )
        self.tolerance = float(tolerance)
        self.precision = precision_
        self.geometry_precision = geometry_

    def sld(self, density: ArrayLike, tangent: ArrayLike, /) -> Array:
        operator = HermitianSylvesterOperator(
            density,
            tolerance=self.tolerance,
            precision=self.precision,
        )
        result = operator.solve(2.0 * self.precision.factorization(tangent))
        return result.value

    def inner(
        self,
        density: ArrayLike,
        left: ArrayLike,
        right: ArrayLike,
        /,
    ) -> Array:
        value = jnp.trace(
            _adjoint(self.geometry_precision.accumulation(left))
            @ self.geometry_precision.accumulation(self.sld(density, right))
        )
        return self.geometry_precision.decision(jnp.real(value))

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
    precision: GeometryPrecisionPolicy
    hermitian_precision: HermitianPrecisionPolicy

    def __init__(
        self,
        dimension: int,
        /,
        *,
        tolerance: float = 1e-9,
        precision: GeometryPrecisionPolicy | None = None,
        hermitian_precision: HermitianPrecisionPolicy | None = None,
    ):
        dimension_ = int(dimension)
        if dimension_ < 2:
            raise ValueError("Density dimension must be at least two.")
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        precision_ = GeometryPrecisionPolicy() if precision is None else precision
        hermitian_ = (
            HermitianPrecisionPolicy()
            if hermitian_precision is None
            else hermitian_precision
        )
        if not isinstance(precision_, GeometryPrecisionPolicy):
            raise TypeError("precision must be a GeometryPrecisionPolicy or None.")
        if not isinstance(hermitian_, HermitianPrecisionPolicy):
            raise TypeError(
                "hermitian_precision must be a HermitianPrecisionPolicy or None."
            )
        self.dimension = dimension_
        self.tolerance = float(tolerance)
        self.manifold_id = f"manifold:density:bures:{dimension_}"
        self.point_shape = (dimension_, dimension_)
        self.retraction_method = "normalized-affine-exponential"
        self.transport_method = "traceless-hermitian-projection"
        self.transport_is_isometric = False
        self.transport_is_parallel = False
        self.precision = precision_
        self.hermitian_precision = hermitian_
        self.sld_geometry = SLDQuantumFisherGeometry(
            tolerance=tolerance,
            precision=hermitian_,
            geometry_precision=precision_,
        )
        self.tangent_space = TracelessHermitianSpace(dimension_)

    @property
    def scalar_field(self) -> str:
        return "complex"

    def _matrix(self, value: ArrayLike, name: str, /) -> Array:
        return _array_with_trailing_shape(value, self.point_shape, name)

    def contains(self, point: ArrayLike, /) -> Array:
        return FaithfulDensityReport(
            self._matrix(point, "density"),
            tolerance=self.tolerance,
            precision=self.hermitian_precision,
        ).valid

    def constraint_residual(self, point: ArrayLike, /) -> Array:
        report = FaithfulDensityReport(
            self._matrix(point, "density"),
            tolerance=self.tolerance,
            precision=self.hermitian_precision,
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
            4.0 * self.sld_geometry.sharp(rho, jnp.conj(ambient_cotangent))
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
        root = hermitian_sqrt(
            rho,
            tolerance=self.tolerance,
            precision=self.hermitian_precision,
        ).value
        inverse_root = hermitian_inverse_sqrt(
            rho,
            tolerance=self.tolerance,
            precision=self.hermitian_precision,
        ).value
        local = inverse_root @ step @ inverse_root
        candidate = (
            root
            @ hermitian_exp(
                local,
                precision=self.hermitian_precision,
            ).value
            @ root
        )
        candidate = 0.5 * (candidate + _adjoint(candidate))
        normalized = candidate / jnp.trace(candidate)
        return jnp.asarray(normalized, dtype=rho.dtype)

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


class FixedRankDensityManifold(AbstractRiemannianManifold):
    """Purification-factor geometry for one fixed density-rank stratum."""

    dimension: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    manifold_id: str = eqx.field(static=True)
    point_shape: tuple[int, int] = eqx.field(static=True)
    retraction_method: str = eqx.field(static=True)
    transport_method: str = eqx.field(static=True)
    transport_is_isometric: bool = eqx.field(static=True)
    transport_is_parallel: bool = eqx.field(static=True)

    def __init__(self, dimension: int, rank: int, /, *, tolerance: float = 1e-9):
        dimension_ = int(dimension)
        rank_ = int(rank)
        if not 1 <= rank_ <= dimension_:
            raise ValueError("rank must lie in [1, dimension].")
        if not isfinite(tolerance) or float(tolerance) <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        self.dimension = dimension_
        self.rank = rank_
        self.tolerance = float(tolerance)
        self.manifold_id = f"manifold:density:fixed-rank:{dimension_}:{rank_}"
        self.point_shape = (dimension_, rank_)
        self.retraction_method = "normalized-factor"
        self.transport_method = "horizontal-projection"
        self.transport_is_isometric = False
        self.transport_is_parallel = False

    @property
    def scalar_field(self) -> str:
        return "complex"

    def _factor(self, factor: ArrayLike, name: str, /) -> Array:
        value = _array_with_trailing_shape(factor, self.point_shape, name)
        if not jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError(f"{name} must use complex floating-point coordinates.")
        return value

    def density(self, factor: ArrayLike, /) -> Array:
        value = self._factor(factor, "Fixed-rank density factor")
        density = value @ _adjoint(value)
        return density / jnp.trace(density, axis1=-2, axis2=-1)[..., None, None]

    from_factor = density

    def _spectrum(self, factor: Array, /) -> Array:
        return jnp.linalg.eigvalsh(_adjoint(factor) @ factor)

    def contains(self, point: ArrayLike, /) -> Array:
        factor = self._factor(point, "Fixed-rank density factor")
        spectrum = self._spectrum(factor)
        norm = jnp.real(jnp.vdot(factor, factor))
        return (
            jnp.all(jnp.isfinite(factor))
            & (jnp.min(spectrum) > self.tolerance)
            & (jnp.abs(norm - 1.0) <= self.tolerance)
        )

    def constraint_residual(self, point: ArrayLike, /) -> Array:
        factor = self._factor(point, "Fixed-rank density factor")
        spectrum = self._spectrum(factor)
        return jnp.maximum(
            jnp.abs(jnp.real(jnp.vdot(factor, factor)) - 1.0),
            jnp.maximum(self.tolerance - jnp.min(spectrum), 0.0),
        )

    def project_tangent(self, point: ArrayLike, ambient_vector: ArrayLike, /) -> Array:
        factor = self._factor(point, "Fixed-rank density factor")
        vector = self._factor(ambient_vector, "Fixed-rank density tangent")
        _same_shape(vector, factor, "Fixed-rank density tangent")
        radial = jnp.real(jnp.vdot(factor, vector))
        sphere_tangent = vector - radial * factor
        gram = _adjoint(factor) @ factor
        right_hand_side = (
            _adjoint(factor) @ sphere_tangent - _adjoint(sphere_tangent) @ factor
        )
        solution = HermitianSylvesterOperator(
            gram,
            tolerance=self.tolerance,
        ).solve(right_hand_side)
        generator = eqx.error_if(
            solution.value,
            jnp.any(~solution.valid),
            "Fixed-rank horizontal projection Sylvester solve failed.",
        )
        return sphere_tangent - factor @ generator

    def egrad_to_rgrad(self, point: ArrayLike, ambient_cotangent: ArrayLike, /) -> Array:
        return self.project_tangent(point, jnp.conj(ambient_cotangent))

    def inner(
        self,
        point: ArrayLike,
        left_tangent: ArrayLike,
        right_tangent: ArrayLike,
        /,
    ) -> Array:
        left = self.project_tangent(point, left_tangent)
        right = self.project_tangent(point, right_tangent)
        return 4.0 * jnp.real(jnp.vdot(left, right))

    def retract(self, point: ArrayLike, tangent_step: ArrayLike, /) -> Array:
        factor = self._factor(point, "Fixed-rank density factor")
        candidate = factor + self.project_tangent(factor, tangent_step)
        candidate = candidate / jnp.sqrt(jnp.real(jnp.vdot(candidate, candidate)))
        minimum = jnp.min(self._spectrum(candidate))
        return eqx.error_if(
            candidate,
            minimum <= self.tolerance,
            "Fixed-rank retraction lost the declared support gap.",
        )

    def transport(
        self,
        point: ArrayLike,
        tangent_step: ArrayLike,
        destination: ArrayLike,
        tangent: ArrayLike,
        /,
    ) -> Array:
        del point, tangent_step
        return self.project_tangent(destination, tangent)

    def rank_residual(self, density: ArrayLike, /) -> Array:
        spectrum = HermitianSpectrum(density, tolerance=self.tolerance)
        return jnp.abs(spectrum.numerical_rank - self.rank)


class RankStratumEvidence(StrictModule):
    rank: Array
    threshold: Array
    support_gap: Array
    discarded_mass: Array
    ambiguous: Array
    valid: Array

    def __init__(
        self,
        *,
        rank: ArrayLike,
        threshold: ArrayLike,
        support_gap: ArrayLike,
        discarded_mass: ArrayLike,
        ambiguous: ArrayLike,
        valid: ArrayLike,
    ):
        self.rank = jnp.asarray(rank, dtype=jnp.int32)
        self.threshold = jnp.asarray(threshold)
        self.support_gap = jnp.asarray(support_gap)
        self.discarded_mass = jnp.asarray(discarded_mass)
        self.ambiguous = jnp.asarray(ambiguous, dtype=bool)
        self.valid = jnp.asarray(valid, dtype=bool)


class RankTransitionProposal(StrictModule):
    factor: Array
    source_rank: int = eqx.field(static=True)
    target_rank: int = eqx.field(static=True)
    trigger: str = eqx.field(static=True)
    state_transfer: str = eqx.field(static=True)
    discarded_mass: Array
    valid: Array

    def __init__(
        self,
        factor: ArrayLike,
        /,
        *,
        source_rank: int,
        target_rank: int,
        trigger: str,
        state_transfer: str,
        discarded_mass: ArrayLike,
        valid: ArrayLike,
    ):
        self.factor = jnp.asarray(factor)
        self.source_rank = int(source_rank)
        self.target_rank = int(target_rank)
        self.trigger = str(trigger)
        self.state_transfer = str(state_transfer)
        self.discarded_mass = jnp.asarray(discarded_mass)
        self.valid = jnp.asarray(valid, dtype=bool)


class DensityRankStratification(StrictModule):
    """Explicit host classifier and epoch-boundary rank-transition proposals."""

    dimension: int = eqx.field(static=True)
    absolute_threshold: float = eqx.field(static=True)
    relative_threshold: float = eqx.field(static=True)
    ambiguity_factor: float = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        /,
        *,
        absolute_threshold: float = 1e-10,
        relative_threshold: float = 1e-8,
        ambiguity_factor: float = 2.0,
    ):
        if int(dimension) < 2:
            raise ValueError("Density dimension must be at least two.")
        if (
            min(
                float(absolute_threshold),
                float(relative_threshold),
                float(ambiguity_factor) - 1.0,
            )
            <= 0.0
        ):
            raise ValueError("Rank thresholds must be positive and ambiguity_factor > 1.")
        self.dimension = int(dimension)
        self.absolute_threshold = float(absolute_threshold)
        self.relative_threshold = float(relative_threshold)
        self.ambiguity_factor = float(ambiguity_factor)

    def classify(self, density: ArrayLike, /) -> RankStratumEvidence:
        value = jnp.asarray(density)
        if value.shape != (self.dimension, self.dimension):
            raise ValueError("Density classifier input has the wrong shape.")
        adjoint = _adjoint(value)
        hermiticity_residual = jnp.max(jnp.abs(value - adjoint))
        trace_residual = jnp.abs(jnp.trace(value) - 1.0)
        eigenvalues = jnp.linalg.eigvalsh(0.5 * (value + adjoint))
        largest = jnp.max(eigenvalues)
        threshold = jnp.maximum(
            self.absolute_threshold, self.relative_threshold * largest
        )
        active = eigenvalues > threshold
        rank = jnp.sum(active, dtype=jnp.int32)
        ratio = eigenvalues / threshold
        ambiguous = jnp.any(
            (ratio >= 1.0 / self.ambiguity_factor) & (ratio <= self.ambiguity_factor)
        )
        support_gap = jnp.min(jnp.where(active, eigenvalues - threshold, jnp.inf))
        discarded = jnp.sum(jnp.where(active, 0.0, jnp.maximum(eigenvalues, 0.0)))
        valid = (
            jnp.all(jnp.isfinite(value))
            & (hermiticity_residual <= self.absolute_threshold)
            & (trace_residual <= self.absolute_threshold)
            & (jnp.min(eigenvalues) >= -self.absolute_threshold)
            & ~ambiguous
            & (rank > 0)
            & jnp.all(jnp.isfinite(eigenvalues))
        )
        return RankStratumEvidence(
            rank=rank,
            threshold=threshold,
            support_gap=support_gap,
            discarded_mass=discarded,
            ambiguous=ambiguous,
            valid=valid,
        )

    def propose_transition(
        self,
        factor: ArrayLike,
        target_rank: int,
        /,
        *,
        trigger: str,
        embedding: ArrayLike | None = None,
    ) -> RankTransitionProposal:
        source = jnp.asarray(factor)
        if source.ndim != 2 or source.shape[0] != self.dimension:
            raise ValueError("Transition source must be a two-dimensional factor.")
        source_rank = source.shape[1]
        target = int(target_rank)
        if not 1 <= target <= self.dimension or target == source_rank:
            raise ValueError("target_rank must select a different supported stratum.")
        u, singular, _ = jnp.linalg.svd(source, full_matrices=False)
        if target < source_rank:
            proposed = u[:, :target] * singular[:target]
            discarded = jnp.sum(singular[target:] ** 2)
            transfer = "truncate-horizontal-state"
        else:
            if embedding is None:
                raise ValueError("Upward rank transition requires a declared embedding.")
            extra = jnp.asarray(embedding, dtype=source.dtype)
            if extra.shape != (self.dimension, target - source_rank):
                raise ValueError("Rank-transition embedding has the wrong shape.")
            gram = _adjoint(source) @ source
            operator = DenseLinearOperator(
                gram,
                properties=OperatorProperties(
                    self_adjoint=True,
                    positive_definite=True,
                    evidence={"positive_definite": "asserted"},
                ),
                operator_id="density-rank-transition:source-gram",
            )
            coordinates = solve(
                LinearSystem(
                    operator,
                    problem_id="density-rank-transition:embedding-projection",
                ),
                _adjoint(source) @ extra,
                policy=_FIXED_RANK_SOLVE,
            ).value
            projected = extra - source @ coordinates
            proposed = jnp.concatenate((source, projected), axis=1)
            discarded = jnp.asarray(0.0, dtype=source.real.dtype)
            transfer = "embed-horizontal-state-with-zero-new-components"
        proposed = proposed / jnp.sqrt(jnp.real(jnp.vdot(proposed, proposed)))
        gap = jnp.min(jnp.linalg.eigvalsh(_adjoint(proposed) @ proposed))
        return RankTransitionProposal(
            proposed,
            source_rank=source_rank,
            target_rank=target,
            trigger=trigger,
            state_transfer=transfer,
            discarded_mass=discarded,
            valid=jnp.isfinite(gap) & (gap > self.absolute_threshold),
        )


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
    "DensityRankStratification",
    "FixedRankDensityManifold",
    "RankStratumEvidence",
    "RankTransitionProposal",
    "SLDQuantumFisherGeometry",
    "UhlmannAlignment",
    "bures_squared_distance",
    "density_fidelity",
    "faithful_density_from_cholesky",
    "faithful_density_from_generator",
    "principal_purification",
    "uhlmann_alignment",
]
