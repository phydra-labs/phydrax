#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._base import AbstractPositiveDefiniteKernel


CompactSpace = Literal["so", "su", "stiefel", "grassmann"]


class PreparedCompactHomogeneousSpectrum(StrictModule):
    """Finite representation frontier with a separately certified tail bound."""

    labels: Array
    casimir_eigenvalues: Array
    multiplicities: Array
    zonal_evaluator: Callable[[Array, Array], Array]
    tail_bound: Array
    space: CompactSpace = eqx.field(static=True)
    frontier: int = eqx.field(static=True)
    tail_certified: bool = eqx.field(static=True)
    spectrum_id: str = eqx.field(static=True)

    def __init__(
        self,
        labels: ArrayLike,
        casimir_eigenvalues: ArrayLike,
        multiplicities: ArrayLike,
        zonal_evaluator: Callable[[Array, Array], Array],
        /,
        *,
        space: CompactSpace,
        tail_bound: ArrayLike,
        tail_certified: bool,
        spectrum_id: str,
    ):
        labels_ = jnp.asarray(labels, dtype=jnp.int32)
        eigenvalues = jnp.asarray(casimir_eigenvalues)
        multiplicities_ = jnp.asarray(multiplicities)
        if (
            labels_.ndim != 2
            or eigenvalues.shape != labels_.shape[:1]
            or multiplicities_.shape != eigenvalues.shape
            or eigenvalues.size == 0
        ):
            raise ValueError(
                "Compact spectrum labels/eigenvalues/multiplicities are incompatible."
            )
        if space not in ("so", "su", "stiefel", "grassmann") or not callable(
            zonal_evaluator
        ):
            raise ValueError("Compact spectrum space/evaluator are unsupported.")
        if bool(jnp.any(eigenvalues < 0.0)) or bool(jnp.any(multiplicities_ <= 0.0)):
            raise ValueError(
                "Casimir eigenvalues and multiplicities must be nonnegative/positive."
            )
        bound = jnp.asarray(tail_bound, dtype=eigenvalues.dtype)
        if bound.shape != () or not bool(jnp.isfinite(bound)) or bool(bound < 0.0):
            raise ValueError(
                "Compact spectrum tail bound must be one finite nonnegative scalar."
            )
        if not bool(tail_certified):
            raise ValueError("Compact PSD spectrum requires a certified truncation tail.")
        self.labels = labels_
        self.casimir_eigenvalues = eigenvalues
        self.multiplicities = multiplicities_
        self.zonal_evaluator = zonal_evaluator
        self.tail_bound = bound
        self.space = space
        self.frontier = int(eigenvalues.shape[0])
        self.tail_certified = True
        self.spectrum_id = str(spectrum_id)

    def zonal(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        values = jnp.asarray(self.zonal_evaluator(jnp.asarray(left), jnp.asarray(right)))
        if values.shape != (self.frontier,):
            raise ValueError("Zonal evaluator must return one value per prepared mode.")
        return values


class KernelEvaluationEvidence(StrictModule):
    truncation_tail_bound: Array
    membership_valid: Array
    branch_valid: Array
    finite: Array
    positive_definite_capability: Array

    def __init__(
        self,
        *,
        truncation_tail_bound: ArrayLike,
        membership_valid: ArrayLike,
        branch_valid: ArrayLike,
        finite: ArrayLike,
        positive_definite_capability: ArrayLike,
    ):
        self.truncation_tail_bound = jnp.asarray(truncation_tail_bound)
        self.membership_valid = jnp.asarray(membership_valid, dtype=bool)
        self.branch_valid = jnp.asarray(branch_valid, dtype=bool)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.positive_definite_capability = jnp.asarray(
            positive_definite_capability, dtype=bool
        )


class _CompactHomogeneousSpectralKernel(AbstractPositiveDefiniteKernel):
    __strict_abstract__ = True

    spectrum: PreparedCompactHomogeneousSpectrum
    weights: Array
    normalize: bool = eqx.field(static=True)
    family: str = eqx.field(static=True)

    def __init__(
        self,
        spectrum: PreparedCompactHomogeneousSpectrum,
        weights: ArrayLike,
        /,
        *,
        normalize: bool,
        family: str,
    ):
        values = jnp.asarray(weights, dtype=spectrum.casimir_eigenvalues.dtype)
        if (
            values.shape != (spectrum.frontier,)
            or bool(jnp.any(values < 0.0))
            or bool(jnp.any(~jnp.isfinite(values)))
        ):
            raise ValueError(
                "Compact spectral kernel weights must be finite and nonnegative."
            )
        if bool(normalize):
            total = jnp.sum(values)
            if not bool(total > 0.0):
                raise ValueError("Normalized compact spectral weights cannot all vanish.")
            values = values / total
        self.spectrum = spectrum
        self.weights = values
        self.normalize = bool(normalize)
        self.family = str(family)

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        value = jnp.sum(self.weights * self.spectrum.zonal(left, right))
        return eqx.error_if(
            value,
            ~jnp.isfinite(value),
            "Compact spectral kernel evaluation is nonfinite.",
        )

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        if left_.ndim < 2 or right_.ndim < 2:
            raise ValueError("Compact kernel matrix inputs require a leading point axis.")
        return jax.vmap(
            lambda first: jax.vmap(lambda second: self.pairwise(first, second))(right_)
        )(left_)

    def diagonal(self, points: ArrayLike, /) -> Array:
        values = jnp.asarray(points)
        return jax.vmap(lambda point: self.pairwise(point, point))(values)

    @property
    def input_ndim(self) -> int:
        return 2

    @property
    def max_derivative_order(self) -> None:
        return None

    @property
    def is_unit_diagonal(self) -> bool:
        return self.normalize

    @property
    def kernel_id(self) -> str:
        return (
            f"{self.family}[{self.spectrum.spectrum_id};"
            f"frontier={self.spectrum.frontier};"
            f"normalize={int(self.normalize)}]"
        )

    def evidence(self, value: ArrayLike, /) -> KernelEvaluationEvidence:
        finite = jnp.all(jnp.isfinite(jnp.asarray(value)))
        return KernelEvaluationEvidence(
            truncation_tail_bound=self.spectrum.tail_bound,
            membership_valid=True,
            branch_valid=True,
            finite=finite,
            positive_definite_capability=True,
        )


class CompactHomogeneousHeatKernel(_CompactHomogeneousSpectralKernel):
    time: float = eqx.field(static=True)

    def __init__(
        self,
        spectrum: PreparedCompactHomogeneousSpectrum,
        /,
        *,
        time: float,
        normalize: bool = True,
    ):
        if float(time) <= 0.0:
            raise ValueError("Heat-kernel time must be positive.")
        weights = spectrum.multiplicities * jnp.exp(
            -float(time) * spectrum.casimir_eigenvalues
        )
        super().__init__(
            spectrum, weights, normalize=normalize, family="compact-homogeneous-heat"
        )
        self.time = float(time)


class CompactHomogeneousMaternKernel(_CompactHomogeneousSpectralKernel):
    smoothness: float = eqx.field(static=True)
    inverse_length_squared: float = eqx.field(static=True)
    spectral_dimension: float = eqx.field(static=True)

    def __init__(
        self,
        spectrum: PreparedCompactHomogeneousSpectrum,
        /,
        *,
        smoothness: float,
        inverse_length_squared: float,
        spectral_dimension: float,
        normalize: bool = True,
    ):
        if (
            min(
                float(smoothness),
                float(inverse_length_squared),
                float(spectral_dimension),
            )
            <= 0.0
        ):
            raise ValueError(
                "Matérn smoothness, inverse length, and spectral dimension must be positive."
            )
        exponent = float(smoothness) + 0.5 * float(spectral_dimension)
        weights = spectrum.multiplicities * (
            float(inverse_length_squared) + spectrum.casimir_eigenvalues
        ) ** (-exponent)
        super().__init__(
            spectrum, weights, normalize=normalize, family="compact-homogeneous-matern"
        )
        self.smoothness = float(smoothness)
        self.inverse_length_squared = float(inverse_length_squared)
        self.spectral_dimension = float(spectral_dimension)


class GeodesicDistanceEvidence(StrictModule):
    distance: Array
    branch_margin: Array
    log_residual: Array
    membership_valid: Array
    branch_valid: Array
    valid: Array

    def __init__(
        self,
        *,
        distance: ArrayLike,
        branch_margin: ArrayLike,
        log_residual: ArrayLike,
        membership_valid: ArrayLike,
        branch_valid: ArrayLike,
        valid: ArrayLike,
    ):
        self.distance = jnp.asarray(distance)
        self.branch_margin = jnp.asarray(branch_margin)
        self.log_residual = jnp.asarray(log_residual)
        self.membership_valid = jnp.asarray(membership_valid, dtype=bool)
        self.branch_valid = jnp.asarray(branch_valid, dtype=bool)
        self.valid = jnp.asarray(valid, dtype=bool)


class GeodesicRadialKernel(StrictModule):
    """Explicit geodesic radial kernel with no automatic PSD capability."""

    __strict_abstract__ = True

    radial_function: Callable[[Array], Array]
    space: CompactSpace = eqx.field(static=True)
    membership_tolerance: float = eqx.field(static=True)
    branch_tolerance: float = eqx.field(static=True)
    stiefel_log: Callable[[Array, Array], tuple[Array, Array]] | None
    positive_definite_capability: bool = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        radial_function: Callable[[Array], Array],
        /,
        *,
        space: CompactSpace,
        membership_tolerance: float = 1e-6,
        branch_tolerance: float = 1e-6,
        stiefel_log: Callable[[Array, Array], tuple[Array, Array]] | None = None,
        positive_definite_theorem: bool = False,
        kernel_id: str = "geodesic-radial",
    ):
        if not callable(radial_function) or space not in (
            "so",
            "su",
            "stiefel",
            "grassmann",
        ):
            raise ValueError("Geodesic radial function/space are invalid.")
        if min(float(membership_tolerance), float(branch_tolerance)) <= 0.0:
            raise ValueError("Geodesic tolerances must be positive.")
        if space == "stiefel" and stiefel_log is None:
            raise ValueError(
                "Stiefel geodesic kernels require a declared bounded shooting/log provider."
            )
        self.radial_function = radial_function
        self.space = space
        self.membership_tolerance = float(membership_tolerance)
        self.branch_tolerance = float(branch_tolerance)
        self.stiefel_log = stiefel_log
        self.positive_definite_capability = bool(positive_definite_theorem)
        self.kernel_id = str(kernel_id)

    def distance_evidence(
        self, left: ArrayLike, right: ArrayLike, /
    ) -> GeodesicDistanceEvidence:
        first = jnp.asarray(left)
        second = jnp.asarray(right, dtype=first.dtype)
        if first.shape != second.shape or first.ndim != 2:
            raise ValueError("Compact geodesic points must be equal-shaped matrices.")
        identity_left = jnp.eye(first.shape[1], dtype=first.dtype)
        gram_first = jnp.swapaxes(jnp.conj(first), -1, -2) @ first
        gram_second = jnp.swapaxes(jnp.conj(second), -1, -2) @ second
        membership = (
            jnp.max(jnp.abs(gram_first - identity_left)) <= self.membership_tolerance
        )
        membership = membership & (
            jnp.max(jnp.abs(gram_second - identity_left)) <= self.membership_tolerance
        )
        residual = jnp.asarray(0.0, dtype=first.real.dtype)
        if self.space in ("so", "su"):
            if first.shape[0] != first.shape[1]:
                raise ValueError("SO/SU geodesic points must be square.")
            membership = membership & (
                jnp.abs(jnp.linalg.det(first) - 1.0) <= self.membership_tolerance
            )
            membership = membership & (
                jnp.abs(jnp.linalg.det(second) - 1.0) <= self.membership_tolerance
            )
            if self.space == "so":
                membership = membership & (not jnp.iscomplexobj(first))
            relative = jnp.swapaxes(jnp.conj(first), -1, -2) @ second
            angles = jnp.angle(jnp.linalg.eigvals(relative))
            margin = jnp.min(math.pi - jnp.abs(angles))
            distance = jnp.sqrt(0.5 * jnp.sum(angles * angles))
            branch = jnp.isfinite(margin) & (margin > self.branch_tolerance)
        elif self.space == "grassmann":
            singular = jnp.linalg.svd(
                jnp.swapaxes(jnp.conj(first), -1, -2) @ second, compute_uv=False
            )
            singular_valid = jnp.all(
                (singular >= -self.membership_tolerance)
                & (singular <= 1.0 + self.membership_tolerance)
            )
            safe = eqx.error_if(
                singular,
                ~singular_valid,
                "Grassmann principal-angle singular values left [0, 1].",
            )
            angles = jnp.arccos(jnp.minimum(jnp.maximum(safe, 0.0), 1.0))
            margin = jnp.min(0.5 * jnp.pi - angles)
            distance = jnp.linalg.norm(angles)
            branch = jnp.isfinite(margin) & (margin > self.branch_tolerance)
        else:
            tangent, residual = self.stiefel_log(first, second)
            tangent = jnp.asarray(tangent)
            residual = jnp.asarray(residual)
            margin = self.branch_tolerance - residual
            distance = jnp.linalg.norm(tangent)
            branch = (
                jnp.isfinite(residual)
                & (residual >= 0.0)
                & (residual <= self.branch_tolerance)
            )
        valid = (
            membership
            & branch
            & (residual <= self.branch_tolerance)
            & jnp.isfinite(distance)
        )
        return GeodesicDistanceEvidence(
            distance=distance,
            branch_margin=margin,
            log_residual=residual,
            membership_valid=membership,
            branch_valid=branch,
            valid=valid,
        )

    def pairwise(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        evidence = self.distance_evidence(left, right)
        value = jnp.asarray(self.radial_function(evidence.distance))
        return eqx.error_if(
            value,
            ~evidence.valid | ~jnp.isfinite(value),
            "Geodesic radial kernel left its membership/log branch epoch.",
        )

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        return jax.vmap(
            lambda first: jax.vmap(lambda second: self.pairwise(first, second))(
                jnp.asarray(right)
            )
        )(jnp.asarray(left))

    def require_positive_definite(self, /) -> None:
        if not self.positive_definite_capability:
            raise ValueError(
                "This geodesic radial family has no declared global positive-definiteness theorem."
            )


class GeodesicExponentialKernel(GeodesicRadialKernel):
    length_scale: float = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        space: CompactSpace,
        length_scale: float,
        membership_tolerance: float = 1e-6,
        branch_tolerance: float = 1e-6,
        stiefel_log: Callable[[Array, Array], tuple[Array, Array]] | None = None,
        positive_definite_theorem: bool = False,
    ):
        if float(length_scale) <= 0.0:
            raise ValueError("Geodesic exponential length_scale must be positive.")
        scale = float(length_scale)
        super().__init__(
            lambda distance: jnp.exp(-(distance * distance) / (2.0 * scale * scale)),
            space=space,
            membership_tolerance=membership_tolerance,
            branch_tolerance=branch_tolerance,
            stiefel_log=stiefel_log,
            positive_definite_theorem=positive_definite_theorem,
            kernel_id=f"geodesic-exponential:{space}",
        )
        self.length_scale = scale


__all__ = [
    "CompactHomogeneousHeatKernel",
    "CompactHomogeneousMaternKernel",
    "GeodesicDistanceEvidence",
    "GeodesicExponentialKernel",
    "GeodesicRadialKernel",
    "KernelEvaluationEvidence",
    "PreparedCompactHomogeneousSpectrum",
]
