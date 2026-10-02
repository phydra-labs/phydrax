#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from ..._differentiation import (
    DerivativeContract,
    DerivativeRegularity,
    DerivativeRoute,
    DerivativeSurface,
    GradientLevel,
    SurfaceDerivative,
)
from ..._model import ModelBinding
from ..._strict import StrictModule
from ..._trainable import fixed_field
from ..._validation import positive_integer
from ...linalg._policies import FailurePolicy
from ...linalg._svd_contracts import (
    DenseSVD,
    RandomizedSVD,
    SVDLeadingEvidence,
    SVDRangeEvidence,
    SVDRankEvidence,
    SVDResourcePolicy,
    SVDSolvePolicy,
    SVDTolerancePolicy,
)
from ...typing import parse, PRNGKey
from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    AbstractRecipe,
    FitResult,
    ML_INFEASIBLE,
    OperationDerivativeContract,
    prediction_fit_contract,
)
from .._numerics import (
    effective_sample_size,
    fit_weighted_subspace,
    parse_subspace_differentiation,
)
from .._numerics._spectral import SubspaceGradientTarget
from .._schema import AbstractFittedModel


# Encoding is an affine metric-scaled projection of the input.
_PREDICTION_CONTRACT = DerivativeContract(
    (
        SurfaceDerivative(DerivativeSurface.INPUT, GradientLevel.SMOOTH),
        SurfaceDerivative(DerivativeSurface.MODEL_PARAMETER, GradientLevel.SMOOTH),
    ),
    route=DerivativeRoute.DIRECT,
    regularity=DerivativeRegularity.smooth(degree_bound=1),
)


@final
class SubspaceDiagnostics(StrictModule):
    """Auditable spectral, weighting, and canonicalization diagnostics."""

    singular_values: Array
    explained_energy: Array
    retained_energy: Array
    residual_energy: Array
    rank_evidence: SVDRankEvidence
    range_evidence: SVDRangeEvidence
    leading_evidence: SVDLeadingEvidence
    cutoff_gap: Array
    internal_gap: Array
    pivot_magnitude: Array
    pivot_margin: Array
    native_status: Array
    derivative_status: Array
    admission_status: Array
    leading_certified: Array
    weighted_orthogonality_error: Array
    minimum_eigengap: Array
    projector_gradient_supported: Array
    basis_gradient_supported: Array
    repeated_spectrum: Array
    canonicalization_valid: Array
    valid: Array
    status: Array
    effective_samples: Array
    sign_phase_convention: str = eqx.field(static=True)
    centering_provenance: str = eqx.field(static=True)
    weighting_provenance: str = eqx.field(static=True)
    mask_provenance: str = eqx.field(static=True)
    query_layout_provenance: tuple[str, ...] = eqx.field(static=True)
    method: str = eqx.field(static=True)
    spectral_method: str = eqx.field(static=True)


@final
class SubspaceModel(AbstractFittedModel):
    """Fixed affine subspace with metric-correct encoding and decoding."""

    offset: Array = fixed_field()
    weighted_components: Array
    feature_metric: Array = fixed_field()
    feature_support: Array
    singular_values: Array = fixed_field()
    frame_correction: Array = fixed_field()
    covariance_correction: Array = fixed_field()
    fit_response_valid: Array = fixed_field()
    fit_response_status: Array = fixed_field()
    fit_response_provenance: bool = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    case_shape: tuple[int, ...] = eqx.field(static=True)
    centered: bool = eqx.field(static=True)
    weighting_provenance: str = eqx.field(static=True)
    centering_provenance: str = eqx.field(static=True)
    mask_provenance: str = eqx.field(static=True)
    query_layout_provenance: tuple[str, ...] = eqx.field(static=True)
    sign_phase_convention: str = eqx.field(static=True)

    _input_binding = ModelBinding.blockwise("flat", pass_key=False)

    def __init__(
        self,
        offset: ArrayLike,
        weighted_components: ArrayLike,
        feature_metric: ArrayLike,
        feature_support: ArrayLike,
        singular_values: ArrayLike,
        /,
        *,
        centered: bool,
        weighting_provenance: str,
        centering_provenance: str,
        mask_provenance: str,
        query_layout_provenance: tuple[str, ...] = (),
        frame_correction: ArrayLike | None = None,
        covariance_correction: ArrayLike | None = None,
        fit_response_valid: ArrayLike | None = None,
        fit_response_status: ArrayLike | None = None,
    ) -> None:
        offset_ = jnp.asarray(offset)
        weighted_ = jnp.asarray(weighted_components)
        metric = jnp.asarray(feature_metric, dtype=offset_.real.dtype)
        support = jnp.asarray(feature_support, dtype=jnp.bool_)
        if offset_.ndim < 1 or weighted_.shape[:-2] != offset_.shape[:-1]:
            raise ValueError("Subspace offset and components must share case axes.")
        if weighted_.shape[-1] != offset_.shape[-1]:
            raise ValueError("Subspace component width must match the feature count.")
        if metric.shape != offset_.shape or support.shape != offset_.shape:
            raise ValueError("Feature metric and support must match the offset shape.")
        self.offset = offset_
        self.weighted_components = weighted_
        self.feature_metric = metric
        self.feature_support = support
        self.singular_values = jnp.asarray(singular_values)
        self.frame_correction = (
            jnp.zeros_like(weighted_)
            if frame_correction is None
            else jnp.asarray(frame_correction)
        )
        core_shape = weighted_.shape[:-1] + (weighted_.shape[-2],)
        self.covariance_correction = (
            jnp.zeros(core_shape, dtype=weighted_.dtype)
            if covariance_correction is None
            else jnp.asarray(covariance_correction)
        )
        self.fit_response_valid = (
            jnp.zeros(offset_.shape[:-1], dtype=jnp.bool_)
            if fit_response_valid is None
            else jnp.asarray(fit_response_valid, dtype=jnp.bool_)
        )
        self.fit_response_status = (
            jnp.full(offset_.shape[:-1], ML_INFEASIBLE, dtype=jnp.int32)
            if fit_response_status is None
            else jnp.asarray(fit_response_status, dtype=jnp.int32)
        )
        self.fit_response_provenance = frame_correction is not None
        if (
            self.frame_correction.shape != weighted_.shape
            or self.covariance_correction.shape != core_shape
        ):
            raise ValueError(
                "Subspace response correction shapes must match the current basis and core."
            )
        self.frame_correction = eqx.error_if(
            self.frame_correction,
            jnp.any(self.frame_correction != 0),
            "Subspace frame correction must be identically zero in the primal.",
        )
        self.covariance_correction = eqx.error_if(
            self.covariance_correction,
            jnp.any(self.covariance_correction != 0),
            "Subspace covariance correction must be identically zero in the primal.",
        )
        self.in_size = offset_.shape[-1]
        self.out_size = weighted_.shape[-2]
        self.case_shape = tuple(offset_.shape[:-1])
        self.centered = bool(centered)
        self.weighting_provenance = str(weighting_provenance)
        self.centering_provenance = str(centering_provenance)
        self.mask_provenance = str(mask_provenance)
        self.query_layout_provenance = tuple(
            str(item) for item in query_layout_provenance
        )
        self.sign_phase_convention = "largest-magnitude-entry-positive-real"

    @property
    def components(self) -> Array:
        """Physical components derived from the single current weighted basis."""
        root = jnp.sqrt(
            jnp.where(
                self.feature_support & (self.feature_metric > 0.0),
                self.feature_metric,
                1.0,
            )
        )
        return jnp.where(
            self.feature_support[..., None, :],
            self.weighted_components / root[..., None, :],
            0,
        )

    def prediction_only(
        self, /, *, weighted_components: ArrayLike | None = None
    ) -> SubspaceModel:
        """Reconstruct current prediction state, explicitly clearing fit-response provenance."""
        return SubspaceModel(
            self.offset,
            self.weighted_components
            if weighted_components is None
            else weighted_components,
            self.feature_metric,
            self.feature_support,
            self.singular_values,
            centered=self.centered,
            weighting_provenance=self.weighting_provenance,
            centering_provenance=self.centering_provenance,
            mask_provenance=self.mask_provenance,
            query_layout_provenance=self.query_layout_provenance,
        )

    def _prediction_contract(self) -> DerivativeContract:
        return _PREDICTION_CONTRACT

    def _flatten_input(
        self, value: Array, width: int, /
    ) -> tuple[Array, tuple[int, ...]]:
        if value.shape[-1:] != (width,):
            raise ValueError(f"Expected a final axis of size {width}; got {value.shape}.")
        leading = tuple(value.shape[:-1])
        if self.case_shape:
            if leading[: len(self.case_shape)] != self.case_shape:
                raise ValueError(
                    f"Input must begin with fitted case shape {self.case_shape}; got {leading}."
                )
            sample_shape = leading[len(self.case_shape) :]
        else:
            sample_shape = leading
        return value.reshape(
            (max(1, _shape_product(self.case_shape)), -1, width)
        ), sample_shape

    def transform(self, x: ArrayLike, /) -> Array:
        value = jnp.asarray(x)
        flat, sample_shape = self._flatten_input(value, self.in_size)
        cases = max(1, _shape_product(self.case_shape))
        offset = self.offset.reshape((cases, 1, self.in_size))
        metric = self.feature_metric.reshape((cases, 1, self.in_size))
        support = self.feature_support.reshape((cases, 1, self.in_size))
        basis = self.weighted_components.reshape((cases, self.out_size, self.in_size))
        root = jnp.sqrt(jnp.where(support & (metric > 0.0), metric, 1.0))
        centered = jnp.where(support, flat - offset, 0)
        scores = ein.contract("cnf,crf->cnr", centered * root, jnp.conj(basis))
        return scores.reshape(self.case_shape + sample_shape + (self.out_size,))

    def inverse_transform(
        self, scores: ArrayLike, /, *, key: PRNGKey | None = None
    ) -> Array:
        del key
        value = jnp.asarray(scores)
        flat, sample_shape = self._flatten_input(value, self.out_size)
        cases = max(1, _shape_product(self.case_shape))
        basis = self.weighted_components.reshape((cases, self.out_size, self.in_size))
        metric = self.feature_metric.reshape((cases, 1, self.in_size))
        support = self.feature_support.reshape((cases, 1, self.in_size))
        root = jnp.sqrt(jnp.where(support & (metric > 0.0), metric, 1.0))
        reconstructed = ein.contract("cnr,crf->cnf", flat, basis) / root
        offset = self.offset.reshape((cases, 1, self.in_size))
        reconstructed = jnp.where(support, reconstructed + offset, offset)
        return reconstructed.reshape(self.case_shape + sample_shape + (self.in_size,))

    def project(self, x: ArrayLike, /) -> Array:
        value = jnp.asarray(x)
        flat, sample_shape = self._flatten_input(value, self.in_size)
        cases = max(1, _shape_product(self.case_shape))
        support = self.feature_support.reshape((cases, 1, self.in_size))
        metric = self.feature_metric.reshape((cases, 1, self.in_size))
        offset = self.offset.reshape((cases, 1, self.in_size))
        root = jnp.sqrt(jnp.where(support & (metric > 0.0), metric, 1.0))
        basis = (self.weighted_components + self.frame_correction).reshape(
            (cases, self.out_size, self.in_size)
        )
        weighted = jnp.where(support, flat - offset, 0) * root
        scores = ein.contract("cnf,crf->cnr", weighted, jnp.conj(basis))
        projected = ein.contract("cnr,crf->cnf", scores, basis) / root
        return jnp.where(support, projected + offset, offset).reshape(
            self.case_shape + sample_shape + (self.in_size,)
        )

    def projector(self, /) -> Array:
        """Return the column-vector projector in the original feature coordinates."""
        root = jnp.sqrt(
            jnp.where(
                self.feature_support & (self.feature_metric > 0.0),
                self.feature_metric,
                1.0,
            )
        )
        basis = self.weighted_components + self.frame_correction
        spectral = jnp.swapaxes(jnp.conj(basis), -1, -2) @ basis
        physical = spectral * root[..., None, :] / root[..., :, None]
        return jnp.where(
            self.feature_support[..., :, None] & self.feature_support[..., None, :],
            physical,
            0,
        )

    def __call__(self, x: ArrayLike, /, *, key: PRNGKey | None = None) -> Array:
        del key
        return self.transform(x)


def _shape_product(shape: tuple[int, ...], /) -> int:
    result = 1
    for size in shape:
        result *= int(size)
    return result


def _derivative_contract(
    target: SubspaceGradientTarget,
    /,
    *,
    projection: bool = False,
    route: DerivativeRoute = DerivativeRoute.SPECTRAL,
) -> DerivativeContract:
    if target == "none" or (target == "projector" and not projection):
        return prediction_fit_contract(_PREDICTION_CONTRACT, route=DerivativeRoute.DIRECT)
    conditions = (
        ("projector response requires retained/discarded separation on fixed support",)
        if target == "projector"
        else (
            "retained values must be individually isolated and canonical pivots locally unique",
        )
    )
    if route is DerivativeRoute.UNROLLED:
        conditions += (
            "fixed-key QR range steps must remain full-column-rank and conditioned",
        )
    return prediction_fit_contract(
        _PREDICTION_CONTRACT,
        (
            SurfaceDerivative(DerivativeSurface.FIT_FEATURES, GradientLevel.CONDITIONAL),
            SurfaceDerivative(DerivativeSurface.FIT_WEIGHTS, GradientLevel.CONDITIONAL),
        ),
        route=route,
        conditions=conditions,
    )


def _operation_derivatives(
    target: SubspaceGradientTarget,
    valid: Array,
    status: Array,
    /,
    *,
    route: DerivativeRoute = DerivativeRoute.SPECTRAL,
) -> tuple[OperationDerivativeContract, ...]:
    surfaces = (DerivativeSurface.FIT_FEATURES, DerivativeSurface.FIT_WEIGHTS)
    return tuple(
        OperationDerivativeContract(
            name,
            _derivative_contract(
                target, projection=name in ("project", "projector"), route=route
            ),
            runtime_surfaces=surfaces,
            runtime_valid=jnp.stack((valid, valid), axis=-1),
            runtime_status=jnp.stack((status, status), axis=-1),
        )
        for name in ("transform", "inverse_transform", "project", "projector")
    )


def _physical_feature_metric(
    offset: Array,
    feature_mass: Array,
    physical_weights: ArrayLike | None,
    weight_policy: WeightPolicy,
    /,
) -> tuple[Array, Array, Array, str]:
    if physical_weights is None:
        metric = jnp.ones_like(offset.real)
        weighting = f"sample:{weight_policy};feature:euclidean"
    else:
        metric = jnp.broadcast_to(
            jnp.asarray(physical_weights, dtype=offset.real.dtype), offset.shape
        )
        weighting = f"sample:{weight_policy};feature:physical"
    valid = jnp.all(jnp.isfinite(metric) & (metric >= 0.0), axis=-1) & jnp.any(
        (metric > 0.0) & (feature_mass > 0.0), axis=-1
    )
    support = (feature_mass > 0.0) & jnp.isfinite(metric) & (metric > 0.0)
    return metric, support, valid, weighting


def _fit_subspace(
    batch: MLBatch,
    /,
    *,
    rank: int,
    centered: bool,
    weight_policy: WeightPolicy,
    physical_weights: ArrayLike | None,
    differentiate: SubspaceGradientTarget,
    method: str,
    query_layout_provenance: tuple[str, ...],
    native_method: DenseSVD | RandomizedSVD | None = None,
    tolerance: SVDTolerancePolicy | None = None,
    resources: SVDResourcePolicy | None = None,
    failure: FailurePolicy | None = None,
    key: PRNGKey | None = None,
) -> FitResult:
    values = batch.dense_features()
    sample_weights = batch.effective_weight(weight_policy)
    observed = (
        batch.feature_mask
        & batch.sample_mask[..., :, None]
        & (jnp.isfinite(sample_weights) & (sample_weights > 0.0))[..., :, None]
    )
    safe_values = jnp.where(observed, values, 0)
    observed_weight = sample_weights[..., :, None] * observed.astype(sample_weights.dtype)
    feature_mass = jnp.sum(observed_weight, axis=-2)
    if centered:
        offset = jnp.where(
            feature_mass > 0.0,
            jnp.sum(observed_weight * safe_values, axis=-2)
            / jnp.maximum(feature_mass, jnp.finfo(sample_weights.dtype).tiny),
            0,
        )
    else:
        offset = jnp.zeros(values.shape[:-2] + (values.shape[-1],), dtype=values.dtype)
    if differentiate == "none":
        offset = jax.lax.stop_gradient(offset)
    centered_values = jnp.where(observed, safe_values - offset[..., None, :], 0)

    metric, support, metric_valid, weighting = _physical_feature_metric(
        offset, feature_mass, physical_weights, weight_policy
    )
    safe_metric = jnp.where(support, metric, 1.0)
    weighted_values = jnp.where(
        observed & support[..., None, :],
        centered_values * jnp.sqrt(safe_metric)[..., None, :],
        0,
    )
    spectral = fit_weighted_subspace(
        weighted_values,
        sample_weights,
        rank=int(rank),
        centered=False,
        differentiate=differentiate,
        method=native_method,
        tolerance=tolerance,
        resources=resources,
        failure=failure,
        key=key,
        input_valid=metric_valid & batch.weights_valid(weight_policy),
    )
    components = spectral.components
    repeated = spectral.internal_gap <= 64.0 * jnp.finfo(
        spectral.singular_values.dtype
    ).eps * jnp.max(spectral.singular_values**2, axis=-1, initial=0.0)
    weights_valid = batch.weights_valid(weight_policy)
    valid = spectral.valid & weights_valid & metric_valid
    canonical = valid & spectral.basis_gradient_supported
    status = jnp.where(
        weights_valid & metric_valid, spectral.status, ML_INFEASIBLE
    ).astype(jnp.int32)
    if differentiate == "none":
        metric = jax.lax.stop_gradient(metric)
    response_status = jnp.where(valid, spectral.derivative_status, status)
    model = SubspaceModel(
        offset,
        components,
        metric,
        support,
        spectral.singular_values,
        centered=centered,
        weighting_provenance=weighting,
        centering_provenance="masked-weighted-feature-mean" if centered else "origin",
        mask_provenance="zero-extension-with-observed-feature-means",
        query_layout_provenance=query_layout_provenance,
        frame_correction=spectral.frame_correction,
        covariance_correction=spectral.covariance_correction,
        fit_response_valid=valid
        & (
            spectral.projector_gradient_supported
            if differentiate == "projector"
            else spectral.basis_gradient_supported
        ),
        fit_response_status=response_status,
    )
    diagnostics = SubspaceDiagnostics(
        singular_values=spectral.singular_values,
        explained_energy=spectral.explained_energy,
        retained_energy=spectral.retained_energy,
        residual_energy=spectral.residual_energy,
        rank_evidence=spectral.rank_evidence,
        range_evidence=spectral.range_evidence,
        leading_evidence=spectral.leading_evidence,
        cutoff_gap=spectral.cutoff_gap,
        internal_gap=spectral.internal_gap,
        pivot_magnitude=spectral.pivot_magnitude,
        pivot_margin=spectral.pivot_margin,
        native_status=spectral.native_status,
        derivative_status=spectral.derivative_status,
        admission_status=spectral.admission_status,
        leading_certified=spectral.leading_certified,
        weighted_orthogonality_error=spectral.orthogonality_error,
        minimum_eigengap=spectral.minimum_retained_gap,
        repeated_spectrum=repeated,
        canonicalization_valid=canonical,
        valid=valid,
        status=status,
        projector_gradient_supported=valid & spectral.projector_gradient_supported,
        basis_gradient_supported=valid & spectral.basis_gradient_supported,
        effective_samples=effective_sample_size(sample_weights),
        sign_phase_convention="largest-magnitude-entry-positive-real",
        centering_provenance="masked-weighted-feature-mean" if centered else "origin",
        weighting_provenance=weighting,
        mask_provenance="zero-extension-with-observed-feature-means",
        query_layout_provenance=query_layout_provenance,
        method=method,
        spectral_method=spectral.method,
    )
    route = (
        DerivativeRoute.UNROLLED
        if isinstance(native_method, RandomizedSVD)
        else DerivativeRoute.SPECTRAL
    )
    return FitResult(
        model,
        diagnostics,
        valid=valid,
        status=status,
        method=method,
        derivative_contract=_derivative_contract(differentiate, route=route),
        operation_derivatives=_operation_derivatives(
            differentiate, model.fit_response_valid, response_status, route=route
        ),
        default_operation="transform",
    )


@final
class PCA(AbstractRecipe):
    """Weighted and masked principal component analysis recipe."""

    n_components: int = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)
    differentiate: SubspaceGradientTarget = eqx.field(static=True)
    method: DenseSVD | RandomizedSVD
    tolerance: SVDTolerancePolicy
    resources: SVDResourcePolicy
    failure: FailurePolicy

    def __init__(
        self,
        n_components: int,
        /,
        *,
        weight_policy: WeightPolicy = "statistical",
        differentiate: SubspaceGradientTarget = "projector",
        method: DenseSVD | RandomizedSVD | None = None,
        tolerance: SVDTolerancePolicy | None = None,
        resources: SVDResourcePolicy | None = None,
        failure: FailurePolicy | None = None,
    ) -> None:
        native, target = _recipe_policy(
            n_components, differentiate, method, tolerance, resources, failure
        )
        weights = parse(weight_policy, WeightPolicy, "weight_policy")
        self.n_components = native.count
        self.weight_policy = weights
        self.differentiate = target
        self.method, self.tolerance = native.method, native.tolerance
        self.resources, self.failure = native.resources, native.failure

    @property
    def accepts_fit_key(self) -> bool:
        return isinstance(self.method, RandomizedSVD)

    def fit_batch(self, batch: MLBatch, /, *, key: PRNGKey | None = None) -> FitResult:
        return _fit_subspace(
            batch,
            rank=self.n_components,
            centered=True,
            weight_policy=self.weight_policy,
            physical_weights=None,
            differentiate=self.differentiate,
            method="pca-weighted-svd",
            query_layout_provenance=(),
            native_method=self.method,
            tolerance=self.tolerance,
            resources=self.resources,
            failure=self.failure,
            key=key,
        )


@final
class TruncatedSVD(AbstractRecipe):
    """Origin-anchored weighted truncated SVD recipe."""

    n_components: int = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)
    differentiate: SubspaceGradientTarget = eqx.field(static=True)
    method: DenseSVD | RandomizedSVD
    tolerance: SVDTolerancePolicy
    resources: SVDResourcePolicy
    failure: FailurePolicy

    def __init__(
        self,
        n_components: int,
        /,
        *,
        weight_policy: WeightPolicy = "statistical",
        differentiate: SubspaceGradientTarget = "projector",
        method: DenseSVD | RandomizedSVD | None = None,
        tolerance: SVDTolerancePolicy | None = None,
        resources: SVDResourcePolicy | None = None,
        failure: FailurePolicy | None = None,
    ) -> None:
        native, target = _recipe_policy(
            n_components, differentiate, method, tolerance, resources, failure
        )
        weights = parse(weight_policy, WeightPolicy, "weight_policy")
        self.n_components = native.count
        self.weight_policy = weights
        self.differentiate = target
        self.method, self.tolerance = native.method, native.tolerance
        self.resources, self.failure = native.resources, native.failure

    @property
    def accepts_fit_key(self) -> bool:
        return isinstance(self.method, RandomizedSVD)

    def fit_batch(self, batch: MLBatch, /, *, key: PRNGKey | None = None) -> FitResult:
        return _fit_subspace(
            batch,
            rank=self.n_components,
            centered=False,
            weight_policy=self.weight_policy,
            physical_weights=None,
            differentiate=self.differentiate,
            method="truncated-weighted-svd",
            query_layout_provenance=(),
            native_method=self.method,
            tolerance=self.tolerance,
            resources=self.resources,
            failure=self.failure,
            key=key,
        )


@final
class POD(AbstractRecipe):
    """Weighted, masked proper orthogonal decomposition in a physical metric."""

    n_components: int = eqx.field(static=True)
    physical_weights: Array | None
    centered: bool = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)
    differentiate: SubspaceGradientTarget = eqx.field(static=True)
    query_layout_provenance: tuple[str, ...] = eqx.field(static=True)
    method: DenseSVD | RandomizedSVD
    tolerance: SVDTolerancePolicy
    resources: SVDResourcePolicy
    failure: FailurePolicy

    def __init__(
        self,
        n_components: int,
        /,
        *,
        physical_weights: ArrayLike | None = None,
        centered: bool = False,
        weight_policy: WeightPolicy = "product",
        differentiate: SubspaceGradientTarget = "projector",
        query_layout_provenance: tuple[str, ...] = (),
        method: DenseSVD | RandomizedSVD | None = None,
        tolerance: SVDTolerancePolicy | None = None,
        resources: SVDResourcePolicy | None = None,
        failure: FailurePolicy | None = None,
    ) -> None:
        native, target = _recipe_policy(
            n_components, differentiate, method, tolerance, resources, failure
        )
        weights = parse(weight_policy, WeightPolicy, "weight_policy")
        if not isinstance(centered, bool):
            raise TypeError("centered must be a boolean.")
        if not isinstance(query_layout_provenance, tuple) or any(
            not isinstance(item, str) for item in query_layout_provenance
        ):
            raise TypeError("query_layout_provenance must be a tuple of strings.")
        metric = (
            None
            if physical_weights is None
            else jnp.asarray(physical_weights, dtype=jnp.float64)
        )
        self.n_components = native.count
        self.physical_weights = metric
        self.centered = centered
        self.weight_policy = weights
        self.query_layout_provenance = query_layout_provenance
        self.differentiate = target
        self.method, self.tolerance = native.method, native.tolerance
        self.resources, self.failure = native.resources, native.failure

    @property
    def accepts_fit_key(self) -> bool:
        return isinstance(self.method, RandomizedSVD)

    def fit_batch(self, batch: MLBatch, /, *, key: PRNGKey | None = None) -> FitResult:
        return _fit_subspace(
            batch,
            rank=self.n_components,
            centered=self.centered,
            weight_policy=self.weight_policy,
            physical_weights=self.physical_weights,
            differentiate=self.differentiate,
            method="proper-orthogonal-decomposition",
            query_layout_provenance=self.query_layout_provenance,
            native_method=self.method,
            tolerance=self.tolerance,
            resources=self.resources,
            failure=self.failure,
            key=key,
        )


def _recipe_policy(
    count: int,
    differentiate: SubspaceGradientTarget,
    method: DenseSVD | RandomizedSVD | None,
    tolerance: SVDTolerancePolicy | None,
    resources: SVDResourcePolicy | None,
    failure: FailurePolicy | None,
    /,
) -> tuple[SVDSolvePolicy, SubspaceGradientTarget]:
    target = parse_subspace_differentiation(differentiate)
    native = SVDSolvePolicy(
        method,
        count=positive_integer(count, "n_components"),
        differentiation=target,
        tolerance=tolerance,
        resources=resources,
        failure=FailurePolicy("status") if failure is None else failure,
    )
    return native, target


__all__ = [
    "PCA",
    "POD",
    "SubspaceDiagnostics",
    "SubspaceGradientTarget",
    "SubspaceModel",
    "TruncatedSVD",
]
