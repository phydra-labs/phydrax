#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import final, Literal

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ...._differentiation import (
    DerivativeAdmission,
    DerivativeContract,
    DerivativeRoute,
    DerivativeSurface,
    DifferentiationRequest,
    GradientLevel,
    RegularityPolicy,
    SurfaceDerivative,
)
from ...._strict import StrictModule
from ...._trainable import fixed_field
from ....linalg._policies import FailurePolicy
from ....linalg.svd import (
    DenseSVD,
    RandomizedSVD,
    SVDLeadingEvidence,
    SVDRangeEvidence,
    SVDRankEvidence,
    SVDResourcePolicy,
    SVDTolerancePolicy,
)
from ....ml._contracts import ML_INFEASIBLE, OperationDerivativeContract
from ....ml._numerics import (
    effective_sample_size,
    fit_weighted_subspace,
    parse_subspace_differentiation,
)
from ....ml._numerics._spectral import SubspaceGradientTarget
from ....typing import PRNGKey
from ..architectures.conditioning._deeponet import PODBasis
from ._dataset import OperatorDataset


@final
class OperatorPODDiagnostics(StrictModule):
    """Physical POD evidence tied to one immutable query geometry."""

    singular_values: Array
    explained_energy: Array
    retained_energy: Array
    residual_energy: Array
    rank_evidence: SVDRankEvidence
    range_evidence: SVDRangeEvidence
    leading_evidence: SVDLeadingEvidence
    weighted_orthogonality_error: Array
    cutoff_gap: Array
    internal_gap: Array
    pivot_magnitude: Array
    pivot_margin: Array
    projector_gradient_supported: Array
    basis_gradient_supported: Array
    repeated_spectrum: Array
    canonicalization_valid: Array
    leading_certified: Array
    native_status: Array
    derivative_status: Array
    admission_status: Array
    effective_samples: Array
    valid: Array
    status: Array
    sign_phase_convention: str = eqx.field(static=True)
    centering_provenance: str = eqx.field(static=True)
    weighting_provenance: str = eqx.field(static=True)
    query_layout_provenance: tuple[str, ...] = eqx.field(static=True)
    geometry_fingerprint: str = eqx.field(static=True)
    method: str = eqx.field(static=True)


@final
class OperatorPODFit(StrictModule):
    """Affine decoder and invariant projection with separate fit contracts.

    The decoder owns the sole current weighted basis. Correction leaves are
    identically zero in the primal and fixed-role: they carry direct fit response,
    not a second independently mutable projector or trainable basis.
    """

    basis: PODBasis
    diagnostics: OperatorPODDiagnostics
    physical_weights: Array = fixed_field()
    frame_correction: Array = fixed_field()
    valid: Array
    status: Array
    derivative_contract: DerivativeContract
    operation_derivatives: tuple[OperationDerivativeContract, ...]
    field_name: str = eqx.field(static=True)
    query_name: str = eqx.field(static=True)
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    out_size: int | Literal["scalar"] = eqx.field(static=True)
    centered: bool = eqx.field(static=True)
    geometry_fingerprint: str = eqx.field(static=True)

    @property
    def components(self) -> Array:
        return self.basis.values.reshape((-1, self.basis.latent_size)).T

    @property
    def weighted_components(self) -> Array:
        return self.basis.weighted_values.reshape((-1, self.basis.latent_size)).T

    @property
    def spatial_mean(self) -> Array:
        trailing = self.sample_shape + (
            () if self.out_size == "scalar" else (self.out_size,)
        )
        return self.basis.offset.reshape(trailing)

    def _flatten(self, values: ArrayLike, /) -> tuple[Array, tuple[int, ...]]:
        array = jnp.asarray(values)
        trailing = self.sample_shape + (
            () if self.out_size == "scalar" else (self.out_size,)
        )
        if array.ndim < len(trailing) or tuple(array.shape[-len(trailing) :]) != trailing:
            raise ValueError(
                f"POD values must end in fitted output layout {trailing}; got {array.shape}."
            )
        return array.reshape(array.shape[: -len(trailing)] + (-1,)), trailing

    def transform(self, values: ArrayLike, /) -> Array:
        flat, _ = self._flatten(values)
        centered = jnp.where(
            self.basis.feature_support.reshape((-1,)),
            flat - self.spatial_mean.reshape((-1,)),
            0,
        )
        return (centered * self.basis.feature_scale.reshape((-1,))) @ jnp.conj(
            self.weighted_components
        ).T

    def inverse_transform(self, coefficients: ArrayLike, /) -> Array:
        coefficients_ = jnp.asarray(coefficients)
        if coefficients_.shape[-1:] != (self.basis.latent_size,):
            raise ValueError("POD coefficients must end in the fitted component count.")
        weighted = coefficients_ @ self.weighted_components
        values = jnp.where(
            self.basis.feature_support.reshape((-1,)),
            weighted / self.basis.feature_scale.reshape((-1,)),
            0,
        ) + self.spatial_mean.reshape((-1,))
        trailing = self.sample_shape + (
            () if self.out_size == "scalar" else (self.out_size,)
        )
        return values.reshape(values.shape[:-1] + trailing)

    def project(self, values: ArrayLike, /) -> Array:
        """Apply the metric-correct affine projector, including admitted fit response."""
        flat, trailing = self._flatten(values)
        root = self.basis.feature_scale.reshape((-1,))
        support = self.basis.feature_support.reshape((-1,))
        weighted = jnp.where(support, flat - self.spatial_mean.reshape((-1,)), 0) * root
        frame = self.weighted_components + self.frame_correction
        projected = (weighted @ jnp.conj(frame).T) @ frame
        physical = jnp.where(support, projected / root, 0)
        return (physical + self.spatial_mean.reshape((-1,))).reshape(
            flat.shape[:-1] + trailing
        )

    def projector(self) -> Array:
        """Return the flattened physical linear projector (without affine mean)."""
        frame = self.weighted_components + self.frame_correction
        root = self.basis.feature_scale.reshape((-1,))
        support = self.basis.feature_support.reshape((-1,))
        gram = jnp.conj(frame).T @ frame
        return jnp.where(
            support[:, None] & support[None, :],
            root[:, None] * gram / root[None, :],
            0,
        )

    def _operation(self, operation: str | None, /) -> OperationDerivativeContract:
        name = "transform" if operation is None else operation
        for entry in self.operation_derivatives:
            if entry.operation == name:
                return entry
        raise ValueError(f"Unknown operator POD derivative operation {name!r}.")

    def derivative_admission(
        self,
        request: DifferentiationRequest,
        /,
        *,
        operation: str | None = None,
        policy: RegularityPolicy | None = None,
    ) -> DerivativeAdmission:
        return self._operation(operation).admit(request, policy=policy)

    def require_derivative(
        self,
        request: DifferentiationRequest,
        /,
        *,
        operation: str | None = None,
        policy: RegularityPolicy | None = None,
    ) -> DerivativeAdmission:
        return self._operation(operation).require(request, policy=policy)


def _operation_contracts(
    differentiate: SubspaceGradientTarget,
    method: DenseSVD | RandomizedSVD | None,
    valid: Array,
    status: Array,
    projector_supported: Array,
    basis_supported: Array,
) -> tuple[OperationDerivativeContract, ...]:
    prediction = (
        SurfaceDerivative(DerivativeSurface.INPUT, GradientLevel.SMOOTH),
        SurfaceDerivative(DerivativeSurface.MODEL_PARAMETER, GradientLevel.SMOOTH),
    )
    fit_surfaces = (DerivativeSurface.FIT_FEATURES, DerivativeSurface.FIT_WEIGHTS)
    route = (
        DerivativeRoute.UNROLLED
        if isinstance(method, RandomizedSVD)
        else DerivativeRoute.SPECTRAL
    )
    result: list[OperationDerivativeContract] = []
    for name in ("transform", "inverse_transform", "decoder", "project", "projector"):
        projection = name in ("project", "projector")
        fit_supported = differentiate == "basis" or (
            differentiate == "projector" and projection
        )
        if fit_supported:
            contract = DerivativeContract(
                prediction
                + tuple(
                    SurfaceDerivative(surface, GradientLevel.CONDITIONAL)
                    for surface in fit_surfaces
                ),
                route=route,
                conditions=(
                    "fixed query geometry, physical support and direct first-order fit response",
                    "certified leading boundary and admitted QR margins"
                    if isinstance(method, RandomizedSVD)
                    else "retained/discarded boundary separation",
                )
                + (
                    ()
                    if projection
                    else ("isolated retained values and unique nonzero phase pivots",)
                ),
            )
            gate = valid & (
                basis_supported if differentiate == "basis" else projector_supported
            )
            result.append(
                OperationDerivativeContract(
                    name,
                    contract,
                    runtime_surfaces=fit_surfaces,
                    runtime_valid=jnp.stack((gate, gate), axis=-1),
                    runtime_status=jnp.stack((status, status), axis=-1),
                )
            )
        else:
            result.append(
                OperationDerivativeContract(
                    name, DerivativeContract(prediction, route=DerivativeRoute.DIRECT)
                )
            )
    return tuple(result)


def _component_capacity(value: int, /) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("n_components must be an integer capacity.")
    if value <= 0:
        raise ValueError("n_components must be positive.")
    return value


def fit_operator_pod(
    dataset: OperatorDataset,
    field_name: str,
    n_components: int,
    /,
    *,
    centered: bool = False,
    sample_weight: ArrayLike | None = None,
    require_physical_quadrature: bool = False,
    differentiate: SubspaceGradientTarget = "projector",
    method: DenseSVD | RandomizedSVD | None = None,
    tolerance: SVDTolerancePolicy | None = None,
    resources: SVDResourcePolicy | None = None,
    failure: FailurePolicy | None = None,
    key: PRNGKey | None = None,
) -> OperatorPODFit:
    """Fit fixed-query physical POD using the shared native weighted SVD adapter."""
    if not isinstance(dataset, OperatorDataset):
        raise TypeError("fit_operator_pod requires an OperatorDataset.")
    rank = _component_capacity(n_components)
    target = parse_subspace_differentiation(differentiate)
    field = dataset.targets.field(field_name)
    query = dataset.batch.query(field.query_name)
    if query.geometry_case_shape:
        raise ValueError(
            "POD fitting requires one shared fixed query layout; register per-case query geometry to a common layout before fitting."
        )
    if require_physical_quadrature and not query.has_physical_quadrature:
        raise ValueError("Physical POD requires explicit quadrature on every query axis.")
    values = jnp.asarray(field.values)
    out_count = 1 if field.spec.channels == "scalar" else field.spec.channels
    flat_values = values.reshape((dataset.size, -1))
    physical_weights = jnp.repeat(query.weights(case_shape=()).reshape((-1,)), out_count)
    metric_valid = jnp.all(
        jnp.isfinite(physical_weights) & (physical_weights >= 0.0)
    ) & jnp.any(physical_weights > 0.0)
    support = jnp.isfinite(physical_weights) & (physical_weights > 0.0)
    root = jnp.sqrt(jnp.where(support, physical_weights, 1.0))
    weighted_values = jnp.where(support[None, :], flat_values * root[None, :], 0)
    snapshot_weights = (
        jnp.ones((dataset.size,), dtype=flat_values.real.dtype)
        if sample_weight is None
        else jnp.broadcast_to(
            jnp.asarray(sample_weight, dtype=flat_values.real.dtype), (dataset.size,)
        )
    )
    valid_weights = jnp.all(
        jnp.isfinite(snapshot_weights) & (snapshot_weights >= 0.0)
    ) & (jnp.sum(snapshot_weights) > 0.0)
    spectral = fit_weighted_subspace(
        weighted_values,
        snapshot_weights,
        rank=rank,
        centered=centered,
        differentiate=target,
        method=method,
        tolerance=tolerance,
        resources=resources,
        failure=failure,
        key=key,
        input_valid=metric_valid,
    )
    spatial_mean = jnp.where(support, spectral.offset / root, 0)
    fingerprint = query.geometry_fingerprint()
    provenance = (
        f"query:{field.query_name}",
        f"sample_shape:{query.sample_shape}",
        f"axis_names:{query.axis_names}",
        f"geometry:{fingerprint}",
    )
    decoder = PODBasis(
        spectral.components.T.reshape(query.sample_shape + (out_count, rank)),
        latent_size=rank,
        out_size=field.spec.channels,
        offset=spatial_mean.reshape(query.sample_shape + (out_count,))
        if centered
        else None,
        feature_scale=root.reshape(query.sample_shape + (out_count,)),
        feature_support=support.reshape(query.sample_shape + (out_count,)),
        query_layout=query,
        geometry_fingerprint=fingerprint,
    )
    valid = spectral.valid & valid_weights & metric_valid
    status = jnp.where(
        valid_weights & metric_valid, spectral.status, ML_INFEASIBLE
    ).astype(jnp.int32)
    if failure is not None and failure.mode == "error":
        decoder = eqx.error_if(
            decoder, ~valid, "Operator POD fitting failed numerical admission."
        )
    largest = jnp.max(spectral.singular_values, initial=0.0)
    repeated = spectral.internal_gap <= 64.0 * jnp.finfo(
        spectral.singular_values.dtype
    ).eps * jnp.maximum(largest, 1.0)
    projection_supported = valid & (
        spectral.projector_gradient_supported | spectral.basis_gradient_supported
    )
    basis_supported = valid & spectral.basis_gradient_supported
    derivative_gate_status = jnp.where(valid, spectral.derivative_status, status)
    operations = _operation_contracts(
        target,
        method,
        valid,
        derivative_gate_status,
        projection_supported,
        basis_supported,
    )
    diagnostics = OperatorPODDiagnostics(
        singular_values=spectral.singular_values,
        explained_energy=spectral.explained_energy,
        retained_energy=spectral.retained_energy,
        residual_energy=spectral.residual_energy,
        rank_evidence=spectral.rank_evidence,
        weighted_orthogonality_error=spectral.orthogonality_error,
        range_evidence=spectral.range_evidence,
        leading_evidence=spectral.leading_evidence,
        cutoff_gap=spectral.cutoff_gap,
        internal_gap=spectral.internal_gap,
        pivot_magnitude=spectral.pivot_magnitude,
        pivot_margin=spectral.pivot_margin,
        projector_gradient_supported=projection_supported,
        basis_gradient_supported=basis_supported,
        repeated_spectrum=repeated,
        canonicalization_valid=basis_supported,
        leading_certified=spectral.leading_certified,
        native_status=spectral.native_status,
        derivative_status=spectral.derivative_status,
        admission_status=spectral.admission_status,
        effective_samples=effective_sample_size(snapshot_weights),
        valid=valid,
        status=status,
        sign_phase_convention="largest-magnitude-entry-positive-real",
        centering_provenance="fixed-spatial-snapshot-mean" if centered else "origin",
        weighting_provenance="query-quadrature-times-mask; snapshot:"
        + ("explicit" if sample_weight is not None else "uniform"),
        query_layout_provenance=provenance,
        geometry_fingerprint=fingerprint,
        method=spectral.method,
    )
    return OperatorPODFit(
        basis=decoder,
        diagnostics=diagnostics,
        physical_weights=physical_weights,
        frame_correction=spectral.frame_correction,
        valid=valid,
        status=status,
        derivative_contract=operations[0].contract,
        operation_derivatives=operations,
        field_name=field_name,
        query_name=field.query_name,
        sample_shape=query.sample_shape,
        out_size=field.spec.channels,
        centered=centered,
        geometry_fingerprint=fingerprint,
    )


def fit_pod_basis(
    dataset: OperatorDataset,
    field_name: str,
    n_components: int,
    /,
    *,
    centered: bool = False,
    sample_weight: ArrayLike | None = None,
    require_physical_quadrature: bool = False,
    differentiate: SubspaceGradientTarget = "projector",
    method: DenseSVD | RandomizedSVD | None = None,
    tolerance: SVDTolerancePolicy | None = None,
    resources: SVDResourcePolicy | None = None,
    failure: FailurePolicy | None = None,
    key: PRNGKey | None = None,
) -> OperatorPODFit:
    """Fit a POD basis directly consumable by POD-DeepONet."""
    return fit_operator_pod(
        dataset,
        field_name,
        n_components,
        centered=centered,
        sample_weight=sample_weight,
        require_physical_quadrature=require_physical_quadrature,
        differentiate=differentiate,
        method=method,
        tolerance=tolerance,
        resources=resources,
        failure=failure,
        key=key,
    )


__all__ = [
    "fit_operator_pod",
    "fit_pod_basis",
    "OperatorPODDiagnostics",
    "OperatorPODFit",
]
