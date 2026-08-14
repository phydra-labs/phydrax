#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._strict import StrictModule
from ....ml._contracts import (
    GradientContract,
    ML_INFEASIBLE,
)
from ....ml._numerics import effective_sample_size, fit_weighted_subspace
from ..architectures.conditioning._deeponet import PODBasis
from ._dataset import OperatorDataset


class OperatorPODDiagnostics(StrictModule):
    """Physical POD diagnostics tied to one operator query layout."""

    singular_values: Array
    explained_energy: Array
    retained_energy: Array
    residual_energy: Array
    numerical_rank: Array
    weighted_orthogonality_error: Array
    minimum_eigengap: Array
    projector_gradient_supported: Array
    basis_gradient_supported: Array
    repeated_spectrum: Array
    canonicalization_valid: Array
    effective_samples: Array
    valid: Array
    status: Array
    sign_phase_convention: str = eqx.field(static=True)
    centering_provenance: str = eqx.field(static=True)
    weighting_provenance: str = eqx.field(static=True)
    query_layout_provenance: tuple[str, ...] = eqx.field(static=True)
    geometry_fingerprint: str = eqx.field(static=True)
    method: str = eqx.field(static=True)


class OperatorPODFit(StrictModule):
    """Immutable affine reduced basis and its physical projection contract."""

    basis: PODBasis
    diagnostics: OperatorPODDiagnostics
    components: Array
    spatial_mean: Array
    physical_weights: Array
    valid: Array
    status: Array
    gradient_contract: GradientContract
    field_name: str = eqx.field(static=True)
    query_name: str = eqx.field(static=True)
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    out_size: int | str = eqx.field(static=True)
    centered: bool = eqx.field(static=True)
    geometry_fingerprint: str = eqx.field(static=True)

    def transform(self, values: ArrayLike, /) -> Array:
        array = jnp.asarray(values)
        expected = self.sample_shape + (
            () if self.out_size == "scalar" else (int(self.out_size),)
        )
        if tuple(int(size) for size in array.shape[-len(expected) :]) != expected:
            raise ValueError(
                f"POD values must end in fitted output layout {expected}; got {array.shape}."
            )
        leading = tuple(int(size) for size in array.shape[: array.ndim - len(expected)])
        flat = array.reshape(leading + (-1,))
        mean = self.spatial_mean.reshape((-1,))
        weights = self.physical_weights.reshape((-1,))
        return jnp.einsum(
            "...f,rf,f->...r",
            flat - mean,
            jnp.conj(self.components),
            weights,
        )

    def inverse_transform(self, coefficients: ArrayLike, /) -> Array:
        values = jnp.asarray(coefficients) @ self.components + self.spatial_mean.reshape(
            (-1,)
        )
        trailing = self.sample_shape + (
            () if self.out_size == "scalar" else (int(self.out_size),)
        )
        return values.reshape(values.shape[:-1] + trailing)


def fit_operator_pod(
    dataset: OperatorDataset,
    field_name: str,
    n_components: int,
    /,
    *,
    centered: bool = False,
    sample_weight: ArrayLike | None = None,
    require_physical_quadrature: bool = False,
    differentiate: str = "projector",
) -> OperatorPODFit:
    """Fit one fixed-query POD decoder through the shared weighted-SVD kernel."""
    if not isinstance(dataset, OperatorDataset):
        raise TypeError("fit_operator_pod requires an OperatorDataset.")
    rank = int(n_components)
    if rank <= 0:
        raise ValueError("n_components must be positive.")
    if differentiate not in ("projector", "basis", "none"):
        raise ValueError("differentiate must be 'projector', 'basis', or 'none'.")
    field = dataset.targets.field(str(field_name))
    query = dataset.batch.query(field.query_name)
    if query.geometry_case_shape:
        raise ValueError(
            "POD fitting requires one shared fixed query layout; per-case query geometry "
            "must be registered to a common layout before fitting."
        )
    if require_physical_quadrature and not query.has_physical_quadrature:
        raise ValueError("Physical POD requires explicit quadrature on every query axis.")

    values = jnp.asarray(field.values)
    out_count = 1 if field.spec.channels == "scalar" else int(field.spec.channels)
    if field.spec.channels == "scalar":
        values = values[..., None]
    cases = int(dataset.size)
    feature_count = 1
    for size in query.sample_shape:
        feature_count *= int(size)
    feature_count *= out_count
    flat_values = values.reshape((cases, feature_count))

    query_weights = query.weights(case_shape=()).reshape((-1,))
    physical_weights = jnp.repeat(query_weights, out_count)
    metric_valid = jnp.all(
        jnp.isfinite(physical_weights) & (physical_weights >= 0.0)
    ) & jnp.any(physical_weights > 0.0)
    support = jnp.isfinite(physical_weights) & (physical_weights > 0.0)
    safe_weights = jnp.where(support, physical_weights, 1.0)
    weighted_values = jnp.where(
        support[None, :],
        flat_values * jnp.sqrt(safe_weights)[None, :],
        0,
    )
    if sample_weight is None:
        snapshot_weights = jnp.ones((cases,), dtype=flat_values.real.dtype)
    else:
        snapshot_weights = jnp.broadcast_to(
            jnp.asarray(sample_weight, dtype=flat_values.real.dtype), (cases,)
        )
    spectral = fit_weighted_subspace(
        weighted_values,
        snapshot_weights,
        rank=rank,
        centered=bool(centered),
    )
    physical_components = jnp.where(
        support[None, :],
        spectral.components / jnp.sqrt(safe_weights)[None, :],
        0,
    )
    spatial_mean = jnp.where(
        support,
        spectral.offset / jnp.sqrt(safe_weights),
        0,
    )
    basis_values = jnp.swapaxes(physical_components, -1, -2).reshape(
        query.sample_shape + (out_count, rank)
    )
    offset_values = spatial_mean.reshape(query.sample_shape + (out_count,))
    fingerprint = query.geometry_fingerprint()
    provenance = (
        f"query:{field.query_name}",
        f"sample_shape:{query.sample_shape}",
        f"axis_names:{query.axis_names}",
        f"geometry:{fingerprint}",
    )
    decoder = PODBasis(
        basis_values,
        latent_size=rank,
        out_size=field.spec.channels,
        offset=offset_values if centered else None,
        query_layout=query,
        geometry_fingerprint=fingerprint,
    )
    valid_weights = jnp.all(
        jnp.isfinite(snapshot_weights) & (snapshot_weights >= 0.0)
    ) & (jnp.sum(snapshot_weights) > 0.0)
    valid = spectral.valid & valid_weights & metric_valid
    status = jnp.where(
        valid_weights & metric_valid, spectral.status, ML_INFEASIBLE
    ).astype(jnp.int32)
    largest = jnp.max(spectral.singular_values, initial=0.0)
    repeated = spectral.minimum_retained_gap <= (
        64.0 * jnp.finfo(spectral.singular_values.dtype).eps * jnp.maximum(largest, 1.0)
    )
    diagnostics = OperatorPODDiagnostics(
        singular_values=spectral.singular_values,
        explained_energy=spectral.explained_energy,
        retained_energy=spectral.retained_energy,
        residual_energy=spectral.residual_energy,
        numerical_rank=spectral.numerical_rank,
        weighted_orthogonality_error=spectral.orthogonality_error,
        minimum_eigengap=spectral.minimum_retained_gap,
        repeated_spectrum=repeated,
        projector_gradient_supported=valid & ~repeated,
        basis_gradient_supported=valid & ~repeated,
        canonicalization_valid=valid & ~repeated,
        effective_samples=effective_sample_size(snapshot_weights),
        valid=valid,
        status=status,
        sign_phase_convention="largest-magnitude-entry-positive-real",
        centering_provenance="fixed-spatial-snapshot-mean" if centered else "origin",
        weighting_provenance=(
            "query-quadrature-times-mask; snapshot:explicit"
            if sample_weight is not None
            else "query-quadrature-times-mask; snapshot:uniform"
        ),
        query_layout_provenance=provenance,
        geometry_fingerprint=fingerprint,
        method="operator-physical-pod-weighted-svd",
    )
    if differentiate == "projector":
        gradient_contract = GradientContract(
            fit_features="conditional",
            fit_weights="conditional",
            fit_mode="spectral",
            conditions=(
                "projector gradients require retained/discarded spectral separation",
            ),
        )
    elif differentiate == "basis":
        gradient_contract = GradientContract(
            fit_features="conditional",
            fit_weights="conditional",
            fit_mode="spectral",
            conditions=(
                "basis gradients require a non-repeated retained spectrum",
                "canonicalization pivots must remain unique and nonzero",
            ),
        )
    else:
        gradient_contract = GradientContract(fit_mode="stopped")
    return OperatorPODFit(
        basis=decoder,
        diagnostics=diagnostics,
        components=physical_components,
        spatial_mean=spatial_mean,
        physical_weights=physical_weights,
        valid=valid,
        status=status,
        gradient_contract=gradient_contract,
        field_name=str(field_name),
        query_name=field.query_name,
        sample_shape=query.sample_shape,
        out_size=field.spec.channels,
        centered=bool(centered),
        geometry_fingerprint=fingerprint,
    )


def fit_pod_basis(
    dataset: OperatorDataset,
    field_name: str,
    n_components: int,
    /,
    **kwargs: Any,
) -> OperatorPODFit:
    """Alias emphasizing that the result is directly consumable by POD-DeepONet."""
    return fit_operator_pod(dataset, field_name, n_components, **kwargs)


__all__ = [
    "fit_operator_pod",
    "fit_pod_basis",
    "OperatorPODDiagnostics",
    "OperatorPODFit",
]
