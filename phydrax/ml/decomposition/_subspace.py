#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._model import AbstractArrayModel, ModelBinding
from ..._strict import StrictModule
from .._batch import MLBatch, WeightPolicy
from .._contracts import (
    AbstractRecipe,
    FitResult,
    GradientContract,
    ML_INFEASIBLE,
)
from .._numerics import effective_sample_size, fit_weighted_subspace


SubspaceGradientTarget = Literal["projector", "basis", "none"]


class SubspaceDiagnostics(StrictModule):
    """Auditable spectral, weighting, and canonicalization diagnostics."""

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
    valid: Array
    status: Array
    effective_samples: Array
    sign_phase_convention: str = eqx.field(static=True)
    centering_provenance: str = eqx.field(static=True)
    weighting_provenance: str = eqx.field(static=True)
    mask_provenance: str = eqx.field(static=True)
    query_layout_provenance: tuple[str, ...] = eqx.field(static=True)
    method: str = eqx.field(static=True)


class SubspaceModel(AbstractArrayModel):
    """Fixed affine subspace with metric-correct encoding and decoding."""

    offset: Array
    components: Array
    weighted_components: Array
    feature_metric: Array
    feature_support: Array
    singular_values: Array
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
    ):
        offset_ = jnp.asarray(offset)
        weighted_ = jnp.asarray(weighted_components)
        metric = jnp.asarray(feature_metric, dtype=offset_.real.dtype)
        support = jnp.asarray(feature_support, dtype=bool)
        if offset_.ndim < 1 or weighted_.shape[:-2] != offset_.shape[:-1]:
            raise ValueError("Subspace offset and components must share case axes.")
        if weighted_.shape[-1] != offset_.shape[-1]:
            raise ValueError("Subspace component width must match the feature count.")
        if metric.shape != offset_.shape or support.shape != offset_.shape:
            raise ValueError("Feature metric and support must match the offset shape.")
        safe_root = jnp.sqrt(jnp.where(support & (metric > 0.0), metric, 1.0))
        components = jnp.where(
            support[..., None, :], weighted_ / safe_root[..., None, :], 0
        )
        self.offset = offset_
        self.components = components
        self.weighted_components = weighted_
        self.feature_metric = metric
        self.feature_support = support
        self.singular_values = jnp.asarray(singular_values)
        self.in_size = int(offset_.shape[-1])
        self.out_size = int(weighted_.shape[-2])
        self.case_shape = tuple(int(size) for size in offset_.shape[:-1])
        self.centered = bool(centered)
        self.weighting_provenance = str(weighting_provenance)
        self.centering_provenance = str(centering_provenance)
        self.mask_provenance = str(mask_provenance)
        self.query_layout_provenance = tuple(
            str(item) for item in query_layout_provenance
        )
        self.sign_phase_convention = "largest-magnitude-entry-positive-real"

    def _flatten_input(
        self, value: Array, width: int, /
    ) -> tuple[Array, tuple[int, ...]]:
        if value.shape[-1:] != (width,):
            raise ValueError(f"Expected a final axis of size {width}; got {value.shape}.")
        leading = tuple(int(size) for size in value.shape[:-1])
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

    def inverse_transform(self, scores: ArrayLike, /) -> Array:
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
        return self.inverse_transform(self.transform(x))

    def projector(self, /) -> Array:
        """Return the column-vector projector in the original feature coordinates."""
        root = jnp.sqrt(
            jnp.where(
                self.feature_support & (self.feature_metric > 0.0),
                self.feature_metric,
                1.0,
            )
        )
        spectral = (
            jnp.swapaxes(jnp.conj(self.weighted_components), -1, -2)
            @ self.weighted_components
        )
        return spectral * root[..., None, :] / root[..., :, None]

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        return self.transform(x)


def _shape_product(shape: tuple[int, ...], /) -> int:
    result = 1
    for size in shape:
        result *= int(size)
    return result


def _gradient_contract(target: SubspaceGradientTarget, /) -> GradientContract:
    if target == "projector":
        return GradientContract(
            fit_features="conditional",
            fit_weights="conditional",
            fit_mode="spectral",
            conditions=(
                "projector gradients require separation between retained and discarded spectra",
            ),
        )
    if target == "basis":
        return GradientContract(
            fit_features="conditional",
            fit_weights="conditional",
            fit_mode="spectral",
            conditions=(
                "basis gradients require a non-repeated retained spectrum",
                "basis representatives use the largest-magnitude-entry positive-real convention",
                "canonicalization pivots must remain unique and nonzero",
            ),
        )
    return GradientContract(fit_mode="stopped")


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
) -> FitResult:
    values = batch.dense_features()
    sample_weights = batch.effective_weight(weight_policy)
    observed = batch.feature_mask & batch.sample_mask[..., :, None]
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
    centered_values = jnp.where(observed, safe_values - offset[..., None, :], 0)

    if physical_weights is None:
        metric = jnp.ones_like(offset.real)
        weighting = f"sample:{weight_policy};feature:euclidean"
    else:
        metric = jnp.broadcast_to(
            jnp.asarray(physical_weights, dtype=offset.real.dtype), offset.shape
        )
        weighting = f"sample:{weight_policy};feature:physical"
    metric_valid = jnp.all(jnp.isfinite(metric) & (metric >= 0.0), axis=-1) & jnp.any(
        (metric > 0.0) & (feature_mass > 0.0), axis=-1
    )
    support = (feature_mass > 0.0) & jnp.isfinite(metric) & (metric > 0.0)
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
    )
    components = jnp.where(support[..., None, :], spectral.components, 0)
    largest = jnp.max(spectral.singular_values, axis=-1, initial=0.0)
    gap_scale = jnp.maximum(largest, jnp.finfo(spectral.singular_values.dtype).tiny)
    repeated = spectral.minimum_retained_gap <= (
        64.0 * jnp.finfo(spectral.singular_values.dtype).eps * gap_scale
    )
    weights_valid = batch.weights_valid(weight_policy)
    valid = spectral.valid & weights_valid & metric_valid
    canonical = valid & ~repeated
    status = jnp.where(
        weights_valid & metric_valid, spectral.status, ML_INFEASIBLE
    ).astype(jnp.int32)
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
    )
    diagnostics = SubspaceDiagnostics(
        singular_values=spectral.singular_values,
        explained_energy=spectral.explained_energy,
        retained_energy=spectral.retained_energy,
        residual_energy=spectral.residual_energy,
        numerical_rank=spectral.numerical_rank,
        weighted_orthogonality_error=spectral.orthogonality_error,
        minimum_eigengap=spectral.minimum_retained_gap,
        repeated_spectrum=repeated,
        canonicalization_valid=canonical,
        valid=valid,
        status=status,
        projector_gradient_supported=valid & ~repeated,
        basis_gradient_supported=canonical,
        effective_samples=effective_sample_size(sample_weights),
        sign_phase_convention="largest-magnitude-entry-positive-real",
        centering_provenance="masked-weighted-feature-mean" if centered else "origin",
        weighting_provenance=weighting,
        mask_provenance="zero-extension-with-observed-feature-means",
        query_layout_provenance=query_layout_provenance,
        method=method,
    )
    return FitResult(
        model,
        diagnostics,
        valid=valid,
        status=status,
        method=method,
        gradient_contract=_gradient_contract(differentiate),
    )


class PCA(AbstractRecipe):
    """Weighted and masked principal component analysis recipe."""

    n_components: int = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)
    differentiate: SubspaceGradientTarget = eqx.field(static=True)

    def __init__(
        self,
        n_components: int,
        /,
        *,
        weight_policy: WeightPolicy = "statistical",
        differentiate: SubspaceGradientTarget = "projector",
    ):
        self.n_components = int(n_components)
        self.weight_policy = weight_policy
        self.differentiate = differentiate
        if self.n_components <= 0:
            raise ValueError("n_components must be positive.")
        if differentiate not in ("projector", "basis", "none"):
            raise ValueError("differentiate must be 'projector', 'basis', or 'none'.")

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_subspace(
            batch,
            rank=self.n_components,
            centered=True,
            weight_policy=self.weight_policy,
            physical_weights=None,
            differentiate=self.differentiate,
            method="pca-weighted-svd",
            query_layout_provenance=(),
        )


class TruncatedSVD(AbstractRecipe):
    """Origin-anchored weighted truncated SVD recipe."""

    n_components: int = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)
    differentiate: SubspaceGradientTarget = eqx.field(static=True)

    def __init__(
        self,
        n_components: int,
        /,
        *,
        weight_policy: WeightPolicy = "statistical",
        differentiate: SubspaceGradientTarget = "projector",
    ):
        self.n_components = int(n_components)
        self.weight_policy = weight_policy
        self.differentiate = differentiate
        if self.n_components <= 0:
            raise ValueError("n_components must be positive.")
        if differentiate not in ("projector", "basis", "none"):
            raise ValueError("differentiate must be 'projector', 'basis', or 'none'.")

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_subspace(
            batch,
            rank=self.n_components,
            centered=False,
            weight_policy=self.weight_policy,
            physical_weights=None,
            differentiate=self.differentiate,
            method="truncated-weighted-svd",
            query_layout_provenance=(),
        )


class POD(AbstractRecipe):
    """Weighted, masked proper orthogonal decomposition in a physical metric."""

    n_components: int = eqx.field(static=True)
    physical_weights: Array | None
    centered: bool = eqx.field(static=True)
    weight_policy: WeightPolicy = eqx.field(static=True)
    differentiate: SubspaceGradientTarget = eqx.field(static=True)
    query_layout_provenance: tuple[str, ...] = eqx.field(static=True)

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
    ):
        self.n_components = int(n_components)
        self.physical_weights = (
            None
            if physical_weights is None
            else jnp.asarray(physical_weights, dtype=float)
        )
        self.centered = bool(centered)
        self.weight_policy = weight_policy
        self.differentiate = differentiate
        self.query_layout_provenance = tuple(
            str(item) for item in query_layout_provenance
        )
        if self.n_components <= 0:
            raise ValueError("n_components must be positive.")
        if differentiate not in ("projector", "basis", "none"):
            raise ValueError("differentiate must be 'projector', 'basis', or 'none'.")

    def fit_batch(self, batch: MLBatch, /, *, key: Any = None) -> FitResult:
        del key
        return _fit_subspace(
            batch,
            rank=self.n_components,
            centered=self.centered,
            weight_policy=self.weight_policy,
            physical_weights=self.physical_weights,
            differentiate=self.differentiate,
            method="proper-orthogonal-decomposition",
            query_layout_provenance=self.query_layout_provenance,
        )


__all__ = [
    "PCA",
    "POD",
    "SubspaceDiagnostics",
    "SubspaceGradientTarget",
    "SubspaceModel",
    "TruncatedSVD",
]
