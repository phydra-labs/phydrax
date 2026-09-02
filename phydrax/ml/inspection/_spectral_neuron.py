#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._precision import PrecisionEvidenceEnvelope
from ..._strict import StrictModule
from ...linalg import HermitianPrecisionPolicy, HermitianSpectrum
from ...nn.layers._spectral_neuron import SpectralNeuron


def _adjoint(value: Array, /) -> Array:
    return jnp.swapaxes(jnp.conj(value), -1, -2)


class SpectralNeuronInspection(StrictModule):
    """Gauge-invariant global and local evidence for one spectral neuron."""

    matrix: Array
    eigenvalues: Array
    selected_eigenvalue: Array
    cluster_mask: Array
    cluster_size: Array
    cluster_lower_index: Array
    cluster_upper_index: Array
    lower_gap: Array
    upper_gap: Array
    minimum_external_gap: Array
    cluster_projector: Array
    global_feature_bounds: Array
    reference_eigenvalue: Array
    perturbation_bound: Array
    enclosure_lower: Array
    enclosure_upper: Array
    local_sensitivities: Array
    local_sensitivity_bounds: Array
    selected_is_numerically_simple: Array
    local_sensitivity_valid: Array
    guaranteed_nondecreasing: Array
    guaranteed_nonincreasing: Array
    valid: Array
    precision: HermitianPrecisionPolicy
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    eigen_index: int = eqx.field(static=True)
    input_count: int = eqx.field(static=True)
    matrix_size: int = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    convex: bool = eqx.field(static=True)
    concave: bool = eqx.field(static=True)
    monotonicity: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        matrix: ArrayLike,
        eigenvalues: ArrayLike,
        selected_eigenvalue: ArrayLike,
        cluster_mask: ArrayLike,
        cluster_size: ArrayLike,
        cluster_lower_index: ArrayLike,
        cluster_upper_index: ArrayLike,
        lower_gap: ArrayLike,
        upper_gap: ArrayLike,
        minimum_external_gap: ArrayLike,
        cluster_projector: ArrayLike,
        global_feature_bounds: ArrayLike,
        reference_eigenvalue: ArrayLike,
        perturbation_bound: ArrayLike,
        enclosure_lower: ArrayLike,
        enclosure_upper: ArrayLike,
        local_sensitivities: ArrayLike,
        local_sensitivity_bounds: ArrayLike,
        selected_is_numerically_simple: ArrayLike,
        local_sensitivity_valid: ArrayLike,
        guaranteed_nondecreasing: ArrayLike,
        guaranteed_nonincreasing: ArrayLike,
        valid: ArrayLike,
        precision: HermitianPrecisionPolicy,
        precision_evidence: PrecisionEvidenceEnvelope,
        eigen_index: int,
        input_count: int,
        matrix_size: int,
        relative_tolerance: float,
        absolute_tolerance: float,
        convex: bool,
        concave: bool,
        monotonicity: tuple[str, ...],
    ):
        self.matrix = jnp.asarray(matrix)
        self.eigenvalues = jnp.asarray(eigenvalues)
        self.selected_eigenvalue = jnp.asarray(selected_eigenvalue)
        self.cluster_mask = jnp.asarray(cluster_mask, dtype=bool)
        self.cluster_size = jnp.asarray(cluster_size, dtype=jnp.int32)
        self.cluster_lower_index = jnp.asarray(cluster_lower_index, dtype=jnp.int32)
        self.cluster_upper_index = jnp.asarray(cluster_upper_index, dtype=jnp.int32)
        self.lower_gap = jnp.asarray(lower_gap)
        self.upper_gap = jnp.asarray(upper_gap)
        self.minimum_external_gap = jnp.asarray(minimum_external_gap)
        self.cluster_projector = jnp.asarray(cluster_projector)
        self.global_feature_bounds = jnp.asarray(global_feature_bounds)
        self.reference_eigenvalue = jnp.asarray(reference_eigenvalue)
        self.perturbation_bound = jnp.asarray(perturbation_bound)
        self.enclosure_lower = jnp.asarray(enclosure_lower)
        self.enclosure_upper = jnp.asarray(enclosure_upper)
        self.local_sensitivities = jnp.asarray(local_sensitivities)
        self.local_sensitivity_bounds = jnp.asarray(local_sensitivity_bounds)
        self.selected_is_numerically_simple = jnp.asarray(
            selected_is_numerically_simple, dtype=bool
        )
        self.local_sensitivity_valid = jnp.asarray(local_sensitivity_valid, dtype=bool)
        self.guaranteed_nondecreasing = jnp.asarray(guaranteed_nondecreasing, dtype=bool)
        self.guaranteed_nonincreasing = jnp.asarray(guaranteed_nonincreasing, dtype=bool)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.precision = precision
        self.precision_evidence = precision_evidence
        self.eigen_index = int(eigen_index)
        self.input_count = int(input_count)
        self.matrix_size = int(matrix_size)
        self.relative_tolerance = float(relative_tolerance)
        self.absolute_tolerance = float(absolute_tolerance)
        self.convex = bool(convex)
        self.concave = bool(concave)
        self.monotonicity = tuple(monotonicity)


def inspect_spectral_neuron(
    model: SpectralNeuron,
    value: ArrayLike,
    /,
    *,
    relative_tolerance: float = 1e-8,
    absolute_tolerance: float = 1e-10,
    precision: HermitianPrecisionPolicy | None = None,
) -> SpectralNeuronInspection:
    """Inspect one selected eigenvalue without exposing an eigenvector gauge."""
    if not isinstance(model, SpectralNeuron):
        raise TypeError("model must be a SpectralNeuron.")
    relative = float(relative_tolerance)
    absolute = float(absolute_tolerance)
    if not math.isfinite(relative) or relative < 0.0:
        raise ValueError("relative_tolerance must be finite and nonnegative.")
    if not math.isfinite(absolute) or absolute < 0.0:
        raise ValueError("absolute_tolerance must be finite and nonnegative.")
    precision_ = model.precision if precision is None else precision
    if not isinstance(precision_, HermitianPrecisionPolicy):
        raise TypeError("precision must be a HermitianPrecisionPolicy or None.")

    matrix = model.matrix_pencil(value)
    spectrum = HermitianSpectrum(
        matrix,
        tolerance=absolute,
        precision=precision_,
    )
    decision_eigenvalues = precision_.decision(spectrum.eigenvalues)
    selected = decision_eigenvalues[..., model.eigen_index]
    scale = jnp.maximum(
        jnp.asarray(1.0, dtype=selected.dtype),
        jnp.max(jnp.abs(decision_eigenvalues), axis=-1),
    )
    tolerance = (
        jnp.asarray(absolute, dtype=selected.dtype)
        + jnp.asarray(relative, dtype=selected.dtype) * scale
    )
    cluster_mask = (
        jnp.abs(decision_eigenvalues - selected[..., None]) <= tolerance[..., None]
    )
    cluster_size = jnp.sum(cluster_mask, axis=-1, dtype=jnp.int32)
    indices = jnp.arange(model.matrix_size, dtype=jnp.int32)
    lower_index = jnp.min(jnp.where(cluster_mask, indices, model.matrix_size), axis=-1)
    upper_index = jnp.max(jnp.where(cluster_mask, indices, -1), axis=-1)

    if model.matrix_size == 1:
        lower_gap = jnp.full(selected.shape, jnp.inf, dtype=selected.dtype)
        upper_gap = jnp.full(selected.shape, jnp.inf, dtype=selected.dtype)
    else:
        differences = decision_eigenvalues[..., 1:] - decision_eigenvalues[..., :-1]
        infinity = jnp.asarray(jnp.inf, dtype=selected.dtype)
        lower_boundary = (~cluster_mask[..., :-1]) & cluster_mask[..., 1:]
        upper_boundary = cluster_mask[..., :-1] & (~cluster_mask[..., 1:])
        lower_gap = jnp.min(jnp.where(lower_boundary, differences, infinity), axis=-1)
        upper_gap = jnp.min(jnp.where(upper_boundary, differences, infinity), axis=-1)
    minimum_external_gap = jnp.minimum(lower_gap, upper_gap)
    selected_is_numerically_simple = (
        spectrum.valid
        & (cluster_size == 1)
        & (lower_gap > tolerance)
        & (upper_gap > tolerance)
    )

    weighted_vectors = spectrum.eigenvectors * cluster_mask[..., None, :]
    projector = contract(
        "...ai,...bi->...ab", weighted_vectors, jnp.conj(spectrum.eigenvectors)
    )
    projector = 0.5 * (projector + _adjoint(projector))

    base, feature_matrices = model.materialize_coefficients()
    coefficient_values = jnp.linalg.eigvalsh(
        precision_.factorization(precision_.compute(feature_matrices))
    )
    global_feature_bounds = precision_.decision(
        jnp.max(jnp.abs(coefficient_values), axis=-1)
    )
    reference_values = jnp.linalg.eigvalsh(
        precision_.factorization(precision_.compute(base))
    )
    reference_eigenvalue = precision_.output(reference_values[model.eigen_index])
    flattened, _ = model._flatten_input(value)
    perturbation_bound = contract(
        "...i,i->...", jnp.abs(flattened), global_feature_bounds
    )
    enclosure_lower = reference_eigenvalue - perturbation_bound
    enclosure_upper = reference_eigenvalue + perturbation_bound

    local_values = contract("...ab,iba->...i", projector, feature_matrices)
    nan = jnp.asarray(jnp.nan, dtype=local_values.real.dtype)
    local_sensitivities = jnp.where(
        selected_is_numerically_simple[..., None], local_values.real, nan
    )
    projected_left = contract("...ab,ibc->...iac", projector, feature_matrices)
    projected = contract("...iac,...cb->...iab", projected_left, projector)
    projected = 0.5 * (projected + _adjoint(projected))
    projected_values = jnp.linalg.eigvalsh(
        precision_.factorization(precision_.compute(projected))
    )
    local_sensitivity_bounds = precision_.decision(
        jnp.max(jnp.abs(projected_values), axis=-1)
    )

    guaranteed_nondecreasing = jnp.asarray(
        tuple(mode == "increasing" for mode in model.monotonicity), dtype=bool
    )
    guaranteed_nonincreasing = jnp.asarray(
        tuple(mode == "decreasing" for mode in model.monotonicity), dtype=bool
    )
    coefficient_valid = (
        jnp.all(jnp.isfinite(base))
        & jnp.all(jnp.isfinite(feature_matrices))
        & jnp.all(jnp.isfinite(global_feature_bounds))
        & jnp.isfinite(reference_eigenvalue)
    )
    case_valid = (
        spectrum.valid
        & jnp.all(jnp.isfinite(projector), axis=(-2, -1))
        & jnp.all(jnp.isfinite(local_sensitivity_bounds), axis=-1)
        & jnp.isfinite(perturbation_bound)
        & jnp.isfinite(enclosure_lower)
        & jnp.isfinite(enclosure_upper)
        & coefficient_valid
    )
    local_sensitivity_valid = case_valid & selected_is_numerically_simple

    return SpectralNeuronInspection(
        matrix=precision_.output(spectrum.matrix),
        eigenvalues=precision_.output(spectrum.eigenvalues),
        selected_eigenvalue=precision_.output(
            spectrum.eigenvalues[..., model.eigen_index]
        ),
        cluster_mask=cluster_mask,
        cluster_size=cluster_size,
        cluster_lower_index=lower_index,
        cluster_upper_index=upper_index,
        lower_gap=lower_gap,
        upper_gap=upper_gap,
        minimum_external_gap=minimum_external_gap,
        cluster_projector=precision_.output(projector),
        global_feature_bounds=global_feature_bounds,
        reference_eigenvalue=reference_eigenvalue,
        perturbation_bound=perturbation_bound,
        enclosure_lower=enclosure_lower,
        enclosure_upper=enclosure_upper,
        local_sensitivities=local_sensitivities,
        local_sensitivity_bounds=local_sensitivity_bounds,
        selected_is_numerically_simple=(case_valid & selected_is_numerically_simple),
        local_sensitivity_valid=local_sensitivity_valid,
        guaranteed_nondecreasing=guaranteed_nondecreasing,
        guaranteed_nonincreasing=guaranteed_nonincreasing,
        valid=case_valid,
        precision=precision_,
        precision_evidence=spectrum.precision_evidence,
        eigen_index=model.eigen_index,
        input_count=model._input_count,
        matrix_size=model.matrix_size,
        relative_tolerance=relative,
        absolute_tolerance=absolute,
        convex=model.is_convex,
        concave=model.is_concave,
        monotonicity=model.monotonicity,
    )


__all__ = ["SpectralNeuronInspection", "inspect_spectral_neuron"]
