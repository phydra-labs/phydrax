#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Correlated lattice continuum/finite-size studies with explicit abstention."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    DenseLinearOperator,
    eigen,
    FactorizationPolicy,
    factorize,
    FailurePolicy,
    MaterializationPolicy,
    OperatorProperties,
)


ContinuumStudyStatus: TypeAlias = Literal["complete", "abstained"]


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


def _finite_positive(value: float, name: str, /) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return resolved


class ScaleSettingCondition(StrictModule, NonTrainableState):
    """Physical reference divided by a measured dimensionless inverse scale."""

    physical_reference: Array
    physical_reference_uncertainty: Array
    minimum_signal_to_noise: float = eqx.field(static=True)
    scheme: str = eqx.field(static=True)
    condition_id: str = eqx.field(static=True)

    def __init__(
        self,
        physical_reference: float,
        /,
        *,
        scheme: str,
        physical_reference_uncertainty: float = 0.0,
        minimum_signal_to_noise: float = 3.0,
    ):
        reference = _finite_positive(physical_reference, "physical_reference")
        uncertainty = float(physical_reference_uncertainty)
        signal = _finite_positive(minimum_signal_to_noise, "minimum_signal_to_noise")
        scheme_id = _identifier(scheme, "scheme")
        if not isfinite(uncertainty) or uncertainty < 0.0:
            raise ValueError(
                "physical_reference_uncertainty must be finite and non-negative."
            )
        self.physical_reference = jnp.asarray(reference)
        self.physical_reference_uncertainty = jnp.asarray(uncertainty)
        self.minimum_signal_to_noise = signal
        self.scheme = scheme_id
        self.condition_id = canonical_fingerprint(
            {
                "kind": "lattice-scale-setting-condition",
                "scheme": scheme_id,
                "physical_reference": reference,
                "physical_reference_uncertainty": uncertainty,
                "relation": "a=physical-reference/measured-inverse-scale",
                "minimum_signal_to_noise": signal,
            }
        )


class RenormalizationCondition(StrictModule, NonTrainableState):
    """Multiplicative operator condition ``O_R = Z**power O_bare``."""

    power: float = eqx.field(static=True)
    scheme: str = eqx.field(static=True)
    scale: str = eqx.field(static=True)
    condition_id: str = eqx.field(static=True)

    def __init__(self, *, scheme: str, scale: str, power: float = 1.0):
        scheme_id = _identifier(scheme, "scheme")
        scale_id = _identifier(scale, "scale")
        power_ = float(power)
        if not isfinite(power_):
            raise ValueError("Renormalization power must be finite.")
        self.power = power_
        self.scheme = scheme_id
        self.scale = scale_id
        self.condition_id = canonical_fingerprint(
            {
                "kind": "multiplicative-renormalization-condition",
                "scheme": scheme_id,
                "scale": scale_id,
                "power": power_,
            }
        )


class CorrelatedContinuumData(StrictModule, NonTrainableState):
    """Joint bare-observable, inverse-scale, and Z-factor covariance."""

    bare_observable: Array
    inverse_scale: Array
    renormalization_factor: Array
    spatial_site_counts: Array
    joint_covariance: Array
    datum_ids: tuple[str, ...] = eqx.field(static=True)
    covariance_id: str = eqx.field(static=True)
    data_id: str = eqx.field(static=True)

    def __init__(
        self,
        bare_observable: ArrayLike,
        inverse_scale: ArrayLike,
        renormalization_factor: ArrayLike,
        spatial_site_counts: ArrayLike,
        joint_covariance: ArrayLike,
        /,
        *,
        datum_ids: Sequence[str],
        covariance_id: str,
        maximum_data_points: int = 4096,
        symmetry_tolerance: float = 1.0e-10,
    ):
        observable = np.asarray(bare_observable, dtype=np.float64).reshape((-1,))
        inverse = np.asarray(inverse_scale, dtype=np.float64).reshape((-1,))
        renormalization = np.asarray(renormalization_factor, dtype=np.float64).reshape(
            (-1,)
        )
        sites = np.asarray(spatial_site_counts, dtype=np.int64).reshape((-1,))
        count = observable.size
        covariance = np.asarray(joint_covariance, dtype=np.float64)
        identifiers = tuple(_identifier(value, "datum_id") for value in datum_ids)
        covariance_identity = _identifier(covariance_id, "covariance_id")
        maximum = int(maximum_data_points)
        tolerance = float(symmetry_tolerance)
        if count < 1 or maximum <= 0 or count > maximum:
            raise ValueError(
                "Continuum data count is empty or exceeds maximum_data_points."
            )
        if (
            inverse.shape != (count,)
            or renormalization.shape != (count,)
            or sites.shape != (count,)
            or covariance.shape != (3 * count, 3 * count)
            or len(identifiers) != count
        ):
            raise ValueError(
                "Continuum arrays must align; joint covariance order is [bare, scale, Z]."
            )
        if len(set(identifiers)) != count:
            raise ValueError("Continuum datum IDs must be unique.")
        if (
            np.any(~np.isfinite(observable))
            or np.any(~np.isfinite(inverse))
            or np.any(~np.isfinite(renormalization))
            or np.any(~np.isfinite(covariance))
            or np.any(inverse <= 0.0)
            or np.any(renormalization <= 0.0)
            or np.any(sites <= 0)
        ):
            raise ValueError(
                "Continuum observables/scales/Z factors must be finite with positive scales."
            )
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("symmetry_tolerance must be finite and non-negative.")
        asymmetry = np.max(np.abs(covariance - covariance.T), initial=0.0)
        covariance_scale = max(1.0, float(np.max(np.abs(covariance), initial=0.0)))
        if asymmetry > tolerance * covariance_scale or np.any(np.diag(covariance) < 0.0):
            raise ValueError(
                "joint_covariance must be symmetric with non-negative diagonal."
            )
        self.bare_observable = jnp.asarray(observable)
        self.inverse_scale = jnp.asarray(inverse)
        self.renormalization_factor = jnp.asarray(renormalization)
        self.spatial_site_counts = jnp.asarray(sites)
        self.joint_covariance = jnp.asarray(0.5 * (covariance + covariance.T))
        self.datum_ids = identifiers
        self.covariance_id = covariance_identity
        self.data_id = canonical_fingerprint(
            {
                "kind": "correlated-continuum-data",
                "datum_ids": identifiers,
                "covariance_id": covariance_identity,
                "arrays": array_tree_fingerprint(
                    (observable, inverse, renormalization, sites, covariance)
                ),
                "joint_order": ["bare-observable", "inverse-scale", "z-factor"],
            }
        )

    @property
    def count(self) -> int:
        return len(self.datum_ids)


class ContinuumSystematicVariation(StrictModule, NonTrainableState):
    """One fit window and cutoff/finite-volume ansatz."""

    maximum_lattice_spacing: float | None = eqx.field(static=True)
    minimum_physical_extent: float | None = eqx.field(static=True)
    cutoff_power: float = eqx.field(static=True)
    finite_volume_power: float | None = eqx.field(static=True)
    variation_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        cutoff_power: float,
        finite_volume_power: float | None,
        maximum_lattice_spacing: float | None = None,
        minimum_physical_extent: float | None = None,
        variation_id: str | None = None,
    ):
        cutoff = _finite_positive(cutoff_power, "cutoff_power")
        volume = (
            None
            if finite_volume_power is None
            else _finite_positive(finite_volume_power, "finite_volume_power")
        )
        maximum = (
            None
            if maximum_lattice_spacing is None
            else _finite_positive(maximum_lattice_spacing, "maximum_lattice_spacing")
        )
        minimum = (
            None
            if minimum_physical_extent is None
            else _finite_positive(minimum_physical_extent, "minimum_physical_extent")
        )
        resolved_id = (
            canonical_fingerprint(
                {
                    "kind": "continuum-systematic-variation",
                    "maximum_lattice_spacing": maximum,
                    "minimum_physical_extent": minimum,
                    "cutoff_power": cutoff,
                    "finite_volume_power": volume,
                }
            )
            if variation_id is None
            else _identifier(variation_id, "variation_id")
        )
        self.maximum_lattice_spacing = maximum
        self.minimum_physical_extent = minimum
        self.cutoff_power = cutoff
        self.finite_volume_power = volume
        self.variation_id = resolved_id


class ContinuumStudyPlan(StrictModule, NonTrainableState):
    """Prespecified correlated-fit variations and resolution/volume gates."""

    scale_setting: ScaleSettingCondition
    renormalization: RenormalizationCondition
    variations: tuple[ContinuumSystematicVariation, ...]
    minimum_distinct_spacings: int = eqx.field(static=True)
    minimum_distinct_volumes: int = eqx.field(static=True)
    minimum_degrees_of_freedom: int = eqx.field(static=True)
    maximum_matrix_entries: int = eqx.field(static=True)
    maximum_matrix_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale_setting: ScaleSettingCondition,
        renormalization: RenormalizationCondition,
        variations: Sequence[ContinuumSystematicVariation],
        /,
        *,
        minimum_distinct_spacings: int = 3,
        minimum_distinct_volumes: int = 2,
        minimum_degrees_of_freedom: int = 1,
        maximum_matrix_entries: int = 1_000_000,
        maximum_matrix_bytes: int = 64 * 1024 * 1024,
    ):
        if not isinstance(scale_setting, ScaleSettingCondition):
            raise TypeError("scale_setting must be ScaleSettingCondition.")
        if not isinstance(renormalization, RenormalizationCondition):
            raise TypeError("renormalization must be RenormalizationCondition.")
        resolved = tuple(variations)
        if not resolved or any(
            not isinstance(value, ContinuumSystematicVariation) for value in resolved
        ):
            raise TypeError(
                "variations must contain at least one ContinuumSystematicVariation."
            )
        if len({value.variation_id for value in resolved}) != len(resolved):
            raise ValueError("Continuum variation IDs must be unique.")
        spacings = int(minimum_distinct_spacings)
        volumes = int(minimum_distinct_volumes)
        degrees = int(minimum_degrees_of_freedom)
        entries = int(maximum_matrix_entries)
        byte_count = int(maximum_matrix_bytes)
        if spacings < 2 or volumes < 1 or degrees < 0:
            raise ValueError("Continuum evidence minima are invalid.")
        if entries <= 0 or byte_count <= 0:
            raise ValueError("Continuum matrix resource limits must be positive.")
        self.scale_setting = scale_setting
        self.renormalization = renormalization
        self.variations = resolved
        self.minimum_distinct_spacings = spacings
        self.minimum_distinct_volumes = volumes
        self.minimum_degrees_of_freedom = degrees
        self.maximum_matrix_entries = entries
        self.maximum_matrix_bytes = byte_count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "correlated-continuum-study-plan",
                "scale_setting": scale_setting.condition_id,
                "renormalization": renormalization.condition_id,
                "variations": tuple(value.variation_id for value in resolved),
                "minimum_distinct_spacings": spacings,
                "minimum_distinct_volumes": volumes,
                "minimum_degrees_of_freedom": degrees,
                "maximum_matrix_entries": entries,
                "maximum_matrix_bytes": byte_count,
            }
        )


class ContinuumFitResult(StrictModule, NonTrainableState):
    coefficients: Array
    coefficient_covariance: Array
    continuum_value: Array
    continuum_standard_error: Array
    chi_square: Array
    degrees_of_freedom: int = eqx.field(static=True)
    datum_indices: tuple[int, ...] = eqx.field(static=True)
    spacing_count: int = eqx.field(static=True)
    volume_count: int = eqx.field(static=True)
    variation_id: str = eqx.field(static=True)
    fit_id: str = eqx.field(static=True)


class ContinuumStudyEvidence(StrictModule, NonTrainableState):
    physical_lattice_spacings: Array
    physical_extents: Array
    renormalized_observables: Array
    propagated_covariance: Array
    scale_signal_to_noise: Array
    accepted_variation_ids: tuple[str, ...] = eqx.field(static=True)
    rejected_variations: tuple[tuple[str, str], ...] = eqx.field(static=True)
    scale_sufficient: bool = eqx.field(static=True)
    resolution_sufficient: bool = eqx.field(static=True)
    volume_sufficient: bool = eqx.field(static=True)
    covariance_usable: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class ContinuumStudyResult(StrictModule, NonTrainableState):
    fits: tuple[ContinuumFitResult, ...]
    continuum_value: Array | None
    statistical_standard_error: Array | None
    systematic_standard_error: Array | None
    total_standard_error: Array | None
    evidence: ContinuumStudyEvidence
    status: ContinuumStudyStatus = eqx.field(static=True)
    abstention_reasons: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    data_id: str = eqx.field(static=True)
    study_id: str = eqx.field(static=True)


def _factorization_policy(plan: ContinuumStudyPlan, /) -> FactorizationPolicy:
    return FactorizationPolicy(
        "cholesky",
        materialization=MaterializationPolicy(
            max_entries=plan.maximum_matrix_entries,
            max_bytes=plan.maximum_matrix_bytes,
        ),
        failure=FailurePolicy("status"),
    )


def _positive_factor(matrix: Array, plan: ContinuumStudyPlan, /):
    symmetric = 0.5 * (matrix + matrix.T)
    self_adjoint = OperatorProperties(
        self_adjoint=True,
        evidence={"self_adjoint": "construction"},
    )
    materialization = MaterializationPolicy(
        max_entries=plan.maximum_matrix_entries,
        max_bytes=plan.maximum_matrix_bytes,
    )
    spectrum = eigen.eigensolve(
        eigen.Eigenproblem(DenseLinearOperator(symmetric, properties=self_adjoint)),
        policy=eigen.EigenSolvePolicy(
            eigen.DenseEigh(),
            count=1,
            which="smallest-algebraic",
            materialization=materialization,
            failure=FailurePolicy("status"),
        ),
    )
    valid = bool(
        np.all(np.asarray(spectrum.successful))
        and np.asarray(spectrum.eigenvalues[0]) > 0.0
    )
    if not valid:
        return None, False
    positive_definite = OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "verified",
        },
    )
    factor = factorize(
        DenseLinearOperator(symmetric, properties=positive_definite),
        _factorization_policy(plan),
    )
    return factor, int(np.asarray(factor.rank())) == matrix.shape[0]


def _fit_once(
    design: Array,
    observable: Array,
    covariance: Array,
    plan: ContinuumStudyPlan,
    /,
) -> tuple[Array, Array, Array, bool]:
    covariance_factor, covariance_valid = _positive_factor(covariance, plan)
    if not covariance_valid:
        zeros = jnp.zeros((design.shape[1],), dtype=observable.dtype)
        square = jnp.zeros((design.shape[1], design.shape[1]), dtype=observable.dtype)
        return zeros, square, jnp.asarray(jnp.inf), False
    precision_observable = covariance_factor.solve(observable)
    precision_design = covariance_factor.solve(design)
    if not bool(
        np.all(np.asarray(precision_observable.successful))
        and np.all(np.asarray(precision_design.successful))
    ):
        zeros = jnp.zeros((design.shape[1],), dtype=observable.dtype)
        square = jnp.zeros((design.shape[1], design.shape[1]), dtype=observable.dtype)
        return zeros, square, jnp.asarray(jnp.inf), False
    normal = contract("ni,nj->ij", design, precision_design.value)
    normal = 0.5 * (normal + normal.T)
    right_hand_side = contract("ni,n->i", design, precision_observable.value)
    normal_factor, normal_valid = _positive_factor(normal, plan)
    if not normal_valid:
        zeros = jnp.zeros((design.shape[1],), dtype=observable.dtype)
        square = jnp.zeros((design.shape[1], design.shape[1]), dtype=observable.dtype)
        return zeros, square, jnp.asarray(jnp.inf), False
    coefficient_result = normal_factor.solve(right_hand_side)
    covariance_result = normal_factor.materialize_inverse()
    if not bool(
        np.all(np.asarray(coefficient_result.successful))
        and np.all(np.asarray(covariance_result.successful))
    ):
        zeros = jnp.zeros((design.shape[1],), dtype=observable.dtype)
        square = jnp.zeros((design.shape[1], design.shape[1]), dtype=observable.dtype)
        return zeros, square, jnp.asarray(jnp.inf), False
    coefficients = jnp.asarray(coefficient_result.value)
    residual = observable - design @ coefficients
    precision_residual = covariance_factor.solve(residual)
    if not bool(np.all(np.asarray(precision_residual.successful))):
        return coefficients, covariance_result.value, jnp.asarray(jnp.inf), False
    chi_square = contract("i,i->", residual, precision_residual.value)
    successful = bool(
        np.isfinite(np.asarray(chi_square))
        and np.all(np.isfinite(np.asarray(coefficients)))
        and np.all(np.isfinite(np.asarray(covariance_result.value)))
    )
    return coefficients, covariance_result.value, chi_square, successful


def _renormalized_jacobian(
    data: CorrelatedContinuumData,
    condition: RenormalizationCondition,
    /,
) -> tuple[Array, Array]:
    count = data.count
    power = condition.power
    factor_power = data.renormalization_factor**power
    observable = factor_power * data.bare_observable
    jacobian = jnp.zeros((count, 3 * count), dtype=observable.dtype)
    indices = jnp.arange(count)
    jacobian = jacobian.at[indices, indices].set(factor_power)
    z_derivative = (
        power * data.bare_observable * data.renormalization_factor ** (power - 1.0)
    )
    jacobian = jacobian.at[indices, 2 * count + indices].set(z_derivative)
    return observable, jacobian


def _effective_covariance(
    data: CorrelatedContinuumData,
    base_jacobian: Array,
    lattice_spacing: Array,
    physical_extent: Array,
    coefficients: Array,
    variation: ContinuumSystematicVariation,
    scale: ScaleSettingCondition,
    /,
) -> Array:
    count = data.count
    cutoff_term = coefficients[1] * lattice_spacing**variation.cutoff_power
    if variation.finite_volume_power is None:
        volume_term = jnp.zeros_like(cutoff_term)
    else:
        volume_term = coefficients[2] * physical_extent ** (
            -variation.finite_volume_power
        )
    model_derivative_inverse_scale = (
        -variation.cutoff_power * cutoff_term
        + (
            0.0
            if variation.finite_volume_power is None
            else variation.finite_volume_power * volume_term
        )
    ) / data.inverse_scale
    residual_jacobian = base_jacobian.at[
        jnp.arange(count), count + jnp.arange(count)
    ].set(-model_derivative_inverse_scale)
    covariance = residual_jacobian @ data.joint_covariance @ residual_jacobian.T
    reference_derivative = (
        variation.cutoff_power * cutoff_term
        - (
            0.0
            if variation.finite_volume_power is None
            else variation.finite_volume_power * volume_term
        )
    ) / scale.physical_reference
    covariance = covariance + (
        scale.physical_reference_uncertainty**2
        * reference_derivative[:, None]
        * reference_derivative[None, :]
    )
    return 0.5 * (covariance + covariance.T)


def _variation_indices(
    variation: ContinuumSystematicVariation,
    lattice_spacing: np.ndarray,
    physical_extent: np.ndarray,
    /,
) -> tuple[int, ...]:
    mask = np.ones(lattice_spacing.shape, dtype=np.bool_)
    if variation.maximum_lattice_spacing is not None:
        mask &= lattice_spacing <= variation.maximum_lattice_spacing
    if variation.minimum_physical_extent is not None:
        mask &= physical_extent >= variation.minimum_physical_extent
    return tuple(np.flatnonzero(mask))


def _design_matrix(
    variation: ContinuumSystematicVariation,
    lattice_spacing: Array,
    physical_extent: Array,
    /,
) -> Array:
    columns = [
        jnp.ones_like(lattice_spacing),
        lattice_spacing**variation.cutoff_power,
    ]
    if variation.finite_volume_power is not None:
        columns.append(physical_extent ** (-variation.finite_volume_power))
    return jnp.stack(tuple(columns), axis=1)


def run_continuum_study(
    plan: ContinuumStudyPlan,
    data: CorrelatedContinuumData,
    /,
) -> ContinuumStudyResult:
    """Run prespecified correlated GLS variations or return auditable abstention."""
    if not isinstance(plan, ContinuumStudyPlan):
        raise TypeError("plan must be ContinuumStudyPlan.")
    if not isinstance(data, CorrelatedContinuumData):
        raise TypeError("data must be CorrelatedContinuumData.")
    count = data.count
    lattice_spacing = plan.scale_setting.physical_reference / data.inverse_scale
    physical_extent = lattice_spacing * data.spatial_site_counts
    observable, observable_jacobian = _renormalized_jacobian(data, plan.renormalization)
    propagated_covariance = (
        observable_jacobian @ data.joint_covariance @ observable_jacobian.T
    )
    propagated_covariance = 0.5 * (propagated_covariance + propagated_covariance.T)
    scale_variance = jnp.diag(data.joint_covariance)[count : 2 * count]
    safe_scale_sigma = jnp.sqrt(jnp.maximum(scale_variance, 0.0))
    scale_signal = jnp.where(
        safe_scale_sigma > 0.0,
        data.inverse_scale / safe_scale_sigma,
        jnp.inf,
    )
    spacing_host = np.asarray(lattice_spacing)
    extent_host = np.asarray(physical_extent)
    resolution_sufficient = np.unique(spacing_host).size >= plan.minimum_distinct_spacings
    volume_sufficient = (
        np.unique(np.asarray(data.spatial_site_counts)).size
        >= plan.minimum_distinct_volumes
    )
    requires_volume = any(
        variation.finite_volume_power is not None for variation in plan.variations
    )
    scale_sufficient = bool(
        np.all(np.asarray(scale_signal) >= plan.scale_setting.minimum_signal_to_noise)
    )
    covariance_factor, covariance_usable = _positive_factor(propagated_covariance, plan)
    del covariance_factor
    accepted: list[ContinuumFitResult] = []
    rejected: list[tuple[str, str]] = []
    if not scale_sufficient:
        rejected.extend(
            (variation.variation_id, "scale-setting-signal-insufficient")
            for variation in plan.variations
        )
    elif not resolution_sufficient:
        rejected.extend(
            (variation.variation_id, "insufficient-distinct-lattice-spacings")
            for variation in plan.variations
        )
    elif not covariance_usable:
        rejected.extend(
            (variation.variation_id, "renormalized-covariance-not-positive-definite")
            for variation in plan.variations
        )
    else:
        for variation in plan.variations:
            indices = _variation_indices(variation, spacing_host, extent_host)
            parameter_count = 2 + int(variation.finite_volume_power is not None)
            degrees = len(indices) - parameter_count
            spacing_count = np.unique(
                spacing_host[np.asarray(indices, dtype=np.int64)]
            ).size
            volume_count = np.unique(
                np.asarray(data.spatial_site_counts)[np.asarray(indices, dtype=np.int64)]
            ).size
            if len(indices) < parameter_count + plan.minimum_degrees_of_freedom:
                rejected.append((variation.variation_id, "fit-window-has-too-few-data"))
                continue
            if spacing_count < plan.minimum_distinct_spacings:
                rejected.append(
                    (variation.variation_id, "fit-window-resolution-insufficient")
                )
                continue
            if (
                variation.finite_volume_power is not None
                and volume_count < plan.minimum_distinct_volumes
            ):
                rejected.append(
                    (variation.variation_id, "fit-window-volume-evidence-insufficient")
                )
                continue
            selection = jnp.asarray(indices, dtype=jnp.int32)
            selected_spacing = lattice_spacing[selection]
            selected_extent = physical_extent[selection]
            selected_observable = observable[selection]
            design = _design_matrix(variation, selected_spacing, selected_extent)
            selected_covariance = propagated_covariance[
                selection[:, None], selection[None, :]
            ]
            coefficients, coefficient_covariance, chi_square, successful = _fit_once(
                design, selected_observable, selected_covariance, plan
            )
            if not successful:
                rejected.append((variation.variation_id, "initial-correlated-fit-failed"))
                continue
            effective_full = _effective_covariance(
                data,
                observable_jacobian,
                lattice_spacing,
                physical_extent,
                coefficients,
                variation,
                plan.scale_setting,
            )
            effective_selected = effective_full[selection[:, None], selection[None, :]]
            coefficients, coefficient_covariance, chi_square, successful = _fit_once(
                design, selected_observable, effective_selected, plan
            )
            if not successful or float(np.asarray(coefficient_covariance[0, 0])) <= 0.0:
                rejected.append(
                    (variation.variation_id, "effective-correlated-fit-failed")
                )
                continue
            standard_error = jnp.sqrt(coefficient_covariance[0, 0])
            fit_id = canonical_fingerprint(
                {
                    "kind": "correlated-continuum-fit",
                    "plan": plan.plan_id,
                    "data": data.data_id,
                    "variation": variation.variation_id,
                    "datum_indices": indices,
                    "covariance": data.covariance_id,
                    "scale_setting": plan.scale_setting.condition_id,
                    "renormalization": plan.renormalization.condition_id,
                }
            )
            accepted.append(
                ContinuumFitResult(
                    coefficients=coefficients,
                    coefficient_covariance=coefficient_covariance,
                    continuum_value=coefficients[0],
                    continuum_standard_error=standard_error,
                    chi_square=chi_square,
                    degrees_of_freedom=degrees,
                    datum_indices=indices,
                    spacing_count=int(spacing_count),
                    volume_count=int(volume_count),
                    variation_id=variation.variation_id,
                    fit_id=fit_id,
                )
            )
    accepted_ids = tuple(value.variation_id for value in accepted)
    evidence_id = canonical_fingerprint(
        {
            "kind": "continuum-study-evidence",
            "plan": plan.plan_id,
            "data": data.data_id,
            "accepted_variations": accepted_ids,
            "rejected_variations": tuple(rejected),
            "resolution_sufficient": resolution_sufficient,
            "volume_sufficient": volume_sufficient,
            "scale_sufficient": scale_sufficient,
            "covariance_usable": covariance_usable,
        }
    )
    evidence = ContinuumStudyEvidence(
        physical_lattice_spacings=lattice_spacing,
        physical_extents=physical_extent,
        renormalized_observables=observable,
        propagated_covariance=propagated_covariance,
        scale_signal_to_noise=scale_signal,
        accepted_variation_ids=accepted_ids,
        rejected_variations=tuple(rejected),
        scale_sufficient=bool(scale_sufficient),
        resolution_sufficient=bool(resolution_sufficient),
        volume_sufficient=bool(volume_sufficient),
        covariance_usable=bool(covariance_usable),
        evidence_id=evidence_id,
    )
    reasons: list[str] = []
    if not scale_sufficient:
        reasons.append("scale-setting-signal-insufficient")
    if not resolution_sufficient:
        reasons.append("insufficient-distinct-lattice-spacings")
    if not accepted and requires_volume and not volume_sufficient:
        reasons.append("insufficient-distinct-spatial-volumes")
    if not covariance_usable:
        reasons.append("renormalized-covariance-not-positive-definite")
    if not accepted:
        reasons.append("no-prespecified-fit-variation-passed")
    if accepted:
        estimates = jnp.stack(tuple(value.continuum_value for value in accepted))
        variances = jnp.stack(
            tuple(value.continuum_standard_error**2 for value in accepted)
        )
        weights = 1.0 / variances
        normalized_weights = weights / jnp.sum(weights)
        continuum = contract("i,i->", normalized_weights, estimates)
        statistical = jnp.sqrt(1.0 / jnp.sum(weights))
        systematic = jnp.sqrt(
            contract("i,i->", normalized_weights, (estimates - continuum) ** 2)
        )
        total = jnp.sqrt(statistical**2 + systematic**2)
        status: ContinuumStudyStatus = "complete"
    else:
        continuum = None
        statistical = None
        systematic = None
        total = None
        status = "abstained"
    study_id = canonical_fingerprint(
        {
            "kind": "continuum-study-result",
            "plan": plan.plan_id,
            "data": data.data_id,
            "evidence": evidence_id,
            "fit_ids": tuple(value.fit_id for value in accepted),
            "status": status,
        }
    )
    return ContinuumStudyResult(
        fits=tuple(accepted),
        continuum_value=continuum,
        statistical_standard_error=statistical,
        systematic_standard_error=systematic,
        total_standard_error=total,
        evidence=evidence,
        status=status,
        abstention_reasons=tuple(dict.fromkeys(reasons)),
        plan_id=plan.plan_id,
        data_id=data.data_id,
        study_id=study_id,
    )


__all__ = [
    "ContinuumFitResult",
    "ContinuumStudyEvidence",
    "ContinuumStudyPlan",
    "ContinuumStudyResult",
    "ContinuumStudyStatus",
    "ContinuumSystematicVariation",
    "CorrelatedContinuumData",
    "RenormalizationCondition",
    "ScaleSettingCondition",
    "run_continuum_study",
]
