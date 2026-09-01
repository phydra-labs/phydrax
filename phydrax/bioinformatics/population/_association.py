#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...linalg import (
    DenseCholesky,
    DenseLinearOperator,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    solve,
)
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._cohort import GenotypeCohort


UNSUPPORTED_ASSOCIATION_ANALYSES: tuple[str, ...] = (
    "fine-mapping",
    "polygenic-risk-score",
    "survival",
)


class AssociationStatus(IntEnum):
    SUCCESS = 0
    INSUFFICIENT_SAMPLES = 1
    MONOMORPHIC = 2
    NONFINITE = 3
    NO_CASES = 4
    NO_CONTROLS = 5
    SEPARATION = 6
    MAXIMUM_ITERATIONS = 7


class LOCOKinship(StrictModule):
    relationship_matrices: Array
    informative_variant_count: Array
    chromosome_labels: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        relationship_matrices: ArrayLike,
        informative_variant_count: ArrayLike,
        /,
        *,
        chromosome_labels: tuple[str, ...],
    ):
        matrices = jnp.asarray(relationship_matrices)
        counts = jnp.asarray(informative_variant_count)
        labels = tuple(str(value) for value in chromosome_labels)
        if matrices.ndim != 3 or matrices.shape[-1] != matrices.shape[-2]:
            raise ValueError(
                "relationship_matrices must have shape (chromosomes, samples, samples)."
            )
        if int(matrices.shape[0]) != len(labels) or counts.shape != (len(labels),):
            raise ValueError("LOCO matrices/counts must align with chromosome_labels.")
        if not jnp.issubdtype(counts.dtype, jnp.integer):
            raise TypeError("informative_variant_count must contain integers.")
        host = np.asarray(matrices)
        if not np.all(np.isfinite(host)) or not np.allclose(
            host, np.swapaxes(host, -1, -2), atol=1e-6, rtol=1e-6
        ):
            raise ValueError("LOCO relationship matrices must be finite and symmetric.")
        if any(not label for label in labels) or len(set(labels)) != len(labels):
            raise ValueError("chromosome_labels must be unique and non-empty.")
        self.relationship_matrices = matrices
        self.informative_variant_count = counts.astype(jnp.int32)
        self.chromosome_labels = labels


class QuantitativeAssociationResult(StrictModule):
    effect: Array
    standard_error: Array
    statistic: Array
    p_value: Array
    residual_variance: Array
    effective_samples: Array
    valid: Array
    status: Array
    evidence: Array
    used_loco: bool = eqx.field(static=True)
    contract: BioinformaticsMethodContract = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(AssociationStatus.SUCCESS))


class BinaryAssociationResult(StrictModule):
    log_odds_effect: Array
    standard_error: Array
    statistic: Array
    p_value: Array
    null_probability: Array
    null_iterations: Array
    case_count: Array
    control_count: Array
    imbalanced: Array
    separated: Array
    effective_samples: Array
    valid: Array
    status: Array
    evidence: Array
    used_loco: bool = eqx.field(static=True)
    contract: BioinformaticsMethodContract = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(AssociationStatus.SUCCESS))


def _association_contract(
    method_name: str, /, *, binary: bool
) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        method_name,
        MethodKind.APPROXIMATE_MODEL,
        ExecutionKind.ITERATIVE_TOLERANCE
        if binary
        else ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.IMPLICIT,
        OutputKind.STRUCTURED,
        conditioning_statement=(
            "Conditioned on posterior-mean biallelic dosage, fixed covariates, the "
            "declared phenotype family, and the supplied whole-genome or LOCO "
            "relationship covariance."
        ),
        truncation_statement="All variants and covariates are retained.",
        capacity_semantics="Dense native solves use the declared sample dimension.",
        assumptions=(
            "Additive allelic effect.",
            "Positive-semidefinite relationship covariance.",
            "Penalized quasi-likelihood working covariance."
            if binary
            else "Gaussian residual law.",
        ),
        nondifferentiable_outputs=(
            ("status", "valid", "imbalanced", "separated", "null_iterations")
            if binary
            else ("status", "valid")
        ),
    )


def _spd_solve(matrix: Array, right_hand_side: Array, /) -> Array:
    operator = DenseLinearOperator(
        matrix,
        properties=OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "construction",
                "positive_definite": "construction",
                "positive_semidefinite": "construction",
            },
        ),
    )
    return solve(
        LinearSystem(operator),
        right_hand_side,
        policy=LinearSolvePolicy(DenseCholesky(), failure=FailurePolicy("error")),
    ).value


def _design(
    covariates: ArrayLike | None, sample_count: int, dtype: jnp.dtype, /
) -> Array:
    if covariates is None:
        values = jnp.zeros((sample_count, 0), dtype=dtype)
    else:
        values = jnp.asarray(covariates, dtype=dtype)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != sample_count:
            raise ValueError("covariates must have shape (samples, covariates).")
    return jnp.concatenate((jnp.ones((sample_count, 1), dtype=dtype), values), axis=1)


def _validate_relationship(matrix: Array, sample_count: int, /) -> None:
    host = np.asarray(matrix)
    if host.shape != (sample_count, sample_count):
        raise ValueError("kinship/relationship must have shape (samples, samples).")
    if not np.all(np.isfinite(host)) or not np.allclose(
        host, host.T, atol=1e-6, rtol=1e-6
    ):
        raise ValueError("kinship/relationship must be finite and symmetric.")
    smallest = np.linalg.eigvalsh(0.5 * (host + host.T)).min(initial=0.0)
    if smallest < -1e-5:
        raise ValueError("kinship/relationship must be positive semidefinite.")


def leave_one_chromosome_out_kinship(cohort: GenotypeCohort, /) -> LOCOKinship:
    """Construct posterior-mean GRMs excluding each chromosome in turn."""
    if not isinstance(cohort, GenotypeCohort):
        raise TypeError("cohort must be a GenotypeCohort.")
    observed = cohort.observed
    allele_number = jnp.sum(jnp.where(observed, cohort.ploidy, 0), axis=1)
    alternate = jnp.sum(jnp.where(observed, cohort.dosage, 0.0), axis=1)
    frequency = alternate / jnp.maximum(allele_number, 1)
    mean_ploidy = jnp.sum(jnp.where(observed, cohort.ploidy, 0), axis=1) / jnp.maximum(
        jnp.sum(observed, axis=1), 1
    )
    variance = mean_ploidy * frequency * (1.0 - frequency)
    informative = (allele_number > 0) & (variance > 0.0)
    expected = mean_ploidy[:, None] * frequency[:, None]
    dosage = jnp.where(observed, cohort.dosage, expected)
    standardized = jnp.where(
        informative[:, None],
        (dosage - expected) / jnp.sqrt(jnp.maximum(variance[:, None], 1e-30)),
        0.0,
    )
    outputs: list[Array] = []
    counts: list[Array] = []
    for chromosome in range(len(cohort.chromosome_labels)):
        selected = informative & (cohort.chromosome_index != chromosome)
        count = jnp.sum(selected).astype(jnp.int32)
        matrix = (
            jnp.swapaxes(jnp.where(selected[:, None], standardized, 0.0), 0, 1)
            @ jnp.where(selected[:, None], standardized, 0.0)
        ) / jnp.maximum(count, 1)
        outputs.append(0.5 * (matrix + jnp.swapaxes(matrix, -1, -2)))
        counts.append(count)
    return LOCOKinship(
        jnp.stack(outputs),
        jnp.stack(counts),
        chromosome_labels=cohort.chromosome_labels,
    )


def _relationship_for_variant(
    cohort: GenotypeCohort,
    variant: int,
    kinship: Array | None,
    loco: LOCOKinship | None,
    /,
) -> Array:
    if loco is not None:
        chromosome = int(np.asarray(cohort.chromosome_index[variant]))
        return loco.relationship_matrices[chromosome]
    if kinship is not None:
        return kinship
    return jnp.zeros(
        (cohort.sample_count, cohort.sample_count), dtype=cohort.dosage.dtype
    )


def _gls_projection(
    covariance: Array,
    design: Array,
    response: Array,
    ridge: float,
    /,
) -> tuple[Array, Array, Array, Array]:
    precision_design = _spd_solve(covariance, design)
    precision_response = _spd_solve(covariance, response)
    normal = jnp.swapaxes(design, -1, -2) @ precision_design
    normal = normal + ridge * jnp.eye(normal.shape[0], dtype=normal.dtype)
    coefficients = _spd_solve(normal, jnp.swapaxes(design, -1, -2) @ precision_response)
    residual = response - design @ coefficients
    precision_residual = _spd_solve(covariance, residual)
    return residual, precision_residual, precision_design, normal


def quantitative_association(
    cohort: GenotypeCohort,
    phenotype: ArrayLike,
    /,
    *,
    covariates: ArrayLike | None = None,
    kinship: ArrayLike | None = None,
    loco: LOCOKinship | None = None,
    relatedness_scale: float = 1.0,
    ridge: float = 1e-8,
) -> QuantitativeAssociationResult:
    """Variant-wise Gaussian GLS association with kinship and optional LOCO GRMs."""
    if not isinstance(cohort, GenotypeCohort):
        raise TypeError("cohort must be a GenotypeCohort.")
    if loco is not None and not isinstance(loco, LOCOKinship):
        raise TypeError("loco must be a LOCOKinship or None.")
    if loco is not None and kinship is not None:
        raise ValueError("kinship and loco are mutually exclusive.")
    response = jnp.asarray(phenotype)
    if response.shape != (cohort.sample_count,):
        raise ValueError("phenotype must contain one value per sample.")
    if not jnp.issubdtype(response.dtype, jnp.inexact):
        response = response.astype(float)
    design = _design(covariates, cohort.sample_count, response.dtype)
    kinship_ = None if kinship is None else jnp.asarray(kinship, dtype=response.dtype)
    if kinship_ is not None:
        _validate_relationship(kinship_, cohort.sample_count)
    if loco is not None:
        if loco.chromosome_labels != cohort.chromosome_labels:
            raise ValueError("LOCO chromosome labels must match the cohort.")
        for matrix in np.asarray(loco.relationship_matrices):
            _validate_relationship(jnp.asarray(matrix), cohort.sample_count)
    scale = float(relatedness_scale)
    ridge_ = float(ridge)
    if not np.isfinite(scale) or scale < 0.0 or not np.isfinite(ridge_) or ridge_ <= 0.0:
        raise ValueError("relatedness_scale must be non-negative and ridge positive.")
    finite_input = jnp.all(jnp.isfinite(response)) & jnp.all(jnp.isfinite(design))
    dosage = cohort.dosage
    effects: list[Array] = []
    errors: list[Array] = []
    statistics: list[Array] = []
    p_values: list[Array] = []
    variances: list[Array] = []
    valid_values: list[Array] = []
    statuses: list[Array] = []
    effective: list[Array] = []
    degrees = cohort.sample_count - design.shape[1]
    eye = jnp.eye(cohort.sample_count, dtype=response.dtype)
    for variant in range(cohort.variant_count):
        relationship = _relationship_for_variant(cohort, variant, kinship_, loco)
        covariance = eye + scale * relationship + ridge_ * eye
        residual, precision_residual, precision_design, normal = _gls_projection(
            covariance, design, response, ridge_
        )
        observed_count = jnp.sum(cohort.observed[variant]).astype(jnp.int32)
        mean = jnp.sum(
            jnp.where(cohort.observed[variant], dosage[variant], 0.0)
        ) / jnp.maximum(observed_count, 1)
        genotype = jnp.where(cohort.observed[variant], dosage[variant], mean)
        precision_genotype = _spd_solve(covariance, genotype)
        nuisance = _spd_solve(normal, jnp.swapaxes(design, -1, -2) @ precision_genotype)
        residual_genotype = genotype - design @ nuisance
        precision_residual_genotype = _spd_solve(covariance, residual_genotype)
        information = residual_genotype @ precision_residual_genotype
        score = residual_genotype @ precision_residual
        residual_variance = (residual @ precision_residual) / max(degrees, 1)
        effect = score / jnp.maximum(information, 1e-30)
        standard_error = jnp.sqrt(
            jnp.maximum(residual_variance, 0.0) / jnp.maximum(information, 1e-30)
        )
        statistic = effect / jnp.maximum(standard_error, 1e-30)
        p_value = jsp.special.erfc(jnp.abs(statistic) / jnp.sqrt(2.0))
        finite = (
            finite_input
            & jnp.isfinite(effect)
            & jnp.isfinite(standard_error)
            & jnp.isfinite(statistic)
            & jnp.isfinite(p_value)
        )
        valid = (degrees > 0) & (observed_count >= 2) & (information > 1e-12) & finite
        status = jnp.where(
            ~finite,
            int(AssociationStatus.NONFINITE),
            jnp.where(
                (degrees <= 0) | (observed_count < 2),
                int(AssociationStatus.INSUFFICIENT_SAMPLES),
                jnp.where(
                    information <= 1e-12,
                    int(AssociationStatus.MONOMORPHIC),
                    int(AssociationStatus.SUCCESS),
                ),
            ),
        ).astype(jnp.int32)
        effects.append(jnp.where(valid, effect, jnp.nan))
        errors.append(jnp.where(valid, standard_error, jnp.nan))
        statistics.append(jnp.where(valid, statistic, jnp.nan))
        p_values.append(jnp.where(valid, p_value, jnp.nan))
        variances.append(jnp.where(valid, residual_variance, jnp.nan))
        valid_values.append(valid)
        statuses.append(status)
        effective.append(observed_count)
    effect = jnp.stack(effects)
    error = jnp.stack(errors)
    statistic = jnp.stack(statistics)
    p_value = jnp.stack(p_values)
    residual_variance = jnp.stack(variances)
    effective_samples = jnp.stack(effective)
    valid = jnp.stack(valid_values)
    status = jnp.stack(statuses)
    evidence = jnp.stack(
        (effective_samples, jnp.full_like(effective_samples, degrees)), axis=-1
    )
    return QuantitativeAssociationResult(
        effect,
        error,
        statistic,
        p_value,
        residual_variance,
        effective_samples,
        valid,
        status,
        evidence,
        loco is not None,
        _association_contract("quantitative-gls-association", binary=False),
    )


def _fit_binary_null(
    response: Array,
    design: Array,
    relationship: Array,
    /,
    *,
    relatedness_scale: float,
    ridge: float,
    maximum_iterations: int,
    tolerance: float,
    minimum_weight: float,
    separation_threshold: float,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    coefficients = jnp.zeros((design.shape[1],), dtype=response.dtype)
    converged = jnp.asarray(False)
    iteration_count = jnp.asarray(maximum_iterations, dtype=jnp.int32)
    covariance = jnp.eye(response.shape[0], dtype=response.dtype)
    precision_design = design
    normal = jnp.swapaxes(design, -1, -2) @ design
    for iteration in range(maximum_iterations):
        linear_predictor = design @ coefficients
        probability = jnn.sigmoid(linear_predictor)
        weight = jnp.maximum(probability * (1.0 - probability), minimum_weight)
        working_response = linear_predictor + (response - probability) / weight
        covariance = (
            jnp.diag(1.0 / weight)
            + relatedness_scale * relationship
            + ridge * jnp.eye(response.shape[0], dtype=response.dtype)
        )
        precision_design = _spd_solve(covariance, design)
        precision_response = _spd_solve(covariance, working_response)
        normal = jnp.swapaxes(design, -1, -2) @ precision_design
        normal = normal + ridge * jnp.eye(normal.shape[0], dtype=normal.dtype)
        proposed = _spd_solve(normal, jnp.swapaxes(design, -1, -2) @ precision_response)
        now = jnp.max(jnp.abs(proposed - coefficients)) <= tolerance * (
            1.0 + jnp.max(jnp.abs(coefficients))
        )
        coefficients = proposed
        if bool(np.asarray(now)):
            iteration_count = jnp.asarray(iteration + 1, dtype=jnp.int32)
            converged = jnp.asarray(True)
            break
        if bool(np.asarray(jnp.max(jnp.abs(proposed)) >= separation_threshold)):
            iteration_count = jnp.asarray(iteration + 1, dtype=jnp.int32)
            break
    probability = jnn.sigmoid(design @ coefficients)
    class_separated = (
        jnp.min(jnp.where(response > 0.5, probability, 1.0)) > 1.0 - 1e-6
    ) & (jnp.max(jnp.where(response < 0.5, probability, 0.0)) < 1e-6)
    separated = (jnp.max(jnp.abs(coefficients)) >= separation_threshold) | class_separated
    return (
        coefficients,
        probability,
        covariance,
        precision_design,
        normal,
        converged,
        iteration_count,
        separated,
    )


def binary_association(
    cohort: GenotypeCohort,
    outcome: ArrayLike,
    /,
    *,
    covariates: ArrayLike | None = None,
    kinship: ArrayLike | None = None,
    loco: LOCOKinship | None = None,
    relatedness_scale: float = 1.0,
    ridge: float = 1e-6,
    maximum_iterations: int = 64,
    tolerance: float = 1e-8,
    minimum_weight: float = 1e-6,
    separation_threshold: float = 30.0,
    imbalance_threshold: float = 0.05,
) -> BinaryAssociationResult:
    """Binary PQL score association with explicit imbalance and separation status."""
    if not isinstance(cohort, GenotypeCohort):
        raise TypeError("cohort must be a GenotypeCohort.")
    if loco is not None and not isinstance(loco, LOCOKinship):
        raise TypeError("loco must be a LOCOKinship or None.")
    if loco is not None and kinship is not None:
        raise ValueError("kinship and loco are mutually exclusive.")
    response = jnp.asarray(outcome)
    if response.shape != (cohort.sample_count,):
        raise ValueError("outcome must contain one value per sample.")
    if not jnp.issubdtype(response.dtype, jnp.inexact):
        response = response.astype(float)
    host_response = np.asarray(response)
    if not np.all(np.isfinite(host_response)) or np.any(
        (host_response != 0.0) & (host_response != 1.0)
    ):
        raise ValueError("outcome must contain finite binary values zero or one.")
    design = _design(covariates, cohort.sample_count, response.dtype)
    if not bool(np.all(np.isfinite(np.asarray(design)))):
        raise ValueError("covariates must be finite.")
    kinship_ = None if kinship is None else jnp.asarray(kinship, dtype=response.dtype)
    if kinship_ is not None:
        _validate_relationship(kinship_, cohort.sample_count)
    if loco is not None:
        if loco.chromosome_labels != cohort.chromosome_labels:
            raise ValueError("LOCO chromosome labels must match the cohort.")
        for matrix in np.asarray(loco.relationship_matrices):
            _validate_relationship(jnp.asarray(matrix), cohort.sample_count)
    parameters = (
        float(relatedness_scale),
        float(ridge),
        int(maximum_iterations),
        float(tolerance),
        float(minimum_weight),
        float(separation_threshold),
        float(imbalance_threshold),
    )
    scale, ridge_, iterations, tolerance_, minimum, separation_limit, imbalance_limit = (
        parameters
    )
    if (
        not np.isfinite(scale)
        or scale < 0.0
        or not np.isfinite(ridge_)
        or ridge_ <= 0.0
        or iterations < 1
        or not np.isfinite(tolerance_)
        or tolerance_ <= 0.0
        or not np.isfinite(minimum)
        or minimum <= 0.0
        or minimum >= 0.25
        or not np.isfinite(separation_limit)
        or separation_limit <= 0.0
        or not np.isfinite(imbalance_limit)
        or imbalance_limit < 0.0
        or imbalance_limit >= 0.5
    ):
        raise ValueError("Invalid binary association numerical policy.")
    case_count = jnp.sum(response).astype(jnp.int32)
    control_count = jnp.asarray(cohort.sample_count, dtype=jnp.int32) - case_count
    imbalanced = (
        jnp.minimum(case_count, control_count) < imbalance_limit * cohort.sample_count
    )
    dosage = cohort.dosage
    effects: list[Array] = []
    errors: list[Array] = []
    statistics: list[Array] = []
    p_values: list[Array] = []
    null_probabilities: list[Array] = []
    null_iterations: list[Array] = []
    separated_values: list[Array] = []
    effective_values: list[Array] = []
    valid_values: list[Array] = []
    statuses: list[Array] = []
    null_fit_cache: dict[
        int,
        tuple[Array, Array, Array, Array, Array, Array, Array, Array],
    ] = {}
    for variant in range(cohort.variant_count):
        relationship_key = (
            int(np.asarray(cohort.chromosome_index[variant])) if loco is not None else 0
        )
        null_fit = null_fit_cache.get(relationship_key)
        if null_fit is None:
            relationship = _relationship_for_variant(cohort, variant, kinship_, loco)
            null_fit = _fit_binary_null(
                response,
                design,
                relationship,
                relatedness_scale=scale,
                ridge=ridge_,
                maximum_iterations=iterations,
                tolerance=tolerance_,
                minimum_weight=minimum,
                separation_threshold=separation_limit,
            )
            null_fit_cache[relationship_key] = null_fit
        (
            null_coefficients,
            probability,
            covariance,
            precision_design,
            normal,
            converged,
            iteration_count,
            separated,
        ) = null_fit
        observed_count = jnp.sum(cohort.observed[variant]).astype(jnp.int32)
        mean = jnp.sum(
            jnp.where(cohort.observed[variant], dosage[variant], 0.0)
        ) / jnp.maximum(observed_count, 1)
        genotype = jnp.where(cohort.observed[variant], dosage[variant], mean)
        precision_genotype = _spd_solve(covariance, genotype)
        nuisance = _spd_solve(normal, jnp.swapaxes(design, -1, -2) @ precision_genotype)
        residual_genotype = genotype - design @ nuisance
        precision_residual_genotype = _spd_solve(covariance, residual_genotype)
        information = residual_genotype @ precision_residual_genotype
        working_residual = (response - probability) / jnp.maximum(
            probability * (1.0 - probability), minimum
        )
        precision_working_residual = _spd_solve(covariance, working_residual)
        score = residual_genotype @ precision_working_residual
        effect = score / jnp.maximum(information, 1e-30)
        error = 1.0 / jnp.sqrt(jnp.maximum(information, 1e-30))
        statistic = score / jnp.sqrt(jnp.maximum(information, 1e-30))
        p_value = jsp.special.erfc(jnp.abs(statistic) / jnp.sqrt(2.0))
        finite = (
            jnp.all(jnp.isfinite(null_coefficients))
            & jnp.isfinite(effect)
            & jnp.isfinite(error)
            & jnp.isfinite(statistic)
            & jnp.isfinite(p_value)
        )
        valid = (
            (case_count > 0)
            & (control_count > 0)
            & converged
            & ~separated
            & (observed_count >= 2)
            & (information > 1e-12)
            & finite
        )
        status = jnp.where(
            case_count == 0,
            int(AssociationStatus.NO_CASES),
            jnp.where(
                control_count == 0,
                int(AssociationStatus.NO_CONTROLS),
                jnp.where(
                    separated,
                    int(AssociationStatus.SEPARATION),
                    jnp.where(
                        ~converged,
                        int(AssociationStatus.MAXIMUM_ITERATIONS),
                        jnp.where(
                            ~finite,
                            int(AssociationStatus.NONFINITE),
                            jnp.where(
                                (observed_count < 2),
                                int(AssociationStatus.INSUFFICIENT_SAMPLES),
                                jnp.where(
                                    information <= 1e-12,
                                    int(AssociationStatus.MONOMORPHIC),
                                    int(AssociationStatus.SUCCESS),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        effects.append(jnp.where(valid, effect, jnp.nan))
        errors.append(jnp.where(valid, error, jnp.nan))
        statistics.append(jnp.where(valid, statistic, jnp.nan))
        p_values.append(jnp.where(valid, p_value, jnp.nan))
        null_probabilities.append(probability)
        null_iterations.append(iteration_count)
        separated_values.append(separated)
        effective_values.append(observed_count)
        valid_values.append(valid)
        statuses.append(status)
    effect = jnp.stack(effects)
    error = jnp.stack(errors)
    statistic = jnp.stack(statistics)
    p_value = jnp.stack(p_values)
    null_probability = jnp.stack(null_probabilities)
    iteration_values = jnp.stack(null_iterations)
    separated_array = jnp.stack(separated_values)
    effective = jnp.stack(effective_values)
    valid = jnp.stack(valid_values)
    status = jnp.stack(statuses)
    evidence = jnp.stack(
        (
            effective,
            jnp.full_like(effective, case_count),
            jnp.full_like(effective, control_count),
            iteration_values,
        ),
        axis=-1,
    )
    return BinaryAssociationResult(
        effect,
        error,
        statistic,
        p_value,
        null_probability,
        iteration_values,
        case_count,
        control_count,
        imbalanced,
        separated_array,
        effective,
        valid,
        status,
        evidence,
        loco is not None,
        _association_contract("binary-pql-score-association", binary=True),
    )


__all__ = [
    "AssociationStatus",
    "BinaryAssociationResult",
    "LOCOKinship",
    "QuantitativeAssociationResult",
    "UNSUPPORTED_ASSOCIATION_ANALYSES",
    "binary_association",
    "leave_one_chromosome_out_kinship",
    "quantitative_association",
]
