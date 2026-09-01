#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...pgm import (
    DenseTableFactorGroup,
    DiscreteFactorGraph,
    DiscreteVariableGroup,
    enumerate_factor_graph,
    ExactFactorGraphResult,
    VariableSelection,
)
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._cohort import GenotypeCohort


class PedigreeStatus(IntEnum):
    SUCCESS = 0
    NO_CANDIDATES = 1
    CAPACITY_EXCEEDED = 2
    NO_INFORMATIVE_VARIANTS = 3
    INFEASIBLE = 4
    NONFINITE = 5


class PedigreeInferencePlan(StrictModule):
    maximum_parent_pairs: int = eqx.field(static=True)
    maximum_configurations: int = eqx.field(static=True)
    minimum_parent_age: float = eqx.field(static=True)
    mutation_probability: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_parent_pairs: int,
        /,
        *,
        maximum_configurations: int | None = None,
        minimum_parent_age: float = 0.0,
        mutation_probability: float = 1e-6,
    ):
        capacity = int(maximum_parent_pairs)
        enumeration_capacity = (
            capacity if maximum_configurations is None else int(maximum_configurations)
        )
        minimum_age = float(minimum_parent_age)
        mutation = float(mutation_probability)
        if capacity < 1 or enumeration_capacity < capacity:
            raise ValueError(
                "maximum_parent_pairs must be positive and no larger than "
                "maximum_configurations."
            )
        if not np.isfinite(minimum_age) or minimum_age < 0.0:
            raise ValueError("minimum_parent_age must be finite and non-negative.")
        if not np.isfinite(mutation) or mutation < 0.0 or mutation >= 1.0:
            raise ValueError("mutation_probability must lie in [0, 1).")
        self.maximum_parent_pairs = capacity
        self.maximum_configurations = enumeration_capacity
        self.minimum_parent_age = minimum_age
        self.plan_id = canonical_fingerprint(
            {
                "kind": "pedigree-inference-plan",
                "maximum_parent_pairs": capacity,
                "maximum_configurations": enumeration_capacity,
                "minimum_parent_age": minimum_age,
                "mutation_probability": mutation,
            }
        )
        self.mutation_probability = mutation


class PedigreeInferenceResult(StrictModule):
    parent_pairs: Array
    posterior_probability: Array
    map_parent_pair: Array
    informative_variants: Array
    candidate_count: Array
    valid: Array
    status: Array
    evidence: Array
    pgm_results: tuple[ExactFactorGraphResult, ...]
    contract: BioinformaticsMethodContract = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid & (self.status == int(PedigreeStatus.SUCCESS))


def _pedigree_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "bounded-mendelian-pedigree-inference",
        MethodKind.EXACT_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.EXACT_AD,
        OutputKind.GRAPH,
        conditioning_statement=(
            "Conditioned on candidate parent constraints and independent biallelic "
            "genotype posterior evidence across informative diploid variants."
        ),
        truncation_statement="Candidate parent pairs are never truncated.",
        capacity_semantics=(
            "The complete candidate set is preflighted against maximum_parent_pairs; "
            "capacity failure is returned before factor-graph inference."
        ),
        assumptions=(
            "Autosomal diploid Mendelian transmission.",
            "Conditional independence of variants given a parent pair.",
        ),
        nondifferentiable_outputs=("parent_pairs", "map_parent_pair", "status", "valid"),
    )


def _transmission_table(dtype: jnp.dtype, mutation_probability: float) -> Array:
    dosage = jnp.arange(3, dtype=dtype)
    q = dosage / 2.0
    first = q[:, None]
    second = q[None, :]
    table = jnp.stack(
        (
            (1.0 - first) * (1.0 - second),
            first * (1.0 - second) + (1.0 - first) * second,
            first * second,
        ),
        axis=-1,
    )
    mutation = jnp.asarray(mutation_probability, dtype=dtype)
    return (1.0 - mutation) * table + mutation / 3.0


def _parent_pair_log_likelihood(
    cohort: GenotypeCohort,
    child: int,
    first_parent: int,
    second_parent: int,
    transmission: Array,
    /,
) -> tuple[Array, Array]:
    informative = (
        cohort.observed[:, child]
        & cohort.observed[:, first_parent]
        & cohort.observed[:, second_parent]
        & (cohort.ploidy[:, child] == 2)
        & (cohort.ploidy[:, first_parent] == 2)
        & (cohort.ploidy[:, second_parent] == 2)
    )
    child_probability = cohort.genotype_probabilities[:, child, :3]
    first_probability = cohort.genotype_probabilities[:, first_parent, :3]
    second_probability = cohort.genotype_probabilities[:, second_parent, :3]
    # Sum over both uncertain parent genotypes and the uncertain child genotype.
    per_variant = jnp.sum(
        first_probability[:, :, None, None]
        * second_probability[:, None, :, None]
        * transmission[None, :, :, :]
        * child_probability[:, None, None, :],
        axis=(1, 2, 3),
    )
    log_likelihood = jnp.sum(
        jnp.where(informative, jnp.log(jnp.maximum(per_variant, 1e-300)), 0.0)
    )
    return log_likelihood, jnp.sum(informative).astype(jnp.int32)


def infer_pedigree(
    cohort: GenotypeCohort,
    plan: PedigreeInferencePlan,
    /,
    *,
    candidate_parent: ArrayLike | None = None,
    birth_time: ArrayLike | None = None,
    sex: ArrayLike | None = None,
) -> PedigreeInferenceResult:
    """Infer an exact posterior over every bounded candidate parent pair.

    ``candidate_parent[child, parent]`` can exclude impossible links. ``birth_time``
    increases forward in time, so a parent must precede a child by at least the
    configured age. ``sex`` uses 0=unknown, 1=female, 2=male; when both parents have
    known sex, same-sex pairs are excluded without assigning maternal/paternal order.
    """
    if not isinstance(cohort, GenotypeCohort):
        raise TypeError("cohort must be a GenotypeCohort.")
    if not isinstance(plan, PedigreeInferencePlan):
        raise TypeError("plan must be a PedigreeInferencePlan.")
    sample_count = cohort.sample_count
    candidates = (
        np.ones((sample_count, sample_count), dtype=bool)
        if candidate_parent is None
        else np.array(candidate_parent, dtype=bool, copy=True)
    )
    if candidates.shape != (sample_count, sample_count):
        raise ValueError("candidate_parent must have shape (samples, samples).")
    np.fill_diagonal(candidates, False)
    if birth_time is not None:
        times = np.asarray(birth_time, dtype=float)
        if times.shape != (sample_count,) or not np.all(np.isfinite(times)):
            raise ValueError("birth_time must contain one finite value per sample.")
        old_enough = times[None, :] + plan.minimum_parent_age <= times[:, None]
        candidates &= old_enough
    sexes = (
        np.zeros((sample_count,), dtype=np.int32)
        if sex is None
        else np.asarray(sex, dtype=np.int32)
    )
    if sexes.shape != (sample_count,) or np.any((sexes < 0) | (sexes > 2)):
        raise ValueError("sex must use codes 0=unknown, 1=female, 2=male.")

    pair_lists: list[list[tuple[int, int]]] = []
    for child in range(sample_count):
        parents = np.flatnonzero(candidates[child]).tolist()
        pairs = [
            (first, second)
            for offset, first in enumerate(parents)
            for second in parents[offset + 1 :]
            if not (
                sexes[first] != 0 and sexes[second] != 0 and sexes[first] == sexes[second]
            )
        ]
        pair_lists.append(pairs)
    required = max((len(pairs) for pairs in pair_lists), default=0)
    shape = (sample_count, plan.maximum_parent_pairs)
    parent_pairs = np.full(shape + (2,), -1, dtype=np.int32)
    posterior = jnp.zeros(shape, dtype=cohort.genotype_probabilities.dtype)
    informative = jnp.zeros(shape, dtype=jnp.int32)
    candidate_count = jnp.asarray([len(pairs) for pairs in pair_lists], dtype=jnp.int32)
    if required > plan.maximum_parent_pairs:
        return PedigreeInferenceResult(
            jnp.asarray(parent_pairs),
            posterior,
            jnp.full((sample_count, 2), -1, dtype=jnp.int32),
            informative,
            candidate_count,
            jnp.asarray(False),
            jnp.asarray(int(PedigreeStatus.CAPACITY_EXCEEDED), dtype=jnp.int32),
            jnp.asarray((required, plan.maximum_parent_pairs), dtype=jnp.int32),
            (),
            _pedigree_contract(),
        )

    transmission = _transmission_table(
        cohort.genotype_probabilities.dtype, plan.mutation_probability
    )
    pgm_results: list[ExactFactorGraphResult] = []
    child_valid: list[Array] = []
    child_status: list[Array] = []
    maps: list[Array] = []
    for child, pairs in enumerate(pair_lists):
        if not pairs:
            child_valid.append(jnp.asarray(False))
            child_status.append(
                jnp.asarray(int(PedigreeStatus.NO_CANDIDATES), dtype=jnp.int32)
            )
            maps.append(jnp.asarray((-1, -1), dtype=jnp.int32))
            continue
        log_likelihoods: list[Array] = []
        counts: list[Array] = []
        for pair_index, (first, second) in enumerate(pairs):
            parent_pairs[child, pair_index] = (first, second)
            value, count = _parent_pair_log_likelihood(
                cohort, child, first, second, transmission
            )
            log_likelihoods.append(value)
            counts.append(count)
        values = jnp.stack(log_likelihoods)
        count_values = jnp.stack(counts)
        variable = DiscreteVariableGroup(
            f"parent-pair-{child}", num_states=len(pairs), shape=()
        )
        factor = DenseTableFactorGroup(
            (VariableSelection(variable, jnp.asarray((0,), dtype=jnp.int32)),),
            values[None, :],
        )
        graph = DiscreteFactorGraph((variable,), (factor,))
        pgm_result = enumerate_factor_graph(
            graph, max_configurations=plan.maximum_configurations
        )
        pgm_results.append(pgm_result)
        probabilities = pgm_result.variable_probabilities.values[: len(pairs)]
        posterior = posterior.at[child, : len(pairs)].set(probabilities)
        informative = informative.at[child, : len(pairs)].set(count_values)
        enough_evidence = jnp.any(count_values > 0)
        finite = jnp.all(jnp.isfinite(values))
        valid = pgm_result.successful & enough_evidence & finite
        status = jnp.where(
            ~finite,
            int(PedigreeStatus.NONFINITE),
            jnp.where(
                ~enough_evidence,
                int(PedigreeStatus.NO_INFORMATIVE_VARIANTS),
                jnp.where(
                    pgm_result.successful,
                    int(PedigreeStatus.SUCCESS),
                    int(PedigreeStatus.INFEASIBLE),
                ),
            ),
        ).astype(jnp.int32)
        best = jnp.argmax(probabilities)
        maps.append(jnp.asarray(parent_pairs[child])[best])
        child_valid.append(valid)
        child_status.append(status)

    valid_array = jnp.stack(child_valid) if child_valid else jnp.zeros((0,), dtype=bool)
    status_array = (
        jnp.stack(child_status) if child_status else jnp.zeros((0,), dtype=jnp.int32)
    )
    map_pairs = jnp.stack(maps) if maps else jnp.zeros((0, 2), dtype=jnp.int32)
    return PedigreeInferenceResult(
        jnp.asarray(parent_pairs),
        posterior,
        map_pairs,
        informative,
        candidate_count,
        valid_array,
        status_array,
        jnp.stack((candidate_count, jnp.max(informative, axis=1)), axis=-1),
        tuple(pgm_results),
        _pedigree_contract(),
    )


__all__ = [
    "PedigreeInferencePlan",
    "PedigreeInferenceResult",
    "PedigreeStatus",
    "infer_pedigree",
]
