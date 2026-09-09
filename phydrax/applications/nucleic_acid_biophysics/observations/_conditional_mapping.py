# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Predeclared Bernoulli observation-law ladder for mapped mutation profiles."""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....uq._map import find_map, MAPResult
from ....uq._posterior import ParameterSpace, PosteriorProblem
from ._mutation_profiles import MutationProfileBatch


MappingLawKind = Literal["binary-accessibility", "context", "hierarchical"]


class ConditionalMappingFit(StrictModule):
    optimization: MAPResult
    mutation_probability: Array
    per_profile_log_score: Array
    design_rank: Array
    singular_values: Array
    residual_correlation: Array
    maximum_absolute_residual_correlation: Array
    identifiable: Array
    observed_event_count: Array
    fit_profile_mask: Array
    model_id: str = eqx.field(static=True)
    parameter_names: tuple[str, ...] = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)


class ConditionalMutationLaw(StrictModule, NonTrainableState):
    """One fixed Bernoulli design; parameters remain external UQ coordinates."""

    batch: MutationProfileBatch
    design: Array
    prior_scale: Array
    parameter_names: tuple[str, ...] = eqx.field(static=True)
    kind: MappingLawKind = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    source_case_ids: tuple[str, ...] = eqx.field(static=True)
    parent_case_ids: tuple[str, ...] = eqx.field(static=True)
    independence_assumption: str = eqx.field(static=True)

    def __init__(
        self,
        batch: MutationProfileBatch,
        design: ArrayLike,
        parameter_names: tuple[str, ...],
        /,
        *,
        kind: MappingLawKind,
        prior_scale: ArrayLike,
        source_case_ids: tuple[str, ...],
        parent_case_ids: tuple[str, ...] = (),
    ):
        if not isinstance(batch, MutationProfileBatch):
            raise TypeError("batch must be a MutationProfileBatch.")
        if kind not in ("binary-accessibility", "context", "hierarchical"):
            raise ValueError("Unknown conditional mutation-law kind.")
        matrix = np.asarray(design, float)
        names = tuple(parameter_names)
        scales = np.asarray(prior_scale, float)
        expected = (batch.profile_count, batch.nucleotide_count, len(names))
        if matrix.shape != expected or not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"Conditional mapping design must be finite with shape {expected}."
            )
        if (
            not names
            or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name for name in names)
        ):
            raise ValueError("Parameter names must be unique non-empty strings.")
        if (
            scales.shape != (len(names),)
            or np.any(~np.isfinite(scales))
            or np.any(scales <= 0)
        ):
            raise ValueError(
                "Every mapping parameter requires a finite positive prior scale."
            )
        sources = tuple(source_case_ids)
        parents = tuple(parent_case_ids)
        if (
            not sources
            or len(set(sources)) != len(sources)
            or len(set(parents)) != len(parents)
            or any(
                not isinstance(value, str) or not value or value != value.strip()
                for value in (*sources, *parents)
            )
        ):
            raise ValueError(
                "Mapping features require canonical source case IDs and optional parents."
            )
        self.batch = batch
        self.design = jnp.asarray(matrix)
        self.prior_scale = jnp.asarray(scales)
        self.parameter_names = names
        self.kind = kind
        self.source_case_ids = sources
        self.parent_case_ids = parents
        self.independence_assumption = (
            "conditionally independent Bernoulli sites; residual correlations require "
            "a richer observation law"
        )
        self.model_id = canonical_fingerprint(
            {
                "kind": f"conditional-mutation-{kind}",
                "batch": batch.batch_fingerprint,
                "parameter_names": names,
                "prior_scale": scales.tolist(),
                "design": array_tree_fingerprint(matrix),
                "assumption": self.independence_assumption,
                "source_cases": sources,
                "parent_cases": parents,
            }
        )

    def logits(self, parameters: ArrayLike, /) -> Array:
        values = jnp.asarray(parameters)
        if values.shape != (len(self.parameter_names),):
            raise ValueError("Mapping parameters have an incompatible shape.")
        return jnp.einsum("nsf,f->ns", self.design, values)

    def mutation_probabilities(self, parameters: ArrayLike, /) -> Array:
        return jax.nn.sigmoid(self.logits(parameters))

    def per_profile_log_likelihood(self, parameters: ArrayLike, /) -> Array:
        logits = self.logits(parameters)
        mutation = self.batch.mutation.astype(logits.dtype)
        event = mutation * jax.nn.log_sigmoid(logits) + (
            1.0 - mutation
        ) * jax.nn.log_sigmoid(-logits)
        mask = self.batch.observed_mask & self.batch.analysis_mask[:, None]
        return jnp.sum(jnp.where(mask, event, 0.0), axis=-1)

    def posterior_problem(
        self,
        initial_parameters: ArrayLike | None = None,
        /,
        *,
        fit_profile_mask: ArrayLike | None = None,
    ) -> PosteriorProblem:
        initial = (
            jnp.zeros((len(self.parameter_names),), dtype=self.design.dtype)
            if initial_parameters is None
            else jnp.asarray(initial_parameters, dtype=self.design.dtype)
        )
        if initial.shape != (len(self.parameter_names),) or bool(
            jnp.any(~jnp.isfinite(initial))
        ):
            raise ValueError(
                "Initial mapping parameters must be one finite value per parameter."
            )
        selected = self.batch.analysis_profile_mask(fit_profile_mask)
        if not bool(jnp.any(selected)):
            raise ValueError(
                "Mapping inference requires observed profiles in the frozen fit split."
            )
        scale = self.prior_scale
        space = ParameterSpace(
            initial,
            log_prior=lambda value: -0.5 * jnp.sum((value / scale) ** 2),
        )
        return PosteriorProblem(
            space,
            lambda value: jnp.sum(
                jnp.where(selected, self.per_profile_log_likelihood(value), 0.0)
            ),
            predict=self.mutation_probabilities,
        )

    def fit(
        self,
        initial_parameters: ArrayLike | None = None,
        /,
        *,
        requested_use=None,
        fit_profile_mask: ArrayLike | None = None,
        max_steps: int = 500,
        gradient_tolerance: float = 1e-6,
    ) -> ConditionalMappingFit:
        use = {} if requested_use is None else dict(requested_use)
        use["training_use"] = True
        selected = self.batch.analysis_profile_mask(fit_profile_mask)
        if not bool(jnp.any(selected)):
            raise ValueError(
                "Mapping inference requires observed profiles in the frozen fit split."
            )
        self.batch.require_rights(use, profile_mask=selected)
        problem = self.posterior_problem(initial_parameters, fit_profile_mask=selected)
        result = find_map(
            problem,
            max_steps=max_steps,
            gradient_tolerance=gradient_tolerance,
            raise_on_failure=False,
        )
        probability = self.mutation_probabilities(result.parameters)
        mask = np.asarray(self.batch.observed_mask & selected[:, None])
        observed_design = np.asarray(self.design)[mask]
        if observed_design.shape[0] == 0:
            singular = np.zeros((0,), dtype=float)
            rank = 0
        else:
            singular = np.linalg.svd(observed_design, compute_uv=False)
            tolerance = (
                np.finfo(singular.dtype).eps * max(observed_design.shape) * singular[0]
            )
            rank = int(np.sum(singular > tolerance))
        correlation = _masked_residual_correlation(
            self.batch.mutation.astype(probability.dtype) - probability,
            self.batch.observed_mask & selected[:, None],
        )
        off_diagonal = ~jnp.eye(self.batch.nucleotide_count, dtype=bool)
        finite_off_diagonal = off_diagonal & jnp.isfinite(correlation)
        maximum = jnp.where(
            jnp.any(finite_off_diagonal),
            jnp.max(jnp.where(finite_off_diagonal, jnp.abs(correlation), -jnp.inf)),
            jnp.nan,
        )
        return ConditionalMappingFit(
            result,
            probability,
            self.per_profile_log_likelihood(result.parameters),
            jnp.asarray(rank, dtype=jnp.int32),
            jnp.asarray(singular),
            correlation,
            maximum,
            jnp.asarray(result.converged and rank == len(self.parameter_names)),
            jnp.sum(mask, dtype=jnp.int32),
            selected,
            self.model_id,
            self.parameter_names,
            self.batch.source_ids,
        )


class ConditionalMappingLadder(StrictModule, NonTrainableState):
    baseline: ConditionalMutationLaw
    context: ConditionalMutationLaw
    hierarchical: ConditionalMutationLaw
    ladder_id: str = eqx.field(static=True)
    source_case_ids: tuple[str, ...] = eqx.field(static=True)
    parent_case_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        baseline: ConditionalMutationLaw,
        context: ConditionalMutationLaw,
        hierarchical: ConditionalMutationLaw,
    ):
        laws = (baseline, context, hierarchical)
        if (
            tuple(law.kind for law in laws)
            != (
                "binary-accessibility",
                "context",
                "hierarchical",
            )
            or len({law.batch.batch_fingerprint for law in laws}) != 1
        ):
            raise ValueError(
                "A mapping ladder requires its three ordered laws on one batch."
            )
        self.baseline, self.context, self.hierarchical = laws
        if (
            len({law.source_case_ids for law in laws}) != 1
            or len({law.parent_case_ids for law in laws}) != 1
        ):
            raise ValueError("Every mapping-law level must share exact feature lineage.")
        self.source_case_ids = baseline.source_case_ids
        self.parent_case_ids = baseline.parent_case_ids
        self.ladder_id = canonical_fingerprint(
            {
                "kind": "conditional-mapping-ladder",
                "models": [law.model_id for law in laws],
                "source_cases": self.source_case_ids,
                "parent_cases": self.parent_case_ids,
            }
        )

    @property
    def laws(self) -> tuple[ConditionalMutationLaw, ...]:
        return self.baseline, self.context, self.hierarchical


def prepare_conditional_mapping_ladder(
    batch: MutationProfileBatch,
    accessibility: ArrayLike,
    /,
    *,
    nucleotide_features: ArrayLike,
    nucleotide_feature_names: tuple[str, ...],
    local_context_features: ArrayLike,
    local_context_feature_names: tuple[str, ...],
    condition_features: ArrayLike,
    condition_feature_names: tuple[str, ...],
    source_case_ids: tuple[str, ...],
    parent_case_ids: tuple[str, ...] = (),
    shared_prior_scale: float = 3.0,
    hierarchical_prior_scale: float = 1.0,
) -> ConditionalMappingLadder:
    """Build the fixed binary/context/hierarchical ladder before fitting.

    Structural accessibility and site features are supplied per declared construct;
    condition features are supplied per declared condition. Reagent, protocol,
    preparation, replicate, and batch terms use explicit treatment contrasts.
    """
    if not isinstance(batch, MutationProfileBatch):
        raise TypeError("batch must be a MutationProfileBatch.")
    constructs, sites = len(batch.construct_ids), batch.nucleotide_count
    accessibility_ = np.asarray(accessibility, float)
    nucleotide_ = np.asarray(nucleotide_features, float)
    local_ = np.asarray(local_context_features, float)
    condition_ = np.asarray(condition_features, float)
    nucleotide_names = tuple(nucleotide_feature_names)
    local_names = tuple(local_context_feature_names)
    condition_names = tuple(condition_feature_names)
    if (
        accessibility_.shape != (constructs, sites)
        or np.any(~np.isfinite(accessibility_))
        or np.any((accessibility_ != 0.0) & (accessibility_ != 1.0))
    ):
        raise ValueError(
            "Binary accessibility must be 0 or 1 for every construct and site."
        )
    if nucleotide_.shape != (constructs, sites, len(nucleotide_names)):
        raise ValueError("Nucleotide feature shape does not match its declared names.")
    if local_.shape != (constructs, sites, len(local_names)):
        raise ValueError("Local-context feature shape does not match its declared names.")
    if condition_.shape != (len(batch.condition_ids), len(condition_names)):
        raise ValueError(
            "Condition feature shape does not match declared conditions and names."
        )
    if any(np.any(~np.isfinite(value)) for value in (nucleotide_, local_, condition_)):
        raise ValueError("Mapping features must be finite.")
    supplied_names = nucleotide_names + local_names + condition_names
    if len(set(supplied_names)) != len(supplied_names) or any(
        not name for name in supplied_names
    ):
        raise ValueError(
            "Caller-supplied mapping feature names must be unique and non-empty."
        )
    shared_scale, group_scale = float(shared_prior_scale), float(hierarchical_prior_scale)
    if (
        not np.isfinite(shared_scale)
        or shared_scale <= 0
        or not np.isfinite(group_scale)
        or group_scale <= 0
    ):
        raise ValueError("Mapping prior scales must be finite and positive.")
    sources = tuple(source_case_ids)
    parents = tuple(parent_case_ids)
    if (
        not sources
        or len(set(sources)) != len(sources)
        or len(set(parents)) != len(parents)
        or any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in (*sources, *parents)
        )
    ):
        raise ValueError(
            "Mapping feature lineage requires source case IDs and optional parents."
        )

    construct_index = np.asarray(batch.construct_index)
    condition_index = np.asarray(batch.condition_index)
    profile_accessibility = accessibility_[construct_index]
    base_design = np.stack(
        (np.ones_like(profile_accessibility), profile_accessibility), axis=-1
    )
    base_names = ("intercept", "binary-accessibility")

    site_parts = [base_design, nucleotide_[construct_index], local_[construct_index]]
    site_names = (
        base_names
        + tuple(f"nucleotide:{name}" for name in nucleotide_names)
        + tuple(f"local-context:{name}" for name in local_names)
    )
    if condition_names:
        selected = condition_[condition_index]
        site_parts.append(
            np.broadcast_to(
                selected[:, None, :], (batch.profile_count, sites, len(condition_names))
            )
        )
        site_names += tuple(f"condition:{name}" for name in condition_names)
    for prefix, indices, labels in (
        ("reagent", batch.reagent_index, batch.reagent_ids),
        ("protocol", batch.protocol_index, batch.protocol_ids),
    ):
        contrasts, names = _treatment_contrasts(indices, labels, sites, prefix)
        if contrasts.shape[-1]:
            site_parts.append(contrasts)
            site_names += names
    context_design = np.concatenate(site_parts, axis=-1)

    hierarchy_parts = [context_design]
    hierarchy_names = site_names
    for prefix, indices, labels in (
        ("preparation", batch.preparation_index, batch.preparation_ids),
        ("replicate", batch.replicate_index, batch.replicate_ids),
        ("batch", batch.batch_index, batch.batch_ids),
    ):
        contrasts, names = _treatment_contrasts(indices, labels, sites, prefix)
        if contrasts.shape[-1]:
            hierarchy_parts.append(contrasts)
            hierarchy_names += names
    hierarchy_design = np.concatenate(hierarchy_parts, axis=-1)
    return ConditionalMappingLadder(
        ConditionalMutationLaw(
            batch,
            base_design,
            base_names,
            kind="binary-accessibility",
            prior_scale=np.full(len(base_names), shared_scale),
            source_case_ids=sources,
            parent_case_ids=parents,
        ),
        ConditionalMutationLaw(
            batch,
            context_design,
            site_names,
            kind="context",
            prior_scale=np.full(len(site_names), shared_scale),
            source_case_ids=sources,
            parent_case_ids=parents,
        ),
        ConditionalMutationLaw(
            batch,
            hierarchy_design,
            hierarchy_names,
            kind="hierarchical",
            prior_scale=np.concatenate(
                (
                    np.full(len(site_names), shared_scale),
                    np.full(len(hierarchy_names) - len(site_names), group_scale),
                )
            ),
            source_case_ids=sources,
            parent_case_ids=parents,
        ),
    )


def _treatment_contrasts(indices, labels, site_count: int, prefix: str):
    count = len(labels)
    if count <= 1:
        return np.zeros((len(indices), site_count, 0)), ()
    rows = np.asarray(indices, dtype=np.int32)
    contrasts = np.stack(
        tuple(rows == index for index in range(1, count)), axis=-1
    ).astype(float)
    return (
        np.broadcast_to(contrasts[:, None, :], (rows.size, site_count, count - 1)),
        tuple(f"{prefix}:{label}" for label in labels[1:]),
    )


def _masked_residual_correlation(residual: Array, mask: Array) -> Array:
    weights = mask.astype(residual.dtype)
    count = weights.T @ weights
    sums = (residual * weights).T @ weights
    pair_mean = jnp.where(count > 0, sums / count, 0.0)
    centered_left = residual[:, :, None] - pair_mean[None, :, :]
    centered_right = residual[:, None, :] - pair_mean.T[None, :, :]
    pair_mask = weights[:, :, None] * weights[:, None, :]
    covariance = jnp.sum(centered_left * centered_right * pair_mask, axis=0)
    left_variance = jnp.sum(centered_left**2 * pair_mask, axis=0)
    right_variance = jnp.sum(centered_right**2 * pair_mask, axis=0)
    denominator = jnp.sqrt(left_variance * right_variance)
    return jnp.where((count >= 2) & (denominator > 0), covariance / denominator, jnp.nan)


__all__ = [
    "ConditionalMappingFit",
    "ConditionalMappingLadder",
    "ConditionalMutationLaw",
    "MappingLawKind",
    "prepare_conditional_mapping_ladder",
]
