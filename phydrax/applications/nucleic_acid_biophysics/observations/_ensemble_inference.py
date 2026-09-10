# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Finite condition-dependent structural mixtures on externally mapped reads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....pgm import (
    DenseTableFactorGroup,
    DiscreteFactorGraph,
    DiscreteVariableGroup,
    prepare_exact_factor_graph,
    PreparedExactFactorGraph,
    run_exact_factor_graph,
    VariableSelection,
)
from ....uq._map import find_map, MAPResult
from ....uq._posterior import (
    IdentityBijector,
    ParameterSpace,
    PosteriorProblem,
    SimplexBijector,
)
from ..secondary_kinetics import SecondaryStructureState
from ._mutation_profiles import MutationProfileBatch


PopulationModelKind = Literal["free-simplex", "logit-linear"]


@dataclass(frozen=True, slots=True)
class StructuralEnsembleHypothesis:
    """Finite, externally motivated states; no structure is inferred from its own reads."""

    state_ids: tuple[str, ...]
    structures: tuple[SecondaryStructureState, ...]
    feature_definition_id: str
    source_ids: tuple[str, ...]
    source_case_ids: tuple[str, ...]
    parent_case_ids: tuple[str, ...]

    def __post_init__(self):
        if (
            not isinstance(self.state_ids, tuple)
            or len(self.state_ids) < 2
            or len(self.structures) != len(self.state_ids)
            or len(set(self.state_ids)) != len(self.state_ids)
            or any(not value or value != value.strip() for value in self.state_ids)
        ):
            raise ValueError(
                "An ensemble hypothesis needs at least two unique named states."
            )
        if any(
            not isinstance(state, SecondaryStructureState) for state in self.structures
        ):
            raise TypeError("structures must contain SecondaryStructureState values.")
        constructs = {state.construct.fingerprint() for state in self.structures}
        if len(constructs) != 1:
            raise ValueError("Every structural state must use the same exact construct.")
        for name, values, allow_empty in (
            ("feature definition ID", (self.feature_definition_id,), False),
            ("source IDs", self.source_ids, False),
            ("source case IDs", self.source_case_ids, False),
            ("parent case IDs", self.parent_case_ids, True),
        ):
            if (
                not isinstance(values, tuple)
                or (not allow_empty and not values)
                or len(set(values)) != len(values)
                or any(
                    not isinstance(value, str) or not value or value != value.strip()
                    for value in values
                )
            ):
                raise ValueError(f"{name} must contain canonical unique identifiers.")

    @property
    def hypothesis_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "structural-ensemble-hypothesis",
                "states": list(
                    zip(
                        self.state_ids,
                        (state.fingerprint() for state in self.structures),
                        strict=True,
                    )
                ),
                "feature_definition": self.feature_definition_id,
                "sources": self.source_ids,
                "source_cases": self.source_case_ids,
                "parent_cases": self.parent_case_ids,
            }
        )

    def binary_accessibility(self) -> Array:
        """Return unpaired=1/paired=0 features on the declared construct order."""
        count = self.structures[0].construct.nucleotide_count
        values = np.ones((len(self.structures), count), dtype=float)
        for row, state in enumerate(self.structures):
            for first, second in state.numeric_pairs:
                values[row, first] = 0.0
                values[row, second] = 0.0
        return jnp.asarray(values)


class ConditionPopulationModel(StrictModule, NonTrainableState):
    """Declared free-simplex or logit-linear condition population model."""

    design: Array
    condition_ids: tuple[str, ...] = eqx.field(static=True)
    state_ids: tuple[str, ...] = eqx.field(static=True)
    feature_names: tuple[str, ...] = eqx.field(static=True)
    kind: PopulationModelKind = eqx.field(static=True)
    reference_state_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        condition_ids: tuple[str, ...],
        state_ids: tuple[str, ...],
        /,
        *,
        kind: PopulationModelKind,
        design: ArrayLike | None = None,
        feature_names: tuple[str, ...] = (),
    ):
        conditions, states, names = (
            tuple(condition_ids),
            tuple(state_ids),
            tuple(feature_names),
        )
        if (
            not conditions
            or len(set(conditions)) != len(conditions)
            or any(not value for value in conditions)
        ):
            raise ValueError("Condition IDs must be unique and non-empty.")
        if (
            len(states) < 2
            or len(set(states)) != len(states)
            or any(not value for value in states)
        ):
            raise ValueError(
                "Population models require at least two unique named states."
            )
        if kind == "free-simplex":
            if design is not None or names:
                raise ValueError(
                    "A free-simplex population model has no condition design."
                )
            matrix = np.zeros((len(conditions), 0), dtype=float)
        elif kind == "logit-linear":
            matrix = np.asarray(design, float)
            if (
                matrix.shape != (len(conditions), len(names))
                or not names
                or np.any(~np.isfinite(matrix))
            ):
                raise ValueError(
                    "Logit-linear population design must match conditions and feature names."
                )
            if len(set(names)) != len(names) or any(not name for name in names):
                raise ValueError("Population feature names must be unique and non-empty.")
        else:
            raise ValueError("Unknown condition population-model kind.")
        self.condition_ids, self.state_ids, self.feature_names = conditions, states, names
        self.kind = kind
        self.design = jnp.asarray(matrix)
        self.reference_state_id = states[-1]
        self.model_id = canonical_fingerprint(
            {
                "kind": f"condition-population-{kind}",
                "conditions": conditions,
                "states": states,
                "features": names,
                "design": array_tree_fingerprint(matrix),
                "reference_state": states[-1],
            }
        )

    @property
    def raw_shape(self) -> tuple[int, ...]:
        if self.kind == "free-simplex":
            return len(self.condition_ids), len(self.state_ids) - 1
        return len(self.feature_names), len(self.state_ids) - 1

    def populations(self, parameters: ArrayLike, /) -> Array:
        values = jnp.asarray(parameters)
        if self.kind == "free-simplex":
            expected = (len(self.condition_ids), len(self.state_ids))
            if values.shape != expected:
                raise ValueError(
                    f"Free condition populations must have shape {expected}."
                )
            return values
        if values.shape != self.raw_shape:
            raise ValueError(f"Population coefficients must have shape {self.raw_shape}.")
        logits = self.design @ values
        return jax.nn.softmax(
            jnp.concatenate((logits, jnp.zeros_like(logits[:, :1])), axis=-1), axis=-1
        )


@dataclass(frozen=True, slots=True)
class EnsembleDiagnosticPolicy:
    minimum_state_separation: float
    minimum_state_population: float
    singular_relative_tolerance: float
    maximum_residual_correlation: float

    def __post_init__(self):
        values = tuple(
            float(value)
            for value in (
                self.minimum_state_separation,
                self.minimum_state_population,
                self.singular_relative_tolerance,
                self.maximum_residual_correlation,
            )
        )
        if (
            any(not np.isfinite(value) or value < 0 for value in values)
            or values[1] >= 1
            or values[2] >= 1
            or values[3] > 1
        ):
            raise ValueError(
                "Ensemble diagnostic thresholds are outside their finite domains."
            )
        for name, value in zip(
            (
                "minimum_state_separation",
                "minimum_state_population",
                "singular_relative_tolerance",
                "maximum_residual_correlation",
            ),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, value)


class EnsemblePosteriorPrediction(StrictModule):
    condition_index: Array
    state_population: Array
    state_mutation_probability: Array
    mutation_probability: Array
    mutation_covariance: Array
    model_id: str = eqx.field(static=True)
    distribution_assumption: str = eqx.field(static=True)

    def sample(self, key: Array, /, *, sample_count: int) -> Array:
        count = int(sample_count)
        if count <= 0:
            raise ValueError("sample_count must be positive.")
        state_key, mutation_key = jr.split(key)
        state = jr.categorical(
            state_key,
            jnp.log(self.state_population),
            shape=(count, self.state_population.shape[0]),
            axis=-1,
        )
        probability = jnp.take_along_axis(
            jnp.broadcast_to(
                self.state_mutation_probability[None],
                (count, *self.state_mutation_probability.shape),
            ),
            state[:, :, None, None],
            axis=2,
        )[:, :, 0, :]
        return jr.bernoulli(mutation_key, probability).astype(jnp.int8)


class PermutationInvariantEnsembleSummary(StrictModule):
    sorted_condition_populations: Array
    sorted_state_mean_mutation: Array
    sorted_pairwise_separation: Array
    posterior_predictive_mean: Array
    summary_id: str = eqx.field(static=True)


class EnsembleDiagnostics(StrictModule):
    pairwise_state_separation: Array
    equivalent_state_pairs: Array
    unsupported_states: Array
    singular_values: Array
    singular_directions: Array
    local_rank: Array
    parameter_count: int = eqx.field(static=True)
    residual_correlation_excess: Array
    profile_mask: Array
    maximum_absolute_residual_correlation: Array
    residual_assumption_supported: Array
    optimization_converged: Array
    numerical_valid: Array
    support_complete: Array
    unique_state_interpretation_supported: Array
    policy: EnsembleDiagnosticPolicy = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    diagnostics_id: str = eqx.field(static=True)


class FiniteEnsembleFit(StrictModule):
    optimization: MAPResult
    prediction: EnsemblePosteriorPrediction
    per_profile_log_score: Array
    state_posterior: Array
    diagnostics: EnsembleDiagnostics
    summary: PermutationInvariantEnsembleSummary
    fit_profile_mask: Array
    model_id: str = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)


class FiniteStructuralEnsembleModel(StrictModule, NonTrainableState):
    """Exact finite latent-state mixture with UQ-constrained populations.

    State response logits are condition-independent. Caller-supplied
    ``observation_offset`` carries the selected context/hierarchical mapping law;
    its identity is mandatory. The site-factorization is an explicit approximation,
    and unexplained cross-site dependence is reported rather than absorbed into an
    unqualified uncertainty multiplier.
    """

    batch: MutationProfileBatch
    observation_offset: Array
    response_prior_mean: Array
    population_model: ConditionPopulationModel
    latent_state_graph: PreparedExactFactorGraph
    hypothesis: StructuralEnsembleHypothesis = eqx.field(static=True)
    mapping_law_id: str = eqx.field(static=True)
    observation_offset_source_case_ids: tuple[str, ...] = eqx.field(static=True)
    observation_offset_parent_case_ids: tuple[str, ...] = eqx.field(static=True)
    observation_offset_derivation_id: str = eqx.field(static=True)
    derivation_id: str = eqx.field(static=True)
    response_prior_scale: float = eqx.field(static=True)
    population_prior_scale: float = eqx.field(static=True)
    dirichlet_concentration: float = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    identification_convention: str = eqx.field(static=True)
    independence_assumption: str = eqx.field(static=True)

    def __init__(
        self,
        batch: MutationProfileBatch,
        hypothesis: StructuralEnsembleHypothesis,
        population_model: ConditionPopulationModel,
        observation_offset: ArrayLike,
        response_prior_mean: ArrayLike,
        /,
        *,
        mapping_law_id: str,
        observation_offset_source_case_ids: tuple[str, ...],
        observation_offset_parent_case_ids: tuple[str, ...] = (),
        response_prior_scale: float,
        population_prior_scale: float = 2.0,
        dirichlet_concentration: float = 1.0,
    ):
        if not isinstance(batch, MutationProfileBatch) or not isinstance(
            hypothesis, StructuralEnsembleHypothesis
        ):
            raise TypeError(
                "Finite ensembles require a mutation batch and structural hypothesis."
            )
        if not isinstance(population_model, ConditionPopulationModel):
            raise TypeError("population_model must be a ConditionPopulationModel.")
        if (
            population_model.condition_ids != batch.condition_ids
            or population_model.state_ids != hypothesis.state_ids
        ):
            raise ValueError(
                "Population condition/state support must match the batch and hypothesis."
            )
        construct_ids = {state.construct.fingerprint() for state in hypothesis.structures}
        if len(batch.construct_ids) != 1 or construct_ids != set(batch.construct_ids):
            raise ValueError(
                "Finite structural inference currently requires one exact construct support."
            )
        states, sites = len(hypothesis.state_ids), batch.nucleotide_count
        if hypothesis.structures[0].construct.nucleotide_count != sites:
            raise ValueError(
                "Structural and mapped nucleotide supports have different sizes."
            )
        offset = np.asarray(observation_offset, float)
        prior_mean = np.asarray(response_prior_mean, float)
        if offset.shape != (batch.profile_count, sites) or np.any(~np.isfinite(offset)):
            raise ValueError(
                "Observation offsets must be finite on every profile and nucleotide."
            )
        if prior_mean.shape != (states, sites) or np.any(~np.isfinite(prior_mean)):
            raise ValueError(
                "State-response prior means must be finite on every state and nucleotide."
            )
        if (
            not isinstance(mapping_law_id, str)
            or not mapping_law_id
            or mapping_law_id != mapping_law_id.strip()
        ):
            raise ValueError(
                "mapping_law_id must identify the preselected observation law."
            )
        offset_sources = tuple(observation_offset_source_case_ids)
        offset_parents = tuple(observation_offset_parent_case_ids)
        if (
            not offset_sources
            or len(set(offset_sources)) != len(offset_sources)
            or len(set(offset_parents)) != len(offset_parents)
            or any(
                not isinstance(value, str) or not value or value != value.strip()
                for value in (*offset_sources, *offset_parents)
            )
        ):
            raise ValueError(
                "Observation-offset derivation requires canonical source case IDs "
                "and optional parent case IDs."
            )
        response_scale, population_scale, concentration = (
            float(response_prior_scale),
            float(population_prior_scale),
            float(dirichlet_concentration),
        )
        if any(
            not np.isfinite(value) or value <= 0
            for value in (response_scale, population_scale, concentration)
        ):
            raise ValueError(
                "Finite-mixture prior parameters must be finite and positive."
            )
        variable = DiscreteVariableGroup("structural-state", num_states=states)
        factor = DenseTableFactorGroup(
            (VariableSelection.all(variable),),
            np.zeros((1, states), dtype=float),
        )
        self.batch = batch
        self.hypothesis = hypothesis
        self.population_model = population_model
        self.observation_offset = jnp.asarray(offset)
        self.response_prior_mean = jnp.asarray(prior_mean)
        self.latent_state_graph = prepare_exact_factor_graph(
            DiscreteFactorGraph((variable,), (factor,))
        )
        self.mapping_law_id = mapping_law_id
        self.observation_offset_source_case_ids = offset_sources
        self.observation_offset_parent_case_ids = offset_parents
        self.observation_offset_derivation_id = canonical_fingerprint(
            {
                "kind": "conditional-ensemble-observation-offset-derivation",
                "mapping_law": mapping_law_id,
                "offset": array_tree_fingerprint(offset),
                "source_cases": offset_sources,
                "parent_cases": offset_parents,
            }
        )
        self.derivation_id = canonical_fingerprint(
            {
                "kind": "finite-structural-ensemble-derivation",
                "hypothesis": hypothesis.hypothesis_id,
                "observation_offset": self.observation_offset_derivation_id,
                "source_cases": tuple(
                    sorted(set(hypothesis.source_case_ids) | set(offset_sources))
                ),
                "parent_cases": tuple(
                    sorted(set(hypothesis.parent_case_ids) | set(offset_parents))
                ),
            }
        )
        self.response_prior_scale = response_scale
        self.population_prior_scale = population_scale
        self.dirichlet_concentration = concentration
        self.identification_convention = "externally-declared-structure-state-id-order"
        self.independence_assumption = (
            "conditionally independent Bernoulli sites within latent state; residual "
            "correlation is diagnostic evidence against this approximation"
        )
        self.model_id = canonical_fingerprint(
            {
                "kind": "finite-structural-mutation-ensemble",
                "batch": batch.batch_fingerprint,
                "hypothesis": hypothesis.hypothesis_id,
                "population_model": population_model.model_id,
                "mapping_law": mapping_law_id,
                "observation_offset": array_tree_fingerprint(offset),
                "response_prior_mean": array_tree_fingerprint(prior_mean),
                "response_prior_scale": response_scale,
                "observation_offset_derivation": self.observation_offset_derivation_id,
                "derivation": self.derivation_id,
                "population_prior_scale": population_scale,
                "dirichlet_concentration": concentration,
                "identification": self.identification_convention,
                "assumption": self.independence_assumption,
            }
        )

    def _parameter_space(self, initial_response_logits=None, initial_population=None):
        response = (
            self.response_prior_mean
            if initial_response_logits is None
            else jnp.asarray(initial_response_logits)
        )
        if response.shape != self.response_prior_mean.shape or bool(
            jnp.any(~jnp.isfinite(response))
        ):
            raise ValueError(
                "Initial state response logits have an incompatible shape or values."
            )
        raw_population = (
            jnp.zeros(self.population_model.raw_shape, dtype=response.dtype)
            if initial_population is None
            else jnp.asarray(initial_population, dtype=response.dtype)
        )
        if raw_population.shape != self.population_model.raw_shape or bool(
            jnp.any(~jnp.isfinite(raw_population))
        ):
            raise ValueError(
                "Initial population coordinates have an incompatible shape or values."
            )
        initial = {"response_logits": response, "population": raw_population}
        population_bijector = (
            SimplexBijector(len(self.hypothesis.state_ids))
            if self.population_model.kind == "free-simplex"
            else IdentityBijector()
        )
        bijectors = {
            "response_logits": IdentityBijector(),
            "population": population_bijector,
        }
        response_mean = self.response_prior_mean
        response_scale = self.response_prior_scale
        population_scale = self.population_prior_scale
        concentration = self.dirichlet_concentration
        free = self.population_model.kind == "free-simplex"

        def log_prior(parameters):
            value = -0.5 * jnp.sum(
                ((parameters["response_logits"] - response_mean) / response_scale) ** 2
            )
            if free:
                value = value + (concentration - 1.0) * jnp.sum(
                    jnp.log(parameters["population"])
                )
            else:
                value = value - 0.5 * jnp.sum(
                    (parameters["population"] / population_scale) ** 2
                )
            return value

        return ParameterSpace(initial, bijectors=bijectors, log_prior=log_prior)

    def condition_populations(self, parameters, /) -> Array:
        return self.population_model.populations(parameters["population"])

    def state_mutation_probabilities(
        self, parameters, /, *, observation_offset=None
    ) -> Array:
        offset = (
            self.observation_offset
            if observation_offset is None
            else jnp.asarray(observation_offset)
        )
        if offset.ndim != 2 or offset.shape[-1] != self.batch.nucleotide_count:
            raise ValueError("Observation offsets must have shape (profile, nucleotide).")
        response = jnp.asarray(parameters["response_logits"])
        if response.shape != self.response_prior_mean.shape:
            raise ValueError("State response logits have an incompatible shape.")
        return jax.nn.sigmoid(response[None, :, :] + offset[:, None, :])

    def _state_log_scores(self, parameters) -> Array:
        probability_logits = (
            parameters["response_logits"][None, :, :]
            + self.observation_offset[:, None, :]
        )
        mutation = self.batch.mutation[:, None, :].astype(probability_logits.dtype)
        event = mutation * jax.nn.log_sigmoid(probability_logits) + (
            1.0 - mutation
        ) * jax.nn.log_sigmoid(-probability_logits)
        observed = self.batch.observed_mask[:, None, :]
        state_likelihood = jnp.sum(jnp.where(observed, event, 0.0), axis=-1)
        populations = self.condition_populations(parameters)[self.batch.condition_index]
        return jnp.log(populations) + state_likelihood

    def per_profile_log_likelihood(self, parameters, /) -> Array:
        scores = self._state_log_scores(parameters)

        def normalize(value):
            result = run_exact_factor_graph(self.latent_state_graph, (value[None, :],))
            return result.log_normalizer

        normalized = jax.vmap(normalize)(scores)
        return jnp.where(self.batch.analysis_mask, normalized, 0.0)

    def state_posterior(self, parameters, /) -> Array:
        scores = self._state_log_scores(parameters)

        def marginal(value):
            result = run_exact_factor_graph(self.latent_state_graph, (value[None, :],))
            return result.variable_probabilities.values

        posterior = jax.vmap(marginal)(scores)
        return jnp.where(self.batch.analysis_mask[:, None], posterior, jnp.nan)

    def posterior_prediction(
        self,
        parameters,
        /,
        *,
        condition_index: ArrayLike | None = None,
        observation_offset: ArrayLike | None = None,
    ) -> EnsemblePosteriorPrediction:
        indices = (
            self.batch.condition_index
            if condition_index is None
            else jnp.asarray(condition_index)
        )
        offset = (
            self.observation_offset
            if observation_offset is None
            else jnp.asarray(observation_offset)
        )
        if (
            indices.ndim != 1
            or not jnp.issubdtype(indices.dtype, jnp.integer)
            or offset.shape != (indices.shape[0], self.batch.nucleotide_count)
        ):
            raise ValueError(
                "Prediction conditions and observation offsets must share a profile axis."
            )
        indices = eqx.error_if(
            indices,
            jnp.any(
                (indices < 0) | (indices >= len(self.population_model.condition_ids))
            ),
            "Prediction condition index is outside the declared population support.",
        )
        offset = eqx.error_if(
            offset,
            jnp.any(~jnp.isfinite(offset)),
            "Prediction observation offsets must be finite.",
        )
        populations = self.condition_populations(parameters)[indices]
        state_probability = self.state_mutation_probabilities(
            parameters, observation_offset=offset
        )
        mean = ein.contract("nk,nks->ns", populations, state_probability)
        second = ein.contract(
            "nk,nks,nkt->nst", populations, state_probability, state_probability
        )
        covariance = second - mean[:, :, None] * mean[:, None, :]
        conditional = ein.contract(
            "nk,nks->ns", populations, state_probability * (1.0 - state_probability)
        )
        diagonal = jnp.arange(self.batch.nucleotide_count)
        covariance = covariance.at[:, diagonal, diagonal].add(conditional)
        return EnsemblePosteriorPrediction(
            indices.astype(jnp.int32),
            populations,
            state_probability,
            mean,
            covariance,
            self.model_id,
            self.independence_assumption,
        )

    def posterior_problem(
        self,
        initial_response_logits=None,
        initial_population=None,
        *,
        fit_profile_mask: ArrayLike | None = None,
    ) -> PosteriorProblem:
        selected = self.batch.analysis_profile_mask(fit_profile_mask)
        if not bool(jnp.any(selected)):
            raise ValueError(
                "Finite-ensemble inference requires observed profiles in the fit split."
            )
        space = self._parameter_space(initial_response_logits, initial_population)
        return PosteriorProblem(
            space,
            lambda parameters: jnp.sum(
                jnp.where(selected, self.per_profile_log_likelihood(parameters), 0.0)
            ),
            predict=lambda parameters: (
                self.posterior_prediction(parameters).mutation_probability
            ),
        )

    def diagnostics(
        self,
        parameters,
        policy: EnsembleDiagnosticPolicy,
        /,
        *,
        profile_mask: ArrayLike | None = None,
        optimization_converged: bool = True,
    ) -> EnsembleDiagnostics:
        if not isinstance(policy, EnsembleDiagnosticPolicy):
            raise TypeError("policy must be an EnsembleDiagnosticPolicy.")
        selected_profiles = self.batch.analysis_profile_mask(profile_mask)
        prediction = self.posterior_prediction(parameters)
        state_probability = prediction.state_mutation_probability
        difference = state_probability[:, :, None, :] - state_probability[:, None, :, :]
        separation = jnp.sqrt(jnp.mean(difference**2, axis=(0, 3)))
        off_diagonal = ~jnp.eye(len(self.hypothesis.state_ids), dtype=bool)
        equivalent = off_diagonal & (separation <= policy.minimum_state_separation)
        populations = self.condition_populations(parameters)
        unsupported = jnp.max(populations, axis=0) < policy.minimum_state_population

        problem = self.posterior_problem(fit_profile_mask=selected_profiles)
        position = problem.parameter_space.unconstrain(parameters)
        flat, unravel = ravel_pytree(position)
        selection = np.asarray(
            self.batch.observed_mask & selected_profiles[:, None]
        ).reshape(-1)
        if np.any(selection):
            jacobian = jax.jacrev(
                lambda value: problem.predict(unravel(value)).reshape(-1)[selection]
            )(flat)
            _, singular, directions = jnp.linalg.svd(jacobian, full_matrices=False)
            tolerance = policy.singular_relative_tolerance * singular[0]
            rank = jnp.sum(singular > tolerance, dtype=jnp.int32)
        else:
            singular = jnp.zeros((0,), dtype=flat.dtype)
            directions = jnp.zeros((0, flat.size), dtype=flat.dtype)
            rank = jnp.asarray(0, dtype=jnp.int32)
        residual_correlation = _residual_correlation_excess(
            self.batch.mutation.astype(prediction.mutation_probability.dtype),
            prediction,
            self.batch.observed_mask & selected_profiles[:, None],
        )
        site_off_diagonal = ~jnp.eye(self.batch.nucleotide_count, dtype=bool)
        available = site_off_diagonal & jnp.isfinite(residual_correlation)
        maximum = jnp.where(
            jnp.any(available),
            jnp.max(jnp.where(available, jnp.abs(residual_correlation), -jnp.inf)),
            jnp.nan,
        )
        residual_supported = jnp.isfinite(maximum) & (
            maximum <= policy.maximum_residual_correlation
        )
        optimization_valid = jnp.asarray(bool(optimization_converged))
        numerical = (
            optimization_valid
            & jnp.all(jnp.isfinite(prediction.mutation_probability))
            & jnp.all(jnp.isfinite(populations))
            & jnp.all(jnp.isfinite(singular))
        )
        complete = numerical & ~jnp.any(unsupported)
        unique = (
            complete & residual_supported & ~jnp.any(equivalent) & (rank == flat.size)
        )
        diagnostics_id = canonical_fingerprint(
            {
                "kind": "finite-ensemble-diagnostics",
                "model": self.model_id,
                "profile_mask": array_tree_fingerprint(selected_profiles),
                "policy": {
                    "minimum_state_separation": policy.minimum_state_separation,
                    "minimum_state_population": policy.minimum_state_population,
                    "singular_relative_tolerance": policy.singular_relative_tolerance,
                    "maximum_residual_correlation": policy.maximum_residual_correlation,
                },
                "values": array_tree_fingerprint(
                    (
                        separation,
                        equivalent,
                        unsupported,
                        singular,
                        directions,
                        rank,
                        residual_correlation,
                        maximum,
                        residual_supported,
                        optimization_valid,
                        numerical,
                        complete,
                        unique,
                    )
                ),
            }
        )
        return EnsembleDiagnostics(
            separation,
            equivalent,
            unsupported,
            singular,
            directions,
            rank,
            int(flat.size),
            residual_correlation,
            selected_profiles,
            maximum,
            residual_supported,
            optimization_valid,
            numerical,
            complete,
            unique,
            policy,
            self.model_id,
            diagnostics_id,
        )

    def permutation_invariant_summary(
        self, parameters, /
    ) -> PermutationInvariantEnsembleSummary:
        prediction = self.posterior_prediction(parameters)
        state_mean = jnp.mean(prediction.state_mutation_probability, axis=(0, 2))
        difference = (
            prediction.state_mutation_probability[:, :, None, :]
            - prediction.state_mutation_probability[:, None, :, :]
        )
        separation = jnp.sqrt(jnp.mean(difference**2, axis=(0, 3)))
        triangle = np.triu_indices(len(self.hypothesis.state_ids), 1)
        values = jnp.sort(separation[triangle])
        populations = jnp.sort(self.condition_populations(parameters), axis=-1)
        response = jnp.sort(state_mean)
        summary_id = canonical_fingerprint(
            {
                "kind": "permutation-invariant-ensemble-summary",
                "model": self.model_id,
                "convention": "sorted populations, sorted state means, sorted pair separations",
            }
        )
        return PermutationInvariantEnsembleSummary(
            populations,
            response,
            values,
            prediction.mutation_probability,
            summary_id,
        )

    def fit(
        self,
        /,
        *,
        policy: EnsembleDiagnosticPolicy,
        initial_response_logits=None,
        initial_population=None,
        fit_profile_mask: ArrayLike | None = None,
        requested_use=None,
        max_steps: int = 500,
        gradient_tolerance: float = 1e-6,
    ) -> FiniteEnsembleFit:
        use = {} if requested_use is None else dict(requested_use)
        use["training_use"] = True
        selected = self.batch.analysis_profile_mask(fit_profile_mask)
        if not bool(jnp.any(selected)):
            raise ValueError(
                "Finite-ensemble inference requires observed profiles in the fit split."
            )
        self.batch.require_rights(use, profile_mask=selected)
        problem = self.posterior_problem(
            initial_response_logits,
            initial_population,
            fit_profile_mask=selected,
        )
        result = find_map(
            problem,
            max_steps=max_steps,
            gradient_tolerance=gradient_tolerance,
            raise_on_failure=False,
        )
        return FiniteEnsembleFit(
            result,
            self.posterior_prediction(result.parameters),
            self.per_profile_log_likelihood(result.parameters),
            self.state_posterior(result.parameters),
            self.diagnostics(
                result.parameters,
                policy,
                profile_mask=selected,
                optimization_converged=result.converged,
            ),
            self.permutation_invariant_summary(result.parameters),
            selected,
            self.model_id,
            self.batch.source_ids,
        )


class EnsembleSupportComparison(StrictModule, NonTrainableState):
    predictive_log_scores: Array
    valid_support: Array
    equivalent_supports: Array
    best_support_index: Array
    unique_best: Array
    support_ids: tuple[str, ...] = eqx.field(static=True)
    model_ids: tuple[str, ...] = eqx.field(static=True)
    hypothesis_ids: tuple[str, ...] = eqx.field(static=True)
    derivation_ids: tuple[str, ...] = eqx.field(static=True)
    score_ids: tuple[str, ...] = eqx.field(static=True)
    campaign_id: str = eqx.field(static=True)
    equivalence_tolerance: float = eqx.field(static=True)
    comparison_id: str = eqx.field(static=True)


def compare_ensemble_supports(
    support_ids: tuple[str, ...],
    predictive_log_scores: ArrayLike,
    /,
    *,
    campaign_id: str,
    model_ids: tuple[str, ...],
    hypothesis_ids: tuple[str, ...],
    derivation_ids: tuple[str, ...],
    score_ids: tuple[str, ...],
    equivalence_tolerance: float,
    support_valid: ArrayLike | None = None,
) -> EnsembleSupportComparison:
    identifiers = tuple(support_ids)
    models = tuple(model_ids)
    hypotheses = tuple(hypothesis_ids)
    derivations = tuple(derivation_ids)
    score_identifiers = tuple(score_ids)
    campaign = str(campaign_id)
    bound_rows = (models, hypotheses, derivations, score_identifiers)
    if (
        not campaign
        or campaign != campaign.strip()
        or any(len(row) != len(identifiers) for row in bound_rows)
        or any(
            not isinstance(value, str) or not value or value != value.strip()
            for row in bound_rows
            for value in row
        )
    ):
        raise ValueError(
            "Support comparison requires campaign, model, hypothesis, derivation, "
            "and grouped-score identity for every support."
        )
    scores = jnp.asarray(predictive_log_scores)
    tolerance = float(equivalence_tolerance)
    if (
        len(identifiers) < 2
        or len(set(identifiers)) != len(identifiers)
        or any(not value for value in identifiers)
    ):
        raise ValueError("Support comparison requires at least two unique support IDs.")
    if scores.shape != (len(identifiers),) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError(
            "Support scores and equivalence tolerance have incompatible domains."
        )
    finite = jnp.isfinite(scores)
    if support_valid is None:
        declared_valid = jnp.ones((len(identifiers),), dtype=bool)
    else:
        declared_valid = jnp.asarray(support_valid)
        if declared_valid.shape != (len(identifiers),) or declared_valid.dtype != bool:
            raise TypeError("support_valid must contain one boolean per support.")
    valid = finite & declared_valid
    safe = jnp.where(valid, scores, -jnp.inf)
    best = jnp.argmax(safe)
    best_valid = jnp.any(valid)
    equivalent = valid & best_valid & ((safe[best] - safe) <= tolerance)
    unique = best_valid & (jnp.sum(equivalent.astype(jnp.int32)) == 1)
    comparison_id = canonical_fingerprint(
        {
            "kind": "ensemble-support-comparison",
            "supports": identifiers,
            "campaign": campaign,
            "models": models,
            "hypotheses": hypotheses,
            "derivations": derivations,
            "score_ids": score_identifiers,
            "tolerance": tolerance,
            "score_semantics": "caller-supplied grouped predictive log score",
            "scores": array_tree_fingerprint(np.asarray(scores)),
            "valid": array_tree_fingerprint(np.asarray(valid)),
        }
    )
    return EnsembleSupportComparison(
        scores,
        valid,
        equivalent,
        jnp.where(unique, best, -1).astype(jnp.int32),
        unique,
        identifiers,
        models,
        hypotheses,
        derivations,
        score_identifiers,
        campaign,
        tolerance,
        comparison_id,
    )


def _residual_correlation_excess(mutation, prediction, mask):
    residual = mutation - prediction.mutation_probability
    pair_mask = mask[:, :, None] & mask[:, None, :]
    count = jnp.sum(pair_mask, axis=0)
    empirical = jnp.sum(residual[:, :, None] * residual[:, None, :] * pair_mask, axis=0)
    empirical = jnp.where(count > 0, empirical / count, jnp.nan)
    model = jnp.sum(jnp.where(pair_mask, prediction.mutation_covariance, 0.0), axis=0)
    model = jnp.where(count > 0, model / count, jnp.nan)
    variance = jnp.diag(empirical)
    denominator = jnp.sqrt(variance[:, None] * variance[None, :])
    return jnp.where(
        (count >= 2) & (denominator > 0), (empirical - model) / denominator, jnp.nan
    )


__all__ = [
    "ConditionPopulationModel",
    "EnsembleDiagnosticPolicy",
    "EnsembleDiagnostics",
    "EnsemblePosteriorPrediction",
    "EnsembleSupportComparison",
    "FiniteEnsembleFit",
    "FiniteStructuralEnsembleModel",
    "PermutationInvariantEnsembleSummary",
    "PopulationModelKind",
    "StructuralEnsembleHypothesis",
    "compare_ensemble_supports",
]
