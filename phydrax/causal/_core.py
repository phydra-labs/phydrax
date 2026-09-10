#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import enum
from collections.abc import Iterable, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class VariableObservability(enum.StrEnum):
    OBSERVED = "observed"
    LATENT = "latent"
    SELECTION = "selection"
    CONTEXT = "context"


class VariableScale(enum.StrEnum):
    CONTINUOUS = "continuous"
    BINARY = "binary"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    INTEGER = "integer"
    COUNT = "count"


class AssumptionKind(enum.StrEnum):
    CAUSAL_MARKOV = "causal_markov"
    FAITHFULNESS = "faithfulness"
    CAUSAL_SUFFICIENCY = "causal_sufficiency"
    CONSISTENCY = "consistency"
    POSITIVITY = "positivity"
    CONDITIONAL_EXCHANGEABILITY = "conditional_exchangeability"
    NO_INTERFERENCE = "no_interference"
    RANDOMIZATION = "randomization"
    MECHANISM_INVARIANCE = "mechanism_invariance"
    MCAR = "mcar"
    SAMPLING_IGNORABILITY = "sampling_ignorability"


class AssumptionDisposition(enum.StrEnum):
    ESTABLISHED_BY_DESIGN = "established_by_design"
    DECLARED = "declared"
    NOT_REJECTED = "not_rejected"
    VIOLATED = "violated"
    UNSUPPORTED = "unsupported"
    NOT_APPLICABLE = "not_applicable"


class AssignmentKind(enum.StrEnum):
    OBSERVATIONAL = "observational"
    RANDOMIZED = "randomized"
    ENCOURAGEMENT = "encouragement"


class TargetPopulationKind(enum.StrEnum):
    SAMPLE = "sample"
    SUBGROUP = "subgroup"


class RecoveryStatus(enum.IntEnum):
    SUCCESS = 0
    MISSING_REQUIRED_VARIABLE = 1
    MISSING_REQUIRED_VALUE = 2
    INTERFERENCE_UNSUPPORTED = 3
    TARGET_LAW_UNAVAILABLE = 4
    ASSUMPTION_UNSUPPORTED = 5


class CausalVariable(StrictModule, NonTrainableState):
    """Intrinsic schema for one scalar or array-valued causal variable."""

    name: str = eqx.field(static=True)
    observability: VariableObservability = eqx.field(static=True)
    scale: VariableScale = eqx.field(static=True)
    event_shape: tuple[int, ...] = eqx.field(static=True)
    cardinality: int | None = eqx.field(static=True)
    quantity_id: str | None = eqx.field(static=True)
    temporal_tier: int | None = eqx.field(static=True)
    variable_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        name: str,
        observability: VariableObservability | str = VariableObservability.OBSERVED,
        scale: VariableScale | str = VariableScale.CONTINUOUS,
        event_shape: Sequence[int] = (),
        cardinality: int | None = None,
        quantity_id: str | None = None,
        temporal_tier: int | None = None,
    ) -> None:
        canonical_name = str(name).strip()
        if not canonical_name:
            raise ValueError("CausalVariable name must be non-empty.")
        canonical_observability = VariableObservability(observability)
        canonical_scale = VariableScale(scale)
        canonical_shape = tuple(int(size) for size in event_shape)
        if any(size < 1 for size in canonical_shape):
            raise ValueError("CausalVariable event_shape entries must be positive.")
        canonical_cardinality = None if cardinality is None else int(cardinality)
        finite_scale = canonical_scale in {
            VariableScale.BINARY,
            VariableScale.CATEGORICAL,
            VariableScale.ORDINAL,
        }
        if canonical_scale is VariableScale.BINARY:
            if canonical_cardinality not in (None, 2):
                raise ValueError("Binary variables have cardinality two.")
            canonical_cardinality = 2
        elif finite_scale:
            if canonical_cardinality is None or canonical_cardinality < 2:
                raise ValueError("Finite categorical variables need cardinality >= 2.")
        elif canonical_cardinality is not None:
            raise ValueError("cardinality is only valid for finite categorical scales.")
        if temporal_tier is not None and int(temporal_tier) < 0:
            raise ValueError("temporal_tier must be non-negative when supplied.")
        payload = {
            "name": canonical_name,
            "observability": canonical_observability.value,
            "scale": canonical_scale.value,
            "event_shape": canonical_shape,
            "cardinality": canonical_cardinality,
            "quantity_id": quantity_id,
            "temporal_tier": temporal_tier,
        }
        object.__setattr__(self, "name", canonical_name)
        object.__setattr__(self, "observability", canonical_observability)
        object.__setattr__(self, "scale", canonical_scale)
        object.__setattr__(self, "event_shape", canonical_shape)
        object.__setattr__(self, "cardinality", canonical_cardinality)
        object.__setattr__(self, "quantity_id", quantity_id)
        object.__setattr__(self, "temporal_tier", temporal_tier)
        object.__setattr__(self, "variable_id", canonical_fingerprint(payload))


class CausalSchema(StrictModule, NonTrainableState):
    """Canonical ordered causal-variable universe."""

    variables: tuple[CausalVariable, ...] = eqx.field(static=True)
    names: tuple[str, ...] = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)

    def __init__(self, variables: Sequence[CausalVariable]) -> None:
        canonical = tuple(variables)
        if not canonical:
            raise ValueError("CausalSchema requires at least one variable.")
        names = tuple(variable.name for variable in canonical)
        if len(set(names)) != len(names):
            raise ValueError("CausalSchema variable names must be unique.")
        object.__setattr__(self, "variables", canonical)
        object.__setattr__(self, "names", names)
        object.__setattr__(
            self,
            "schema_id",
            canonical_fingerprint({"variables": [v.variable_id for v in canonical]}),
        )

    def index(self, name: str, /) -> int:
        if name not in self.names:
            raise KeyError(f"Unknown causal variable {name!r}.")
        return self.names.index(name)

    def variable(self, name: str, /) -> CausalVariable:
        return self.variables[self.index(name)]


class CausalDataset(StrictModule, NonTrainableState):
    """Role-free aligned observations over one causal schema."""

    schema: CausalSchema = eqx.field(static=True)
    values: tuple[Array, ...]
    observed: tuple[Array, ...]
    sample_mask: Array
    sampling_weight: Array
    measure_weight: Array
    cluster_id: Array
    n_samples: int = eqx.field(static=True)
    data_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        schema: CausalSchema,
        values: Sequence[Array],
        observed: Sequence[Array] | None = None,
        sample_mask: Array | None = None,
        sampling_weight: Array | None = None,
        measure_weight: Array | None = None,
        cluster_id: Array | None = None,
    ) -> None:
        canonical_values = tuple(jnp.asarray(value) for value in values)
        if len(canonical_values) != len(schema.variables):
            raise ValueError("values must contain one array per schema variable.")
        if not canonical_values or canonical_values[0].ndim < 1:
            raise ValueError("CausalDataset values need a leading sample axis.")
        n_samples = int(canonical_values[0].shape[0])
        if n_samples < 1:
            raise ValueError("CausalDataset requires at least one sample.")
        for variable, value in zip(schema.variables, canonical_values, strict=True):
            if value.shape != (n_samples,) + variable.event_shape:
                raise ValueError(
                    f"Variable {variable.name!r} has shape {value.shape}; expected "
                    f"{(n_samples,) + variable.event_shape}."
                )
            host = np.asarray(value)
            if variable.scale in {
                VariableScale.BINARY,
                VariableScale.CATEGORICAL,
                VariableScale.ORDINAL,
            }:
                if not np.issubdtype(host.dtype, np.integer):
                    raise TypeError(
                        f"Variable {variable.name!r} must use an integer dtype."
                    )
                cardinality = variable.cardinality
                if cardinality is None:
                    raise ValueError("Finite variables require a declared cardinality.")
                if np.any(host < 0) or np.any(host >= cardinality):
                    raise ValueError(
                        f"Variable {variable.name!r} leaves its finite support."
                    )
            elif not np.all(np.isfinite(host)):
                raise ValueError(
                    f"Variable {variable.name!r} contains non-finite values."
                )
        if observed is None:
            canonical_observed = tuple(
                jnp.ones((n_samples,), dtype=bool) for _ in canonical_values
            )
        else:
            canonical_observed = tuple(jnp.asarray(mask, dtype=bool) for mask in observed)
            if len(canonical_observed) != len(canonical_values):
                raise ValueError("observed must contain one mask per schema variable.")
            if any(mask.shape != (n_samples,) for mask in canonical_observed):
                raise ValueError("Every observed mask must have shape (n_samples,).")
        canonical_sample_mask = (
            jnp.ones((n_samples,), dtype=bool)
            if sample_mask is None
            else jnp.asarray(sample_mask, dtype=bool)
        )
        canonical_sampling_weight = (
            jnp.ones((n_samples,), dtype=jnp.float64)
            if sampling_weight is None
            else jnp.asarray(sampling_weight)
        )
        canonical_measure_weight = (
            jnp.ones((n_samples,), dtype=jnp.float64)
            if measure_weight is None
            else jnp.asarray(measure_weight)
        )
        canonical_cluster_id = (
            jnp.arange(n_samples, dtype=jnp.int32)
            if cluster_id is None
            else jnp.asarray(cluster_id)
        )
        for name, array in (
            ("sample_mask", canonical_sample_mask),
            ("sampling_weight", canonical_sampling_weight),
            ("measure_weight", canonical_measure_weight),
            ("cluster_id", canonical_cluster_id),
        ):
            if array.shape != (n_samples,):
                raise ValueError(f"{name} must have shape (n_samples,).")
        for name, weight in (
            ("sampling_weight", canonical_sampling_weight),
            ("measure_weight", canonical_measure_weight),
        ):
            host = np.asarray(weight)
            if not np.all(np.isfinite(host)) or np.any(host < 0):
                raise ValueError(f"{name} must be finite and non-negative.")
            if float(np.sum(host[np.asarray(canonical_sample_mask)])) <= 0:
                raise ValueError(f"{name} must have positive active mass.")
        payload = {
            "schema_id": schema.schema_id,
            "values": canonical_values,
            "observed": canonical_observed,
            "sample_mask": canonical_sample_mask,
            "sampling_weight": canonical_sampling_weight,
            "measure_weight": canonical_measure_weight,
            "cluster_id": canonical_cluster_id,
        }
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "values", canonical_values)
        object.__setattr__(self, "observed", canonical_observed)
        object.__setattr__(self, "sample_mask", canonical_sample_mask)
        object.__setattr__(self, "sampling_weight", canonical_sampling_weight)
        object.__setattr__(self, "measure_weight", canonical_measure_weight)
        object.__setattr__(self, "cluster_id", canonical_cluster_id)
        object.__setattr__(self, "n_samples", n_samples)
        object.__setattr__(self, "data_id", canonical_fingerprint(payload))

    def value(self, name: str, /) -> Array:
        return self.values[self.schema.index(name)]

    def observed_mask(self, name: str, /) -> Array:
        return self.observed[self.schema.index(name)]


class CausalDataSplit(StrictModule, NonTrainableState):
    """Disjoint sample ownership for structure, estimation, calibration, evaluation."""

    structure: Array
    estimation: Array
    calibration: Array
    evaluation: Array
    n_samples: int = eqx.field(static=True)
    split_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        n_samples: int,
        structure: Array | Sequence[int] = (),
        estimation: Array | Sequence[int] = (),
        calibration: Array | Sequence[int] = (),
        evaluation: Array | Sequence[int] = (),
    ) -> None:
        size = int(n_samples)
        if size < 1:
            raise ValueError("n_samples must be positive.")
        arrays = tuple(
            jnp.asarray(value, dtype=jnp.int32)
            for value in (structure, estimation, calibration, evaluation)
        )
        host_arrays = tuple(np.asarray(value) for value in arrays)
        for value in host_arrays:
            if value.ndim != 1 or np.any(value < 0) or np.any(value >= size):
                raise ValueError("Split indices must be one-dimensional and in range.")
            if np.unique(value).size != value.size:
                raise ValueError("Each split role must contain unique sample indices.")
        combined = (
            np.concatenate(host_arrays) if host_arrays else np.empty((0,), dtype=int)
        )
        if np.unique(combined).size != combined.size:
            raise ValueError("Causal data split roles must be disjoint.")
        object.__setattr__(self, "structure", arrays[0])
        object.__setattr__(self, "estimation", arrays[1])
        object.__setattr__(self, "calibration", arrays[2])
        object.__setattr__(self, "evaluation", arrays[3])
        object.__setattr__(self, "n_samples", size)
        object.__setattr__(
            self,
            "split_id",
            canonical_fingerprint(
                {
                    "n_samples": size,
                    "structure": arrays[0],
                    "estimation": arrays[1],
                    "calibration": arrays[2],
                    "evaluation": arrays[3],
                }
            ),
        )


class CausalAssumption(StrictModule, NonTrainableState):
    kind: AssumptionKind = eqx.field(static=True)
    disposition: AssumptionDisposition = eqx.field(static=True)
    subject_ids: tuple[str, ...] = eqx.field(static=True)
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    statement: str = eqx.field(static=True)
    assumption_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        kind: AssumptionKind | str,
        disposition: AssumptionDisposition | str,
        statement: str,
        subject_ids: Sequence[str] = (),
        evidence_ids: Sequence[str] = (),
    ) -> None:
        canonical_kind = AssumptionKind(kind)
        canonical_disposition = AssumptionDisposition(disposition)
        canonical_statement = str(statement).strip()
        if not canonical_statement:
            raise ValueError("Causal assumption statement must be non-empty.")
        canonical_subjects = tuple(sorted(set(str(item) for item in subject_ids)))
        canonical_evidence = tuple(sorted(set(str(item) for item in evidence_ids)))
        payload = {
            "kind": canonical_kind.value,
            "disposition": canonical_disposition.value,
            "statement": canonical_statement,
            "subject_ids": canonical_subjects,
            "evidence_ids": canonical_evidence,
        }
        object.__setattr__(self, "kind", canonical_kind)
        object.__setattr__(self, "disposition", canonical_disposition)
        object.__setattr__(self, "statement", canonical_statement)
        object.__setattr__(self, "subject_ids", canonical_subjects)
        object.__setattr__(self, "evidence_ids", canonical_evidence)
        object.__setattr__(self, "assumption_id", canonical_fingerprint(payload))


class AssumptionLedger(StrictModule, NonTrainableState):
    assumptions: tuple[CausalAssumption, ...] = eqx.field(static=True)
    ledger_id: str = eqx.field(static=True)

    def __init__(self, assumptions: Iterable[CausalAssumption]) -> None:
        canonical = tuple(sorted(assumptions, key=lambda item: item.assumption_id))
        if len({item.assumption_id for item in canonical}) != len(canonical):
            raise ValueError("AssumptionLedger contains duplicate assumptions.")
        object.__setattr__(self, "assumptions", canonical)
        object.__setattr__(
            self,
            "ledger_id",
            canonical_fingerprint({"assumptions": [a.assumption_id for a in canonical]}),
        )

    def disposition(self, kind: AssumptionKind, /) -> AssumptionDisposition | None:
        matches = [item.disposition for item in self.assumptions if item.kind is kind]
        if not matches:
            return None
        if len(set(matches)) != 1:
            return AssumptionDisposition.UNSUPPORTED
        return matches[0]


class CausalStudyDesign(StrictModule, NonTrainableState):
    schema_id: str = eqx.field(static=True)
    assignment_kind: AssignmentKind = eqx.field(static=True)
    assignment_variable: str = eqx.field(static=True)
    exposure_variable: str = eqx.field(static=True)
    source_population_id: str = eqx.field(static=True)
    no_interference: bool = eqx.field(static=True)
    assumptions: AssumptionLedger = eqx.field(static=True)
    known_assignment_probability: Array | None
    design_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        schema: CausalSchema,
        assignment_kind: AssignmentKind | str,
        assignment_variable: str,
        exposure_variable: str,
        source_population_id: str,
        assumptions: AssumptionLedger,
        no_interference: bool,
        known_assignment_probability: Array | None = None,
    ) -> None:
        canonical_kind = AssignmentKind(assignment_kind)
        schema.index(assignment_variable)
        schema.index(exposure_variable)
        population_id = str(source_population_id).strip()
        if not population_id:
            raise ValueError("source_population_id must be non-empty.")
        probability = (
            None
            if known_assignment_probability is None
            else jnp.asarray(known_assignment_probability)
        )
        if probability is not None:
            host = np.asarray(probability)
            if host.ndim not in (1, 2):
                raise ValueError(
                    "known_assignment_probability must have one or two axes."
                )
            if not np.all(np.isfinite(host)) or np.any(host <= 0) or np.any(host > 1):
                raise ValueError("Known assignment probabilities must lie in (0, 1].")
            if host.ndim == 2 and not np.allclose(host.sum(axis=-1), 1.0):
                raise ValueError("Multi-arm assignment probabilities must sum to one.")
        payload = {
            "schema_id": schema.schema_id,
            "assignment_kind": canonical_kind.value,
            "assignment_variable": assignment_variable,
            "exposure_variable": exposure_variable,
            "source_population_id": population_id,
            "no_interference": bool(no_interference),
            "assumptions_id": assumptions.ledger_id,
            "known_assignment_probability": probability,
        }
        object.__setattr__(self, "schema_id", schema.schema_id)
        object.__setattr__(self, "assignment_kind", canonical_kind)
        object.__setattr__(self, "assignment_variable", assignment_variable)
        object.__setattr__(self, "exposure_variable", exposure_variable)
        object.__setattr__(self, "source_population_id", population_id)
        object.__setattr__(self, "no_interference", bool(no_interference))
        object.__setattr__(self, "assumptions", assumptions)
        object.__setattr__(self, "known_assignment_probability", probability)
        object.__setattr__(self, "design_id", canonical_fingerprint(payload))


class TargetPopulation(StrictModule, NonTrainableState):
    kind: TargetPopulationKind = eqx.field(static=True)
    population_id: str = eqx.field(static=True)
    source_population_id: str = eqx.field(static=True)
    eligibility_basis_id: str | None = eqx.field(static=True)
    eligibility: Array | None

    def __init__(
        self,
        *,
        source_population_id: str,
        kind: TargetPopulationKind | str = TargetPopulationKind.SAMPLE,
        eligibility: Array | None = None,
        eligibility_basis_id: str | None = None,
    ) -> None:
        canonical_kind = TargetPopulationKind(kind)
        population = str(source_population_id).strip()
        if not population:
            raise ValueError("source_population_id must be non-empty.")
        canonical_eligibility = (
            None if eligibility is None else jnp.asarray(eligibility, dtype=bool)
        )
        basis = (
            None if eligibility_basis_id is None else str(eligibility_basis_id).strip()
        )
        if canonical_kind is TargetPopulationKind.SUBGROUP:
            if canonical_eligibility is None or canonical_eligibility.ndim != 1:
                raise ValueError(
                    "Subgroup populations require a one-dimensional eligibility mask."
                )
            if not bool(np.any(np.asarray(canonical_eligibility))):
                raise ValueError("Subgroup eligibility must contain at least one member.")
            if not basis:
                raise ValueError(
                    "Subgroup populations require a predeclared eligibility_basis_id."
                )
        elif canonical_eligibility is not None or basis is not None:
            raise ValueError(
                "Sample target populations do not accept eligibility metadata."
            )
        payload = {
            "kind": canonical_kind.value,
            "source_population_id": population,
            "eligibility": canonical_eligibility,
            "eligibility_basis_id": basis,
        }
        object.__setattr__(self, "kind", canonical_kind)
        object.__setattr__(self, "source_population_id", population)
        object.__setattr__(self, "eligibility_basis_id", basis)
        object.__setattr__(self, "eligibility", canonical_eligibility)
        object.__setattr__(self, "population_id", canonical_fingerprint(payload))


class TreatmentRegime(StrictModule, NonTrainableState):
    exposure_variable: str = eqx.field(static=True)
    value: int | float = eqx.field(static=True)
    regime_id: str = eqx.field(static=True)

    def __init__(self, *, exposure_variable: str, value: int | float) -> None:
        canonical_variable = str(exposure_variable).strip()
        if not canonical_variable:
            raise ValueError("exposure_variable must be non-empty.")
        canonical_value = (
            int(value) if isinstance(value, (int, np.integer)) else float(value)
        )
        if not np.isfinite(canonical_value):
            raise ValueError("Treatment regime value must be finite.")
        object.__setattr__(self, "exposure_variable", canonical_variable)
        object.__setattr__(self, "value", canonical_value)
        object.__setattr__(
            self,
            "regime_id",
            canonical_fingerprint(
                {
                    "exposure_variable": canonical_variable,
                    "value": canonical_value,
                }
            ),
        )


class TreatmentContrast(StrictModule, NonTrainableState):
    active: TreatmentRegime = eqx.field(static=True)
    reference: TreatmentRegime = eqx.field(static=True)
    contrast_id: str = eqx.field(static=True)

    def __init__(self, *, active: TreatmentRegime, reference: TreatmentRegime) -> None:
        if active.exposure_variable != reference.exposure_variable:
            raise ValueError("Treatment contrast regimes must target the same variable.")
        if active.value == reference.value:
            raise ValueError("Treatment contrast regimes must have distinct values.")
        object.__setattr__(self, "active", active)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(
            self,
            "contrast_id",
            canonical_fingerprint(
                {
                    "active_id": active.regime_id,
                    "reference_id": reference.regime_id,
                }
            ),
        )


class CausalQuery(StrictModule, NonTrainableState):
    schema_id: str = eqx.field(static=True)
    outcome_variable: str = eqx.field(static=True)
    contrast: TreatmentContrast = eqx.field(static=True)
    population: TargetPopulation
    query_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        schema: CausalSchema,
        outcome_variable: str,
        contrast: TreatmentContrast,
        population: TargetPopulation,
    ) -> None:
        outcome = schema.variable(outcome_variable)
        exposure = schema.variable(contrast.active.exposure_variable)
        if outcome.observability is not VariableObservability.OBSERVED:
            raise ValueError("The observed-data query outcome must be observed.")
        if exposure.observability is not VariableObservability.OBSERVED:
            raise ValueError("The observed-data query exposure must be observed.")
        if outcome.event_shape:
            raise ValueError(
                "Observed-data mean-effect estimators require scalar outcomes."
            )
        if exposure.event_shape:
            raise ValueError(
                "Observed-data treatment contrasts require scalar exposures."
            )
        if exposure.cardinality is not None:
            values = (contrast.active.value, contrast.reference.value)
            if any(
                not isinstance(value, int) or value < 0 or value >= exposure.cardinality
                for value in values
            ):
                raise ValueError("Treatment contrast leaves the exposure support.")
        object.__setattr__(self, "schema_id", schema.schema_id)
        object.__setattr__(self, "outcome_variable", outcome_variable)
        object.__setattr__(self, "contrast", contrast)
        object.__setattr__(self, "population", population)
        object.__setattr__(
            self,
            "query_id",
            canonical_fingerprint(
                {
                    "schema_id": schema.schema_id,
                    "outcome_variable": outcome_variable,
                    "contrast_id": contrast.contrast_id,
                    "population_id": population.population_id,
                    "functional": "mean_difference",
                }
            ),
        )


class AvailableLaw(StrictModule, NonTrainableState):
    status: RecoveryStatus = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)
    design_id: str = eqx.field(static=True)
    query_id: str = eqx.field(static=True)
    data_id: str = eqx.field(static=True)
    basis: str = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    @property
    def successful(self) -> bool:
        return self.status is RecoveryStatus.SUCCESS


class CausalProblem(StrictModule, NonTrainableState):
    dataset: CausalDataset
    design: CausalStudyDesign
    query: CausalQuery
    available_law: AvailableLaw = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        dataset: CausalDataset,
        design: CausalStudyDesign,
        query: CausalQuery,
    ) -> None:
        law = recover_available_law(dataset=dataset, design=design, query=query)
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "design", design)
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "available_law", law)
        object.__setattr__(
            self,
            "problem_id",
            canonical_fingerprint(
                {
                    "data_id": dataset.data_id,
                    "design_id": design.design_id,
                    "query_id": query.query_id,
                    "law_id": law.law_id,
                }
            ),
        )


def recover_available_law(
    *,
    dataset: CausalDataset,
    design: CausalStudyDesign,
    query: CausalQuery,
) -> AvailableLaw:
    """Establish the observed target law available to identification."""
    if (
        design.schema_id != dataset.schema.schema_id
        or query.schema_id != dataset.schema.schema_id
    ):
        raise ValueError("Dataset, design, and query schemas must match exactly.")
    status = RecoveryStatus.SUCCESS
    reason = "Complete observed source law equals the declared target law."
    if not design.no_interference:
        status = RecoveryStatus.INTERFERENCE_UNSUPPORTED
        reason = (
            "Current observed-data estimators require an explicit no-interference design."
        )
    elif query.population.source_population_id != design.source_population_id:
        status = RecoveryStatus.TARGET_LAW_UNAVAILABLE
        reason = (
            "External-population recovery is not available without a supplied target law."
        )
    else:
        required = {
            design.assignment_variable,
            design.exposure_variable,
            query.outcome_variable,
        }
        for name in required:
            variable = dataset.schema.variable(name)
            if variable.observability is not VariableObservability.OBSERVED:
                status = RecoveryStatus.MISSING_REQUIRED_VARIABLE
                reason = f"Required variable {name!r} is not observed."
                break
            mask = np.asarray(dataset.observed_mask(name) & dataset.sample_mask)
            if not np.all(mask[np.asarray(dataset.sample_mask)]):
                status = RecoveryStatus.MISSING_REQUIRED_VALUE
                reason = f"Required variable {name!r} has missing active observations."
                break
    basis = (
        "randomized_regime"
        if status is RecoveryStatus.SUCCESS
        and design.assignment_kind is AssignmentKind.RANDOMIZED
        else "observational_population"
    )
    payload = {
        "status": int(status),
        "schema_id": dataset.schema.schema_id,
        "design_id": design.design_id,
        "query_id": query.query_id,
        "data_id": dataset.data_id,
        "basis": basis,
        "reason": reason,
    }
    return AvailableLaw(
        status=status,
        schema_id=dataset.schema.schema_id,
        design_id=design.design_id,
        query_id=query.query_id,
        data_id=dataset.data_id,
        basis=basis,
        reason=reason,
        law_id=canonical_fingerprint(payload),
    )


__all__ = [
    "AssignmentKind",
    "AssumptionDisposition",
    "AssumptionKind",
    "AssumptionLedger",
    "AvailableLaw",
    "CausalAssumption",
    "CausalDataSplit",
    "CausalDataset",
    "CausalProblem",
    "CausalQuery",
    "CausalSchema",
    "CausalStudyDesign",
    "CausalVariable",
    "RecoveryStatus",
    "TargetPopulation",
    "TargetPopulationKind",
    "TreatmentContrast",
    "TreatmentRegime",
    "VariableObservability",
    "VariableScale",
    "recover_available_law",
]
