#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import enum
import itertools
from collections.abc import Iterable, Sequence
from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._core import (
    AssignmentKind,
    AssumptionDisposition,
    AssumptionKind,
    CausalProblem,
    CausalSchema,
    VariableObservability,
    VariableScale,
)
from ._graph import (
    CausalADMG,
    CausalCPDAG,
    CausalDAG,
    CausalPAG,
    d_separated,
    descendants,
    m_separated,
    SeparationStatus,
)


class IdentificationStatus(enum.StrEnum):
    IDENTIFIED = "identified"
    NOT_IDENTIFIED = "not_identified"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    INVALID_ASSUMPTIONS = "invalid_assumptions"
    LAW_UNAVAILABLE = "law_unavailable"


class IdentificationBasis(enum.StrEnum):
    RANDOMIZED_DESIGN = "randomized_design"
    ADJUSTMENT = "adjustment"
    GENERAL_ID = "general_id"
    EQUIVALENCE_CLASS_ADJUSTMENT = "equivalence_class_adjustment"


class EvaluationStatus(enum.StrEnum):
    SUCCESS = "success"
    UNSUPPORTED = "unsupported"
    UNDEFINED_SUPPORT = "undefined_support"
    NONFINITE = "nonfinite"
    RESOURCE_EXHAUSTED = "resource_exhausted"


class ObservedJointExpression(StrictModule, NonTrainableState):
    variables: tuple[str, ...] = eqx.field(static=True)
    expression_id: str = eqx.field(static=True)

    def __init__(self, variables: Sequence[str]) -> None:
        canonical = tuple(variables)
        if not canonical:
            raise ValueError("Observed joint expressions require variables.")
        object.__setattr__(self, "variables", canonical)
        object.__setattr__(
            self,
            "expression_id",
            canonical_fingerprint({"kind": "observed_joint", "variables": canonical}),
        )


class MarginalizeExpression(StrictModule, NonTrainableState):
    variables: tuple[str, ...] = eqx.field(static=True)
    operand: Expression = eqx.field(static=True)
    expression_id: str = eqx.field(static=True)

    def __init__(self, *, variables: Iterable[str], operand: Expression) -> None:
        canonical = tuple(sorted(set(variables)))
        object.__setattr__(self, "variables", canonical)
        object.__setattr__(self, "operand", operand)
        object.__setattr__(
            self,
            "expression_id",
            canonical_fingerprint(
                {
                    "kind": "marginalize",
                    "variables": canonical,
                    "operand_id": operand.expression_id,
                }
            ),
        )


class ProductExpression(StrictModule, NonTrainableState):
    operands: tuple[Expression, ...] = eqx.field(static=True)
    expression_id: str = eqx.field(static=True)

    def __init__(self, operands: Iterable[Expression]) -> None:
        canonical = tuple(sorted(operands, key=lambda item: item.expression_id))
        if not canonical:
            raise ValueError("Product expressions require operands.")
        object.__setattr__(self, "operands", canonical)
        object.__setattr__(
            self,
            "expression_id",
            canonical_fingerprint(
                {
                    "kind": "product",
                    "operand_ids": [item.expression_id for item in canonical],
                }
            ),
        )


class QFactorExpression(StrictModule, NonTrainableState):
    variables: tuple[str, ...] = eqx.field(static=True)
    order: tuple[str, ...] = eqx.field(static=True)
    operand: Expression = eqx.field(static=True)
    expression_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        variables: Iterable[str],
        order: Sequence[str],
        operand: Expression,
    ) -> None:
        canonical_order = tuple(order)
        selected = set(variables)
        canonical_variables = tuple(node for node in canonical_order if node in selected)
        if not canonical_variables or selected != set(canonical_variables):
            raise ValueError("Q-factor variables must be a non-empty subset of order.")
        object.__setattr__(self, "variables", canonical_variables)
        object.__setattr__(self, "order", canonical_order)
        object.__setattr__(self, "operand", operand)
        object.__setattr__(
            self,
            "expression_id",
            canonical_fingerprint(
                {
                    "kind": "q_factor",
                    "variables": canonical_variables,
                    "order": canonical_order,
                    "operand_id": operand.expression_id,
                }
            ),
        )


class AdjustmentExpression(StrictModule, NonTrainableState):
    treatment: str = eqx.field(static=True)
    outcome: str = eqx.field(static=True)
    adjustment: tuple[str, ...] = eqx.field(static=True)
    expression_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        treatment: str,
        outcome: str,
        adjustment: Iterable[str],
    ) -> None:
        canonical = tuple(sorted(set(adjustment)))
        object.__setattr__(self, "treatment", treatment)
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "adjustment", canonical)
        object.__setattr__(
            self,
            "expression_id",
            canonical_fingerprint(
                {
                    "kind": "adjustment",
                    "treatment": treatment,
                    "outcome": outcome,
                    "adjustment": canonical,
                }
            ),
        )


class ConditionalExpression(StrictModule, NonTrainableState):
    outcomes: tuple[str, ...] = eqx.field(static=True)
    conditioned: tuple[str, ...] = eqx.field(static=True)
    operand: Expression = eqx.field(static=True)
    expression_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        outcomes: Iterable[str],
        conditioned: Iterable[str],
        operand: Expression,
    ) -> None:
        canonical_outcomes = tuple(sorted(set(outcomes)))
        canonical_conditioned = tuple(sorted(set(conditioned)))
        if not canonical_outcomes or not canonical_conditioned:
            raise ValueError("Conditional expressions require outcomes and conditions.")
        if set(canonical_outcomes) & set(canonical_conditioned):
            raise ValueError("Conditional outcomes and conditions must be disjoint.")
        object.__setattr__(self, "outcomes", canonical_outcomes)
        object.__setattr__(self, "conditioned", canonical_conditioned)
        object.__setattr__(self, "operand", operand)
        object.__setattr__(
            self,
            "expression_id",
            canonical_fingerprint(
                {
                    "kind": "conditional",
                    "outcomes": canonical_outcomes,
                    "conditioned": canonical_conditioned,
                    "operand_id": operand.expression_id,
                }
            ),
        )


Expression: TypeAlias = (
    ObservedJointExpression
    | MarginalizeExpression
    | ProductExpression
    | QFactorExpression
    | AdjustmentExpression
    | ConditionalExpression
)


class HedgeWitness(StrictModule, NonTrainableState):
    district: tuple[str, ...] = eqx.field(static=True)
    graph_id: str = eqx.field(static=True)
    witness_id: str = eqx.field(static=True)

    def __init__(self, *, district: Iterable[str], graph_id: str) -> None:
        canonical = tuple(sorted(set(district)))
        object.__setattr__(self, "district", canonical)
        object.__setattr__(self, "graph_id", graph_id)
        object.__setattr__(
            self,
            "witness_id",
            canonical_fingerprint({"district": canonical, "graph_id": graph_id}),
        )


class IdentificationResult(StrictModule, NonTrainableState):
    status: IdentificationStatus = eqx.field(static=True)
    basis: IdentificationBasis | None = eqx.field(static=True)
    expression: Expression | None = eqx.field(static=True)
    adjustment_sets: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    structure_id: str | None = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    query_id: str = eqx.field(static=True)
    assumptions_id: str = eqx.field(static=True)
    witness: HedgeWitness | None = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    @property
    def identified(self) -> bool:
        return self.status is IdentificationStatus.IDENTIFIED


class IdentificationCertificate(StrictModule, NonTrainableState):
    basis: IdentificationBasis = eqx.field(static=True)
    expression: Expression = eqx.field(static=True)
    structure_id: str | None = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)
    design_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    query_id: str = eqx.field(static=True)
    population_id: str = eqx.field(static=True)
    assumptions_id: str = eqx.field(static=True)
    evaluator_capability: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)


class GraphicalIdentificationResult(StrictModule, NonTrainableState):
    status: IdentificationStatus = eqx.field(static=True)
    expression: Expression | None = eqx.field(static=True)
    witness: HedgeWitness | None = eqx.field(static=True)
    graph_id: str = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    @property
    def identified(self) -> bool:
        return self.status is IdentificationStatus.IDENTIFIED


class FiniteDistributionResult(StrictModule, NonTrainableState):
    status: EvaluationStatus = eqx.field(static=True)
    variables: tuple[str, ...] = eqx.field(static=True)
    probabilities: Array
    expression_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    reason: str = eqx.field(static=True)

    @property
    def successful(self) -> bool:
        return self.status is EvaluationStatus.SUCCESS


class FiniteObservedLaw(StrictModule, NonTrainableState):
    schema: CausalSchema = eqx.field(static=True)
    probabilities: Array
    cardinalities: tuple[int, ...] = eqx.field(static=True)
    law_id: str = eqx.field(static=True)

    def __init__(self, *, schema: CausalSchema, probabilities: Array) -> None:
        cardinalities: list[int] = []
        for variable in schema.variables:
            if variable.event_shape or variable.scale not in {
                VariableScale.BINARY,
                VariableScale.CATEGORICAL,
                VariableScale.ORDINAL,
            }:
                raise ValueError("FiniteObservedLaw requires scalar finite variables.")
            cardinality = variable.cardinality
            if cardinality is None:
                raise ValueError("Finite variables require declared cardinalities.")
            cardinalities.append(cardinality)
        canonical = jnp.asarray(probabilities)
        if canonical.shape != tuple(cardinalities):
            raise ValueError(
                f"Finite law shape {canonical.shape} does not match {tuple(cardinalities)}."
            )
        host = np.asarray(canonical)
        if not np.all(np.isfinite(host)) or np.any(host < 0):
            raise ValueError("Finite law probabilities must be finite and non-negative.")
        if not np.isclose(host.sum(), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("Finite law probabilities must sum to one.")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "probabilities", canonical)
        object.__setattr__(self, "cardinalities", tuple(cardinalities))
        object.__setattr__(
            self,
            "law_id",
            canonical_fingerprint(
                {
                    "schema_id": schema.schema_id,
                    "probabilities": canonical,
                }
            ),
        )


class FiniteEffectResult(StrictModule, NonTrainableState):
    status: EvaluationStatus = eqx.field(static=True)
    active_mean: Array
    reference_mean: Array
    effect: Array
    certificate_id: str = eqx.field(static=True)
    finite_law_id: str = eqx.field(static=True)
    reason: str = eqx.field(static=True)

    @property
    def successful(self) -> bool:
        return self.status is EvaluationStatus.SUCCESS


def enumerate_adjustment_sets(
    graph: CausalDAG | CausalADMG,
    *,
    treatment: str,
    outcome: str,
    maximum_sets: int = 4096,
) -> tuple[tuple[str, ...], ...]:
    """Enumerate valid observed backdoor sets in deterministic minimal-first order."""
    graph.schema.index(treatment)
    graph.schema.index(outcome)
    if treatment == outcome:
        raise ValueError("Treatment and outcome must be distinct.")
    forbidden = set(descendants(graph, (treatment,)))
    forbidden.update((treatment, outcome))
    candidates = tuple(
        variable.name
        for variable in graph.schema.variables
        if variable.observability is VariableObservability.OBSERVED
        and variable.name not in forbidden
    )
    backdoor_directed = tuple(
        edge for edge in graph.directed_edges if edge[0] != treatment
    )
    if isinstance(graph, CausalDAG):
        backdoor: CausalDAG | CausalADMG = CausalDAG(
            schema=graph.schema,
            directed_edges=backdoor_directed,
        )
    else:
        backdoor = CausalADMG(
            schema=graph.schema,
            directed_edges=backdoor_directed,
            bidirected_edges=graph.bidirected_edges,
        )
    valid: list[tuple[str, ...]] = []
    for size in range(len(candidates) + 1):
        for candidate in itertools.combinations(candidates, size):
            separated = (
                d_separated(backdoor, (treatment,), (outcome,), candidate)
                if isinstance(backdoor, CausalDAG)
                else m_separated(backdoor, (treatment,), (outcome,), candidate)
            )
            if separated.separated and not any(
                set(existing) <= set(candidate) for existing in valid
            ):
                valid.append(candidate)
                if len(valid) > maximum_sets:
                    raise ValueError("Adjustment-set enumeration exceeded maximum_sets.")
    return tuple(valid)


def identify_causal_effect(
    problem: CausalProblem,
    graph: CausalDAG | CausalADMG | CausalCPDAG | CausalPAG | None,
    *,
    adjustment_set: Iterable[str] | None = None,
    maximum_sets: int = 4096,
    maximum_id_depth: int = 128,
) -> IdentificationResult:
    """Identify one mean treatment contrast from design or graph semantics."""
    if not problem.available_law.successful:
        return _identification_result(
            status=IdentificationStatus.LAW_UNAVAILABLE,
            basis=None,
            expression=None,
            problem=problem,
            structure_id=None if graph is None else graph.graph_id,
            adjustment_sets=(),
            witness=None,
            reason=problem.available_law.reason,
        )
    randomized_exposure = (
        problem.design.assignment_kind is AssignmentKind.RANDOMIZED
        and problem.design.assignment_variable == problem.design.exposure_variable
    )
    assumption_error = _required_assumption_error(
        problem,
        require_graph=not randomized_exposure,
    )
    if assumption_error is not None:
        return _identification_result(
            status=IdentificationStatus.INVALID_ASSUMPTIONS,
            basis=None,
            expression=None,
            problem=problem,
            structure_id=None if graph is None else graph.graph_id,
            adjustment_sets=(),
            witness=None,
            reason=assumption_error,
        )
    treatment = problem.query.contrast.active.exposure_variable
    outcome = problem.query.outcome_variable
    if randomized_exposure:
        expression = AdjustmentExpression(
            treatment=treatment,
            outcome=outcome,
            adjustment=(),
        )
        return _identification_result(
            status=IdentificationStatus.IDENTIFIED,
            basis=IdentificationBasis.RANDOMIZED_DESIGN,
            expression=expression,
            problem=problem,
            structure_id=None if graph is None else graph.graph_id,
            adjustment_sets=((),),
            witness=None,
            reason="Randomized assignment identifies the declared exposure contrast.",
        )
    if graph is None:
        return _identification_result(
            status=IdentificationStatus.NOT_IDENTIFIED,
            basis=None,
            expression=None,
            problem=problem,
            structure_id=None,
            adjustment_sets=(),
            witness=None,
            reason="Observational identification requires a causal structure.",
        )
    if graph.schema.schema_id != problem.dataset.schema.schema_id:
        raise ValueError("Causal structure and problem schemas must match exactly.")
    if isinstance(graph, CausalPAG):
        return _identification_result(
            status=IdentificationStatus.UNSUPPORTED,
            basis=None,
            expression=None,
            problem=problem,
            structure_id=graph.graph_id,
            adjustment_sets=(),
            witness=None,
            reason="PAG effects require a separately proved class-wide adjustment result.",
        )
    if isinstance(graph, CausalCPDAG):
        per_extension = [
            enumerate_adjustment_sets(
                extension,
                treatment=treatment,
                outcome=outcome,
                maximum_sets=maximum_sets,
            )
            for extension in graph.extensions
        ]
        common = set(per_extension[0]) if per_extension else set()
        for sets in per_extension[1:]:
            common.intersection_update(sets)
        candidates = tuple(sorted(common, key=lambda item: (len(item), item)))
        return _select_adjustment_identification(
            problem=problem,
            structure_id=graph.graph_id,
            basis=IdentificationBasis.EQUIVALENCE_CLASS_ADJUSTMENT,
            candidates=candidates,
            requested=adjustment_set,
        )
    candidates = enumerate_adjustment_sets(
        graph,
        treatment=treatment,
        outcome=outcome,
        maximum_sets=maximum_sets,
    )
    selected = _select_adjustment_identification(
        problem=problem,
        structure_id=graph.graph_id,
        basis=IdentificationBasis.ADJUSTMENT,
        candidates=candidates,
        requested=adjustment_set,
    )
    if selected.identified or isinstance(graph, CausalDAG) or adjustment_set is not None:
        return selected
    expression, witness, exhausted = _identify_id(
        graph,
        outcomes=(outcome,),
        interventions=(treatment,),
        maximum_depth=maximum_id_depth,
    )
    if exhausted:
        return _identification_result(
            status=IdentificationStatus.RESOURCE_EXHAUSTED,
            basis=None,
            expression=None,
            problem=problem,
            structure_id=graph.graph_id,
            adjustment_sets=(),
            witness=None,
            reason="General ID recursion exceeded maximum_id_depth.",
        )
    if expression is None:
        return _identification_result(
            status=IdentificationStatus.NOT_IDENTIFIED,
            basis=None,
            expression=None,
            problem=problem,
            structure_id=graph.graph_id,
            adjustment_sets=(),
            witness=witness,
            reason="The ADMG contains a hedge for the requested intervention.",
        )
    return _identification_result(
        status=IdentificationStatus.IDENTIFIED,
        basis=IdentificationBasis.GENERAL_ID,
        expression=expression,
        problem=problem,
        structure_id=graph.graph_id,
        adjustment_sets=(),
        witness=None,
        reason="The general ID algorithm produced an observational functional.",
    )


def identify_conditional_distribution(
    graph: CausalADMG,
    *,
    outcomes: Iterable[str],
    interventions: Iterable[str],
    conditioned: Iterable[str],
    maximum_id_depth: int = 128,
) -> GraphicalIdentificationResult:
    """Run IDC for a conditional interventional distribution on an ADMG."""
    outcome_set = set(outcomes)
    intervention_set = set(interventions)
    conditioned_set = set(conditioned)
    graph_nodes = set(graph.schema.names)
    if not outcome_set or not intervention_set or not conditioned_set:
        raise ValueError(
            "IDC requires non-empty outcome, intervention, and condition sets."
        )
    if (
        not outcome_set.isdisjoint(intervention_set)
        or not outcome_set.isdisjoint(conditioned_set)
        or not intervention_set.isdisjoint(conditioned_set)
    ):
        raise ValueError(
            "IDC outcome, intervention, and condition sets must be disjoint."
        )
    if not (outcome_set | intervention_set | conditioned_set) <= graph_nodes:
        raise ValueError("IDC query references variables outside the ADMG.")
    expression, witness, exhausted = _identify_idc(
        graph,
        outcomes=outcome_set,
        interventions=intervention_set,
        conditioned=conditioned_set,
        maximum_depth=int(maximum_id_depth),
    )
    if exhausted:
        status = IdentificationStatus.RESOURCE_EXHAUSTED
        reason = "IDC recursion exceeded maximum_id_depth."
    elif expression is None:
        status = IdentificationStatus.NOT_IDENTIFIED
        reason = "IDC reduced to a nonidentified interventional distribution."
    else:
        status = IdentificationStatus.IDENTIFIED
        reason = "IDC produced a conditional observational functional."
    return GraphicalIdentificationResult(
        status=status,
        expression=expression,
        witness=witness,
        graph_id=graph.graph_id,
        reason=reason,
        result_id=canonical_fingerprint(
            {
                "status": status.value,
                "expression_id": (
                    None if expression is None else expression.expression_id
                ),
                "witness_id": None if witness is None else witness.witness_id,
                "graph_id": graph.graph_id,
                "reason": reason,
            }
        ),
    )


def issue_identification_certificate(
    result: IdentificationResult,
    problem: CausalProblem,
) -> IdentificationCertificate:
    if not result.identified or result.basis is None or result.expression is None:
        raise ValueError(
            "Only a successful identification result can issue a certificate."
        )
    if (
        result.law_id != problem.available_law.law_id
        or result.query_id != problem.query.query_id
        or result.assumptions_id != problem.design.assumptions.ledger_id
    ):
        raise ValueError("Identification result is stale for this causal problem.")
    capability = (
        "adjustment_mean"
        if isinstance(result.expression, AdjustmentExpression)
        else "finite_discrete"
    )
    payload = {
        "basis": result.basis.value,
        "expression_id": result.expression.expression_id,
        "structure_id": result.structure_id,
        "schema_id": problem.dataset.schema.schema_id,
        "design_id": problem.design.design_id,
        "law_id": result.law_id,
        "query_id": result.query_id,
        "population_id": problem.query.population.population_id,
        "assumptions_id": result.assumptions_id,
        "evaluator_capability": capability,
    }
    return IdentificationCertificate(
        basis=result.basis,
        expression=result.expression,
        structure_id=result.structure_id,
        schema_id=problem.dataset.schema.schema_id,
        design_id=problem.design.design_id,
        law_id=result.law_id,
        query_id=result.query_id,
        population_id=problem.query.population.population_id,
        assumptions_id=result.assumptions_id,
        evaluator_capability=capability,
        certificate_id=canonical_fingerprint(payload),
    )


def evaluate_finite_effect(
    certificate: IdentificationCertificate,
    problem: CausalProblem,
    law: FiniteObservedLaw,
) -> FiniteEffectResult:
    if certificate.evaluator_capability != "finite_discrete" and not isinstance(
        certificate.expression,
        AdjustmentExpression,
    ):
        return _finite_failure(
            EvaluationStatus.UNSUPPORTED,
            certificate,
            law,
            "Certificate does not carry a finite-discrete evaluator capability.",
        )
    _verify_certificate_problem(certificate, problem)
    if law.schema.schema_id != certificate.schema_id:
        raise ValueError("Finite law schema does not match the certificate.")
    active = _evaluate_regime_mean(
        certificate.expression,
        law,
        problem.query.outcome_variable,
        problem.query.contrast.active.exposure_variable,
        int(problem.query.contrast.active.value),
    )
    reference = _evaluate_regime_mean(
        certificate.expression,
        law,
        problem.query.outcome_variable,
        problem.query.contrast.reference.exposure_variable,
        int(problem.query.contrast.reference.value),
    )
    if active is None or reference is None:
        return _finite_failure(
            EvaluationStatus.UNDEFINED_SUPPORT,
            certificate,
            law,
            "The identified functional has a zero-probability conditioning event.",
        )
    values = np.asarray((active, reference, active - reference), dtype=float)
    if not np.all(np.isfinite(values)):
        return _finite_failure(
            EvaluationStatus.NONFINITE,
            certificate,
            law,
            "Finite identified-functional evaluation produced non-finite values.",
        )
    return FiniteEffectResult(
        status=EvaluationStatus.SUCCESS,
        active_mean=jnp.asarray(active),
        reference_mean=jnp.asarray(reference),
        effect=jnp.asarray(active - reference),
        certificate_id=certificate.certificate_id,
        finite_law_id=law.law_id,
        reason="Finite identified functional evaluated successfully.",
    )


def evaluate_finite_distribution(
    result: GraphicalIdentificationResult,
    law: FiniteObservedLaw,
    *,
    fixed_values: dict[str, int],
) -> FiniteDistributionResult:
    """Evaluate one identified finite distribution at fixed intervention/context values."""
    if not result.identified or result.expression is None:
        return FiniteDistributionResult(
            status=EvaluationStatus.UNSUPPORTED,
            variables=(),
            probabilities=jnp.asarray(jnp.nan),
            expression_id="unavailable",
            law_id=law.law_id,
            reason="A successful graphical identification result is required.",
        )
    factor = _evaluate_expression(result.expression, law)
    names = list(factor.variables)
    values = factor.values
    for name, value in fixed_values.items():
        if name not in names:
            raise ValueError(f"Fixed variable {name!r} is absent from the functional.")
        cardinality = _finite_cardinality(law.schema, name)
        if int(value) < 0 or int(value) >= cardinality:
            raise ValueError(f"Fixed value for {name!r} leaves finite support.")
    for name in sorted(fixed_values, key=names.index, reverse=True):
        axis = names.index(name)
        values = np.take(values, int(fixed_values[name]), axis=axis)
        names.pop(axis)
    retained = [name for name in names if name in factor.random]
    for name in tuple(names):
        if name not in retained:
            axis = names.index(name)
            values = values.sum(axis=axis)
            names.pop(axis)
    if tuple(names) != tuple(retained):
        raise ValueError("Finite conditional functional retained unexpected coordinates.")
    total = float(np.sum(values))
    if not np.isfinite(total) or total <= 0 or not np.all(np.isfinite(values)):
        return FiniteDistributionResult(
            status=EvaluationStatus.UNDEFINED_SUPPORT,
            variables=tuple(names),
            probabilities=jnp.full(np.asarray(values).shape, jnp.nan),
            expression_id=result.expression.expression_id,
            law_id=law.law_id,
            reason="Conditional functional is undefined on the requested support.",
        )
    probabilities = np.asarray(values) / total
    return FiniteDistributionResult(
        status=EvaluationStatus.SUCCESS,
        variables=tuple(names),
        probabilities=jnp.asarray(probabilities),
        expression_id=result.expression.expression_id,
        law_id=law.law_id,
        reason="Finite conditional functional evaluated successfully.",
    )


def _select_adjustment_identification(
    *,
    problem: CausalProblem,
    structure_id: str,
    basis: IdentificationBasis,
    candidates: tuple[tuple[str, ...], ...],
    requested: Iterable[str] | None,
) -> IdentificationResult:
    treatment = problem.query.contrast.active.exposure_variable
    outcome = problem.query.outcome_variable
    if not candidates:
        return _identification_result(
            status=IdentificationStatus.NOT_IDENTIFIED,
            basis=None,
            expression=None,
            problem=problem,
            structure_id=structure_id,
            adjustment_sets=(),
            witness=None,
            reason="No valid observed adjustment set was found.",
        )
    if requested is None:
        if len(candidates) != 1:
            return _identification_result(
                status=IdentificationStatus.AMBIGUOUS,
                basis=basis,
                expression=None,
                problem=problem,
                structure_id=structure_id,
                adjustment_sets=candidates,
                witness=None,
                reason="Multiple valid adjustment sets require an explicit selection.",
            )
        selected = candidates[0]
    else:
        selected = tuple(
            name for name in problem.dataset.schema.names if name in set(requested)
        )
        if selected not in candidates:
            return _identification_result(
                status=IdentificationStatus.NOT_IDENTIFIED,
                basis=None,
                expression=None,
                problem=problem,
                structure_id=structure_id,
                adjustment_sets=candidates,
                witness=None,
                reason="The requested adjustment set does not satisfy the criterion.",
            )
    expression = AdjustmentExpression(
        treatment=treatment,
        outcome=outcome,
        adjustment=selected,
    )
    return _identification_result(
        status=IdentificationStatus.IDENTIFIED,
        basis=basis,
        expression=expression,
        problem=problem,
        structure_id=structure_id,
        adjustment_sets=candidates,
        witness=None,
        reason="The selected covariates satisfy the adjustment criterion.",
    )


def _required_assumption_error(
    problem: CausalProblem,
    *,
    require_graph: bool,
) -> str | None:
    required = [
        AssumptionKind.CONSISTENCY,
        AssumptionKind.POSITIVITY,
        AssumptionKind.NO_INTERFERENCE,
    ]
    if require_graph:
        required.append(AssumptionKind.CAUSAL_MARKOV)
    for kind in required:
        disposition = problem.design.assumptions.disposition(kind)
        if disposition is None:
            return f"Required assumption {kind.value!r} is not declared."
        if disposition in {
            AssumptionDisposition.VIOLATED,
            AssumptionDisposition.UNSUPPORTED,
        }:
            return f"Required assumption {kind.value!r} is not admissible."
    return None


def _identification_result(
    *,
    status: IdentificationStatus,
    basis: IdentificationBasis | None,
    expression: Expression | None,
    problem: CausalProblem,
    structure_id: str | None,
    adjustment_sets: tuple[tuple[str, ...], ...],
    witness: HedgeWitness | None,
    reason: str,
) -> IdentificationResult:
    payload = {
        "status": status.value,
        "basis": None if basis is None else basis.value,
        "expression_id": None if expression is None else expression.expression_id,
        "adjustment_sets": adjustment_sets,
        "structure_id": structure_id,
        "law_id": problem.available_law.law_id,
        "query_id": problem.query.query_id,
        "assumptions_id": problem.design.assumptions.ledger_id,
        "witness_id": None if witness is None else witness.witness_id,
        "reason": reason,
    }
    return IdentificationResult(
        status=status,
        basis=basis,
        expression=expression,
        adjustment_sets=adjustment_sets,
        structure_id=structure_id,
        law_id=problem.available_law.law_id,
        query_id=problem.query.query_id,
        assumptions_id=problem.design.assumptions.ledger_id,
        witness=witness,
        reason=reason,
        result_id=canonical_fingerprint(payload),
    )


def _subgraph(graph: CausalADMG, nodes: set[str]) -> CausalADMG:
    variables = tuple(
        variable for variable in graph.schema.variables if variable.name in nodes
    )
    return CausalADMG(
        schema=CausalSchema(variables),
        directed_edges=(
            edge for edge in graph.directed_edges if edge[0] in nodes and edge[1] in nodes
        ),
        bidirected_edges=(
            edge
            for edge in graph.bidirected_edges
            if edge[0] in nodes and edge[1] in nodes
        ),
    )


def _districts(graph: CausalADMG) -> tuple[tuple[str, ...], ...]:
    adjacency = {name: set() for name in graph.schema.names}
    for left, right in graph.bidirected_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    unseen = set(graph.schema.names)
    result: list[tuple[str, ...]] = []
    while unseen:
        source = next(name for name in graph.schema.names if name in unseen)
        component: set[str] = set()
        stack = [source]
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node] - component)
        unseen.difference_update(component)
        result.append(tuple(name for name in graph.schema.names if name in component))
    return tuple(result)


def _directed_ancestors(graph: CausalADMG, nodes: set[str]) -> set[str]:
    result = set(nodes)
    changed = True
    while changed:
        changed = False
        for source, target in graph.directed_edges:
            if target in result and source not in result:
                result.add(source)
                changed = True
    return result


def _identify_idc(
    graph: CausalADMG,
    *,
    outcomes: set[str],
    interventions: set[str],
    conditioned: set[str],
    maximum_depth: int,
) -> tuple[Expression | None, HedgeWitness | None, bool]:
    def recurse(
        y: set[str],
        x: set[str],
        z: set[str],
        depth: int,
    ) -> tuple[Expression | None, HedgeWitness | None, bool]:
        if depth > maximum_depth:
            return None, None, True
        for variable in graph.schema.names:
            if variable not in z:
                continue
            directed = tuple(
                edge
                for edge in graph.directed_edges
                if edge[1] not in x and edge[0] != variable
            )
            bidirected = tuple(
                edge
                for edge in graph.bidirected_edges
                if edge[0] not in x and edge[1] not in x
            )
            rule_two_graph = CausalADMG(
                schema=graph.schema,
                directed_edges=directed,
                bidirected_edges=bidirected,
            )
            separation = m_separated(
                rule_two_graph,
                y,
                (variable,),
                x | (z - {variable}),
            )
            if separation.status is SeparationStatus.RESOURCE_EXHAUSTED:
                return None, None, True
            if separation.separated:
                return recurse(y, x | {variable}, z - {variable}, depth + 1)
        joint, witness, exhausted = _identify_id(
            graph,
            outcomes=tuple(name for name in graph.schema.names if name in y | z),
            interventions=tuple(name for name in graph.schema.names if name in x),
            maximum_depth=maximum_depth - depth,
        )
        if joint is None:
            return None, witness, exhausted
        if not z:
            return joint, None, exhausted
        return (
            ConditionalExpression(outcomes=y, conditioned=z, operand=joint),
            None,
            exhausted,
        )

    return recurse(outcomes, interventions, conditioned, 0)


def _identify_id(
    graph: CausalADMG,
    *,
    outcomes: tuple[str, ...],
    interventions: tuple[str, ...],
    maximum_depth: int,
) -> tuple[Expression | None, HedgeWitness | None, bool]:
    base = ObservedJointExpression(graph.schema.names)

    def recurse(
        current: CausalADMG,
        y: set[str],
        x: set[str],
        distribution: Expression,
        depth: int,
    ) -> tuple[Expression | None, HedgeWitness | None, bool]:
        if depth > maximum_depth:
            return None, None, True
        vertices = set(current.schema.names)
        if not x:
            return (
                MarginalizeExpression(variables=vertices - y, operand=distribution),
                None,
                False,
            )
        ancestral = _directed_ancestors(current, y)
        if ancestral != vertices:
            restricted_distribution = MarginalizeExpression(
                variables=vertices - ancestral,
                operand=distribution,
            )
            return recurse(
                _subgraph(current, ancestral),
                y,
                x & ancestral,
                restricted_distribution,
                depth + 1,
            )
        without_x = _subgraph(current, vertices - x)
        augmented = (vertices - x) - _directed_ancestors(without_x, y)
        if augmented:
            return recurse(current, y, x | augmented, distribution, depth + 1)
        districts = _districts(without_x)
        if len(districts) > 1:
            terms: list[Expression] = []
            for district in districts:
                term, witness, exhausted = recurse(
                    current,
                    set(district),
                    vertices - set(district),
                    distribution,
                    depth + 1,
                )
                if term is None:
                    return None, witness, exhausted
                terms.append(term)
            return (
                MarginalizeExpression(
                    variables=vertices - (y | x),
                    operand=ProductExpression(terms),
                ),
                None,
                False,
            )
        district = set(districts[0])
        current_districts = _districts(current)
        if len(current_districts) == 1 and set(current_districts[0]) == vertices:
            return None, HedgeWitness(district=vertices, graph_id=current.graph_id), False
        for containing in current_districts:
            containing_set = set(containing)
            if district == containing_set:
                q_factor = QFactorExpression(
                    variables=district,
                    order=current.topological_order,
                    operand=distribution,
                )
                return (
                    MarginalizeExpression(variables=district - y, operand=q_factor),
                    None,
                    False,
                )
            if district < containing_set:
                q_factor = QFactorExpression(
                    variables=containing_set,
                    order=current.topological_order,
                    operand=distribution,
                )
                return recurse(
                    _subgraph(current, containing_set),
                    y,
                    x & containing_set,
                    q_factor,
                    depth + 1,
                )
        return None, HedgeWitness(district=district, graph_id=current.graph_id), False

    return recurse(graph, set(outcomes), set(interventions), base, 0)


class _Factor:
    def __init__(
        self, variables: tuple[str, ...], random: tuple[str, ...], values: np.ndarray
    ):
        self.variables = variables
        self.random = random
        self.values = values


def _evaluate_expression(expression: Expression, law: FiniteObservedLaw) -> _Factor:
    cardinality = dict(zip(law.schema.names, law.cardinalities, strict=True))
    if isinstance(expression, ObservedJointExpression):
        if expression.variables != law.schema.names:
            raise ValueError("Observed-joint expression and finite-law schema differ.")
        return _Factor(
            expression.variables, expression.variables, np.asarray(law.probabilities)
        )
    if isinstance(expression, AdjustmentExpression):
        raise TypeError("Adjustment expressions use the dedicated finite evaluator.")
    if isinstance(expression, ConditionalExpression):
        factor = _evaluate_expression(expression.operand, law)
        denominator = _marginalize_factor(factor, set(expression.outcomes))
        denominator_values = _align_factor(
            denominator,
            factor.variables,
            cardinality,
        )
        values = np.divide(
            factor.values,
            denominator_values,
            out=np.full_like(factor.values, np.nan, dtype=float),
            where=denominator_values > 0,
        )
        return _Factor(factor.variables, expression.outcomes, values)
    if isinstance(expression, MarginalizeExpression):
        factor = _evaluate_expression(expression.operand, law)
        values = factor.values
        variables = list(factor.variables)
        for variable in sorted(
            set(expression.variables) & set(variables),
            key=variables.index,
            reverse=True,
        ):
            axis = variables.index(variable)
            values = values.sum(axis=axis)
            variables.pop(axis)
        random = tuple(node for node in factor.random if node in variables)
        return _Factor(tuple(variables), random, values)
    if isinstance(expression, ProductExpression):
        factors = [_evaluate_expression(operand, law) for operand in expression.operands]
        union = tuple(
            name
            for name in law.schema.names
            if any(name in factor.variables for factor in factors)
        )
        values = np.ones(tuple(cardinality[name] for name in union), dtype=float)
        random: set[str] = set()
        for factor in factors:
            values = values * _align_factor(factor, union, cardinality)
            random.update(factor.random)
        return _Factor(union, tuple(name for name in union if name in random), values)
    if isinstance(expression, QFactorExpression):
        base = _evaluate_expression(expression.operand, law)
        fixed = tuple(name for name in base.variables if name not in base.random)
        selected = set(expression.variables)
        product: _Factor | None = None
        prefix: list[str] = []
        for node in expression.order:
            if node not in base.random:
                continue
            prefix.append(node)
            if node not in selected:
                continue
            keep_numerator = set(fixed) | set(prefix)
            numerator = _marginalize_factor(base, set(base.variables) - keep_numerator)
            denominator = _marginalize_factor(numerator, {node})
            union = tuple(
                name
                for name in law.schema.names
                if name in set(numerator.variables) | set(denominator.variables)
            )
            numerator_values = _align_factor(numerator, union, cardinality)
            denominator_values = _align_factor(denominator, union, cardinality)
            conditional = np.divide(
                numerator_values,
                denominator_values,
                out=np.full_like(numerator_values, np.nan, dtype=float),
                where=denominator_values > 0,
            )
            current = _Factor(union, (node,), conditional)
            if product is None:
                product = current
            else:
                product_union = tuple(
                    name
                    for name in law.schema.names
                    if name in set(product.variables) | set(current.variables)
                )
                product = _Factor(
                    product_union,
                    tuple(name for name in product_union if name in selected),
                    _align_factor(product, product_union, cardinality)
                    * _align_factor(current, product_union, cardinality),
                )
        if product is None:
            raise ValueError("Q-factor selected no variables from its operand.")
        return product
    raise TypeError(f"Unsupported causal expression type {type(expression).__name__}.")


def _marginalize_factor(factor: _Factor, variables: set[str]) -> _Factor:
    values = factor.values
    names = list(factor.variables)
    for variable in sorted(variables & set(names), key=names.index, reverse=True):
        axis = names.index(variable)
        values = values.sum(axis=axis)
        names.pop(axis)
    return _Factor(
        tuple(names),
        tuple(name for name in factor.random if name in names),
        values,
    )


def _align_factor(
    factor: _Factor,
    union: tuple[str, ...],
    cardinality: dict[str, int],
) -> np.ndarray:
    present = tuple(name for name in union if name in factor.variables)
    permutation = tuple(factor.variables.index(name) for name in present)
    values = np.transpose(factor.values, permutation) if permutation else factor.values
    shape = tuple(cardinality[name] if name in factor.variables else 1 for name in union)
    return np.reshape(values, shape)


def _finite_cardinality(schema: CausalSchema, name: str) -> int:
    cardinality = schema.variable(name).cardinality
    if cardinality is None:
        raise ValueError(f"Variable {name!r} is not finite categorical.")
    return cardinality


def _evaluate_regime_mean(
    expression: Expression,
    law: FiniteObservedLaw,
    outcome: str,
    treatment: str,
    treatment_value: int,
) -> float | None:
    outcome_cardinality = _finite_cardinality(law.schema, outcome)
    treatment_cardinality = _finite_cardinality(law.schema, treatment)
    if treatment_value < 0 or treatment_value >= treatment_cardinality:
        raise ValueError("Treatment regime value leaves finite support.")
    if isinstance(expression, AdjustmentExpression):
        return _evaluate_adjustment_mean(expression, law, treatment_value)
    factor = _evaluate_expression(expression, law)
    if treatment not in factor.variables or outcome not in factor.variables:
        raise ValueError("Identified functional lacks treatment or outcome coordinates.")
    values = factor.values
    names = list(factor.variables)
    axis = names.index(treatment)
    values = np.take(values, treatment_value, axis=axis)
    names.pop(axis)
    for variable in tuple(names):
        if variable != outcome:
            other_axis = names.index(variable)
            values = values.sum(axis=other_axis)
            names.pop(other_axis)
    total = float(np.sum(values))
    if not np.isfinite(total) or total <= 0:
        return None
    distribution = np.asarray(values) / total
    if distribution.shape != (outcome_cardinality,):
        raise ValueError("Finite identified outcome distribution has the wrong shape.")
    return float(np.dot(np.arange(outcome_cardinality, dtype=float), distribution))


def _evaluate_adjustment_mean(
    expression: AdjustmentExpression,
    law: FiniteObservedLaw,
    treatment_value: int,
) -> float | None:
    names = law.schema.names
    probabilities = np.asarray(law.probabilities)
    outcome_axis = names.index(expression.outcome)
    treatment_axis = names.index(expression.treatment)
    adjustment_axes = tuple(names.index(name) for name in expression.adjustment)
    outcome_values = np.arange(probabilities.shape[outcome_axis], dtype=float)
    total = 0.0
    adjustment_ranges = [range(probabilities.shape[axis]) for axis in adjustment_axes]
    assignments = itertools.product(*adjustment_ranges) if adjustment_ranges else [()]
    for assignment in assignments:
        index: list[int | slice] = [slice(None)] * probabilities.ndim
        index[treatment_axis] = treatment_value
        for axis, value in zip(adjustment_axes, assignment, strict=True):
            index[axis] = value
        conditional_slice = probabilities[tuple(index)]
        remaining_names = [
            name
            for position, name in enumerate(names)
            if position != treatment_axis and position not in adjustment_axes
        ]
        conditional_outcome_axis = remaining_names.index(expression.outcome)
        other_axes = tuple(
            axis
            for axis in range(conditional_slice.ndim)
            if axis != conditional_outcome_axis
        )
        outcome_mass = (
            conditional_slice.sum(axis=other_axes) if other_axes else conditional_slice
        )
        denominator = float(outcome_mass.sum())
        marginal_index: list[int | slice] = [slice(None)] * probabilities.ndim
        for axis, value in zip(adjustment_axes, assignment, strict=True):
            marginal_index[axis] = value
        marginal_slice = probabilities[tuple(marginal_index)]
        remaining_marginal = [
            name for position, name in enumerate(names) if position not in adjustment_axes
        ]
        axes_to_sum = tuple(range(len(remaining_marginal)))
        p_adjustment = (
            float(marginal_slice.sum(axis=axes_to_sum))
            if axes_to_sum
            else float(marginal_slice)
        )
        if p_adjustment > 0 and denominator <= 0:
            return None
        if p_adjustment > 0:
            total += p_adjustment * float(
                np.dot(outcome_values, outcome_mass / denominator)
            )
    return total


def _finite_failure(
    status: EvaluationStatus,
    certificate: IdentificationCertificate,
    law: FiniteObservedLaw,
    reason: str,
) -> FiniteEffectResult:
    nan = jnp.asarray(jnp.nan)
    return FiniteEffectResult(
        status=status,
        active_mean=nan,
        reference_mean=nan,
        effect=nan,
        certificate_id=certificate.certificate_id,
        finite_law_id=law.law_id,
        reason=reason,
    )


def _verify_certificate_problem(
    certificate: IdentificationCertificate,
    problem: CausalProblem,
) -> None:
    expected = (
        problem.dataset.schema.schema_id,
        problem.design.design_id,
        problem.available_law.law_id,
        problem.query.query_id,
        problem.query.population.population_id,
        problem.design.assumptions.ledger_id,
    )
    actual = (
        certificate.schema_id,
        certificate.design_id,
        certificate.law_id,
        certificate.query_id,
        certificate.population_id,
        certificate.assumptions_id,
    )
    if actual != expected:
        raise ValueError("Identification certificate is stale for this causal problem.")


__all__ = [
    "AdjustmentExpression",
    "ConditionalExpression",
    "EvaluationStatus",
    "FiniteEffectResult",
    "FiniteDistributionResult",
    "FiniteObservedLaw",
    "HedgeWitness",
    "GraphicalIdentificationResult",
    "IdentificationBasis",
    "IdentificationCertificate",
    "IdentificationResult",
    "IdentificationStatus",
    "MarginalizeExpression",
    "ObservedJointExpression",
    "ProductExpression",
    "QFactorExpression",
    "enumerate_adjustment_sets",
    "evaluate_finite_effect",
    "evaluate_finite_distribution",
    "identify_causal_effect",
    "identify_conditional_distribution",
    "issue_identification_certificate",
]
