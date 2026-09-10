#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import enum
from collections.abc import Callable, Iterable, Sequence
from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from .._fingerprint import canonical_fingerprint
from .._identity import callable_payload
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..pgm import (
    DenseTableFactorGroup,
    DiscreteFactorGraph,
    DiscreteVariableGroup,
    enumerate_assignments,
    factor_graph_log_score,
    VariableSelection,
)
from ..uq import PosteriorProblem
from ._core import CausalSchema
from ._graph import CausalDAG, descendants
from ._identify import FiniteObservedLaw


class MechanismKind(enum.StrEnum):
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"
    INVERTIBLE_NOISE = "invertible_noise"
    FINITE_CONDITIONAL = "finite_conditional"


class SCMStatus(enum.IntEnum):
    SUCCESS = 0
    UNSUPPORTED_MECHANISM = 1
    INVALID_OUTPUT = 2
    NONFINITE = 3
    RESOURCE_EXHAUSTED = 4


class CounterfactualStatus(enum.IntEnum):
    SUCCESS = 0
    ABDUCTION_UNSUPPORTED = 1
    INTERVENTION_UNSUPPORTED = 2
    INVALID_FACTUAL = 3
    NONFINITE = 4


class DeterministicMechanism(StrictModule):
    output: str = eqx.field(static=True)
    parents: tuple[str, ...] = eqx.field(static=True)
    function: Callable[[tuple[Array, ...]], Array] = eqx.field(static=True)
    semantic_id: str = eqx.field(static=True)
    numeric_id: str = eqx.field(static=True)
    mechanism_id: str = eqx.field(static=True)
    kind: MechanismKind = eqx.field(static=True)

    def __init__(
        self,
        *,
        output: str,
        parents: Sequence[str],
        function: Callable[[tuple[Array, ...]], Array],
        semantic_id: str | None = None,
        numeric_id: str | None = None,
    ) -> None:
        payload = callable_payload(
            function,
            semantic_id=semantic_id,
            numeric_id=numeric_id,
        )
        mechanism_id = canonical_fingerprint(
            {
                "kind": MechanismKind.DETERMINISTIC.value,
                "output": output,
                "parents": tuple(parents),
                "semantic_id": payload["semantic_content_id"],
                "numeric_id": payload["numeric_content_id"],
            }
        )
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "parents", tuple(parents))
        object.__setattr__(self, "function", function)
        object.__setattr__(self, "semantic_id", payload["semantic_content_id"])
        object.__setattr__(self, "numeric_id", payload["numeric_content_id"])
        object.__setattr__(self, "mechanism_id", mechanism_id)
        object.__setattr__(self, "kind", MechanismKind.DETERMINISTIC)

    def evaluate(self, parents: tuple[Array, ...], /) -> Array:
        return jnp.asarray(self.function(parents))


class StochasticMechanism(StrictModule):
    output: str = eqx.field(static=True)
    parents: tuple[str, ...] = eqx.field(static=True)
    sampler: Callable[[PRNGKeyArray, tuple[Array, ...], int], Array] = eqx.field(
        static=True
    )
    semantic_id: str = eqx.field(static=True)
    numeric_id: str = eqx.field(static=True)
    mechanism_id: str = eqx.field(static=True)
    kind: MechanismKind = eqx.field(static=True)

    def __init__(
        self,
        *,
        output: str,
        parents: Sequence[str],
        sampler: Callable[[PRNGKeyArray, tuple[Array, ...], int], Array],
        semantic_id: str | None = None,
        numeric_id: str | None = None,
    ) -> None:
        payload = callable_payload(
            sampler,
            semantic_id=semantic_id,
            numeric_id=numeric_id,
        )
        mechanism_id = canonical_fingerprint(
            {
                "kind": MechanismKind.STOCHASTIC.value,
                "output": output,
                "parents": tuple(parents),
                "semantic_id": payload["semantic_content_id"],
                "numeric_id": payload["numeric_content_id"],
            }
        )
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "parents", tuple(parents))
        object.__setattr__(self, "sampler", sampler)
        object.__setattr__(self, "semantic_id", payload["semantic_content_id"])
        object.__setattr__(self, "numeric_id", payload["numeric_content_id"])
        object.__setattr__(self, "mechanism_id", mechanism_id)
        object.__setattr__(self, "kind", MechanismKind.STOCHASTIC)

    def sample(
        self,
        key: PRNGKeyArray,
        parents: tuple[Array, ...],
        n_samples: int,
        /,
    ) -> Array:
        return jnp.asarray(self.sampler(key, parents, n_samples))


class InvertibleNoiseMechanism(StrictModule):
    output: str = eqx.field(static=True)
    parents: tuple[str, ...] = eqx.field(static=True)
    noise_sampler: Callable[[PRNGKeyArray, int], Array] = eqx.field(static=True)
    forward: Callable[[tuple[Array, ...], Array], Array] = eqx.field(static=True)
    inverse: Callable[[tuple[Array, ...], Array], Array] = eqx.field(static=True)
    semantic_id: str = eqx.field(static=True)
    numeric_id: str = eqx.field(static=True)
    mechanism_id: str = eqx.field(static=True)
    kind: MechanismKind = eqx.field(static=True)

    def __init__(
        self,
        *,
        output: str,
        parents: Sequence[str],
        noise_sampler: Callable[[PRNGKeyArray, int], Array],
        forward: Callable[[tuple[Array, ...], Array], Array],
        inverse: Callable[[tuple[Array, ...], Array], Array],
        semantic_ids: Sequence[str] | None = None,
        numeric_ids: Sequence[str] | None = None,
    ) -> None:
        semantic = None if semantic_ids is None else tuple(semantic_ids)
        numeric = None if numeric_ids is None else tuple(numeric_ids)
        if (semantic is None) != (numeric is None):
            raise ValueError("semantic_ids and numeric_ids must be supplied together.")
        if (
            semantic is not None
            and numeric is not None
            and (len(semantic) != 3 or len(numeric) != 3)
        ):
            raise ValueError("Invertible mechanisms require three semantic/numeric IDs.")
        callables = (noise_sampler, forward, inverse)
        payloads = tuple(
            callable_payload(
                function,
                semantic_id=None if semantic is None else semantic[index],
                numeric_id=None if numeric is None else numeric[index],
            )
            for index, function in enumerate(callables)
        )
        semantic_id = canonical_fingerprint(
            [payload["semantic_content_id"] for payload in payloads]
        )
        numeric_id = canonical_fingerprint(
            [payload["numeric_content_id"] for payload in payloads]
        )
        mechanism_id = canonical_fingerprint(
            {
                "kind": MechanismKind.INVERTIBLE_NOISE.value,
                "output": output,
                "parents": tuple(parents),
                "semantic_id": semantic_id,
                "numeric_id": numeric_id,
            }
        )
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "parents", tuple(parents))
        object.__setattr__(self, "noise_sampler", noise_sampler)
        object.__setattr__(self, "forward", forward)
        object.__setattr__(self, "inverse", inverse)
        object.__setattr__(self, "semantic_id", semantic_id)
        object.__setattr__(self, "numeric_id", numeric_id)
        object.__setattr__(self, "mechanism_id", mechanism_id)
        object.__setattr__(self, "kind", MechanismKind.INVERTIBLE_NOISE)

    def sample_noise(self, key: PRNGKeyArray, n_samples: int, /) -> Array:
        return jnp.asarray(self.noise_sampler(key, n_samples))

    def evaluate(self, parents: tuple[Array, ...], noise: Array, /) -> Array:
        return jnp.asarray(self.forward(parents, noise))

    def abduct(self, parents: tuple[Array, ...], value: Array, /) -> Array:
        return jnp.asarray(self.inverse(parents, value))


class FiniteConditionalMechanism(StrictModule):
    output: str = eqx.field(static=True)
    parents: tuple[str, ...] = eqx.field(static=True)
    probabilities: Array
    semantic_id: str = eqx.field(static=True)
    numeric_id: str = eqx.field(static=True)
    mechanism_id: str = eqx.field(static=True)
    kind: MechanismKind = eqx.field(static=True)

    def __init__(
        self,
        *,
        output: str,
        parents: Sequence[str],
        probabilities: Array,
    ) -> None:
        canonical = jnp.asarray(probabilities)
        host = np.asarray(canonical)
        if host.ndim < 1 or not np.all(np.isfinite(host)) or np.any(host < 0):
            raise ValueError(
                "Finite mechanism probabilities must be finite and non-negative."
            )
        if not np.allclose(host.sum(axis=-1), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError(
                "Finite mechanism probabilities must normalize over the child."
            )
        semantic_id = canonical_fingerprint(
            {
                "kind": MechanismKind.FINITE_CONDITIONAL.value,
                "output": output,
                "parents": tuple(parents),
                "shape": canonical.shape,
                "dtype": str(canonical.dtype),
            }
        )
        numeric_id = canonical_fingerprint(
            {
                "semantic_id": semantic_id,
                "probabilities": canonical,
            }
        )
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "parents", tuple(parents))
        object.__setattr__(self, "probabilities", canonical)
        object.__setattr__(self, "semantic_id", semantic_id)
        object.__setattr__(self, "numeric_id", numeric_id)
        object.__setattr__(
            self,
            "mechanism_id",
            canonical_fingerprint({"semantic_id": semantic_id, "numeric_id": numeric_id}),
        )
        object.__setattr__(self, "kind", MechanismKind.FINITE_CONDITIONAL)


Mechanism: TypeAlias = (
    DeterministicMechanism
    | StochasticMechanism
    | InvertibleNoiseMechanism
    | FiniteConditionalMechanism
)


class SCMPlan(StrictModule):
    graph: CausalDAG = eqx.field(static=True)
    mechanisms: tuple[Mechanism, ...]
    mechanism_ids: tuple[str, ...] = eqx.field(static=True)
    exogenous_independence: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        graph: CausalDAG,
        mechanisms: Sequence[Mechanism],
        exogenous_independence: bool,
    ) -> None:
        if not exogenous_independence:
            raise ValueError(
                "Correlated exogenous laws require an explicit joint-noise mechanism, "
                "which is not available in this SCM execution contract."
            )
        by_output: dict[str, Mechanism] = {}
        for mechanism in mechanisms:
            if not isinstance(
                mechanism,
                (
                    DeterministicMechanism,
                    StochasticMechanism,
                    InvertibleNoiseMechanism,
                    FiniteConditionalMechanism,
                ),
            ):
                raise TypeError("SCMPlan received an unsupported mechanism type.")
            if mechanism.output in by_output:
                raise ValueError(f"Duplicate mechanism for {mechanism.output!r}.")
            by_output[mechanism.output] = mechanism
        if set(by_output) != set(graph.schema.names):
            raise ValueError("SCMPlan needs exactly one mechanism per graph variable.")
        ordered = tuple(by_output[name] for name in graph.topological_order)
        for mechanism in ordered:
            expected = graph.parents(mechanism.output)
            if mechanism.parents != expected:
                raise ValueError(
                    f"Mechanism {mechanism.output!r} parents {mechanism.parents} "
                    f"do not match graph parents {expected}."
                )
            _validate_mechanism_schema(graph.schema, mechanism)
        mechanism_ids = tuple(mechanism.mechanism_id for mechanism in ordered)
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "mechanisms", ordered)
        object.__setattr__(self, "mechanism_ids", mechanism_ids)
        object.__setattr__(self, "exogenous_independence", True)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "graph_id": graph.graph_id,
                    "mechanism_ids": mechanism_ids,
                    "exogenous_independence": True,
                }
            ),
        )

    def mechanism(self, output: str, /) -> Mechanism:
        self.graph.schema.index(output)
        return self.mechanisms[self.graph.topological_order.index(output)]


class PerfectIntervention(StrictModule, NonTrainableState):
    variable: str = eqx.field(static=True)
    value: Array
    intervention_id: str = eqx.field(static=True)

    def __init__(self, *, variable: str, value: Array | float) -> None:
        canonical = jnp.asarray(value)
        host = np.asarray(canonical)
        if not np.all(np.isfinite(host)):
            raise ValueError("Perfect intervention values must be finite.")
        object.__setattr__(self, "variable", variable)
        object.__setattr__(self, "value", canonical)
        object.__setattr__(
            self,
            "intervention_id",
            canonical_fingerprint(
                {"kind": "perfect", "variable": variable, "value": canonical}
            ),
        )


class ExternalStochasticIntervention(StrictModule):
    variable: str = eqx.field(static=True)
    mechanism: StochasticMechanism
    intervention_id: str = eqx.field(static=True)

    def __init__(self, *, variable: str, mechanism: StochasticMechanism) -> None:
        if mechanism.output != variable or mechanism.parents:
            raise ValueError(
                "External stochastic interventions need a parentless mechanism for the target."
            )
        object.__setattr__(self, "variable", variable)
        object.__setattr__(self, "mechanism", mechanism)
        object.__setattr__(
            self,
            "intervention_id",
            canonical_fingerprint(
                {
                    "kind": "external_stochastic",
                    "variable": variable,
                    "mechanism_id": mechanism.mechanism_id,
                }
            ),
        )


class SoftMechanismIntervention(StrictModule):
    variable: str = eqx.field(static=True)
    mechanism: Mechanism
    intervention_id: str = eqx.field(static=True)

    def __init__(self, *, variable: str, mechanism: Mechanism) -> None:
        if mechanism.output != variable:
            raise ValueError(
                "Soft intervention replacement output must equal its target."
            )
        object.__setattr__(self, "variable", variable)
        object.__setattr__(self, "mechanism", mechanism)
        object.__setattr__(
            self,
            "intervention_id",
            canonical_fingerprint(
                {
                    "kind": "soft_mechanism",
                    "variable": variable,
                    "mechanism_id": mechanism.mechanism_id,
                }
            ),
        )


class PolicyIntervention(StrictModule):
    variable: str = eqx.field(static=True)
    mechanism: DeterministicMechanism | StochasticMechanism
    intervention_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        variable: str,
        mechanism: DeterministicMechanism | StochasticMechanism,
    ) -> None:
        if mechanism.output != variable:
            raise ValueError(
                "Policy intervention replacement output must equal its target."
            )
        object.__setattr__(self, "variable", variable)
        object.__setattr__(self, "mechanism", mechanism)
        object.__setattr__(
            self,
            "intervention_id",
            canonical_fingerprint(
                {
                    "kind": "policy",
                    "variable": variable,
                    "mechanism_id": mechanism.mechanism_id,
                }
            ),
        )


Intervention: TypeAlias = (
    PerfectIntervention
    | ExternalStochasticIntervention
    | SoftMechanismIntervention
    | PolicyIntervention
)


class InterventionSet(StrictModule):
    interventions: tuple[Intervention, ...]
    intervention_set_id: str = eqx.field(static=True)

    def __init__(self, interventions: Iterable[Intervention]) -> None:
        canonical = tuple(interventions)
        targets = tuple(intervention.variable for intervention in canonical)
        if len(set(targets)) != len(targets):
            raise ValueError("An intervention set may target each variable at most once.")
        ordered = tuple(sorted(canonical, key=lambda item: item.variable))
        object.__setattr__(self, "interventions", ordered)
        object.__setattr__(
            self,
            "intervention_set_id",
            canonical_fingerprint(
                {"interventions": [item.intervention_id for item in ordered]}
            ),
        )


class MechanismRegime(StrictModule):
    baseline: SCMPlan
    interventions: InterventionSet
    graph: CausalDAG = eqx.field(static=True)
    regime_id: str = eqx.field(static=True)

    def __init__(self, *, baseline: SCMPlan, interventions: InterventionSet) -> None:
        replacements = {item.variable: item for item in interventions.interventions}
        unknown = set(replacements) - set(baseline.graph.schema.names)
        if unknown:
            raise ValueError(f"Intervention targets are unknown: {sorted(unknown)}.")
        directed = [
            edge for edge in baseline.graph.directed_edges if edge[1] not in replacements
        ]
        for target, intervention in replacements.items():
            mechanism = _replacement_mechanism(intervention)
            if mechanism is None:
                continue
            descendants_of_target = set(descendants(baseline.graph, (target,))) - {target}
            if set(mechanism.parents) & descendants_of_target:
                raise ValueError(
                    "Policy/soft intervention cannot depend on a target descendant."
                )
            directed.extend((parent, target) for parent in mechanism.parents)
        graph = CausalDAG(schema=baseline.graph.schema, directed_edges=directed)
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "interventions", interventions)
        object.__setattr__(self, "graph", graph)
        object.__setattr__(
            self,
            "regime_id",
            canonical_fingerprint(
                {
                    "baseline_id": baseline.plan_id,
                    "intervention_set_id": interventions.intervention_set_id,
                    "graph_id": graph.graph_id,
                }
            ),
        )


class SCMResult(StrictModule, NonTrainableState):
    values: tuple[Array, ...]
    schema: CausalSchema = eqx.field(static=True)
    status: Array
    regime_id: str = eqx.field(static=True)
    n_samples: int = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == int(SCMStatus.SUCCESS)

    def value(self, name: str, /) -> Array:
        return self.values[self.schema.index(name)]


class FactualObservation(StrictModule, NonTrainableState):
    schema: CausalSchema = eqx.field(static=True)
    values: tuple[Array, ...]
    factual_id: str = eqx.field(static=True)

    def __init__(self, *, schema: CausalSchema, values: Sequence[Array]) -> None:
        canonical = tuple(jnp.asarray(value) for value in values)
        if len(canonical) != len(schema.variables):
            raise ValueError("FactualObservation needs one value per schema variable.")
        n_samples = int(canonical[0].shape[0])
        for variable, value in zip(schema.variables, canonical, strict=True):
            if value.shape != (n_samples,) + variable.event_shape:
                raise ValueError("Factual observation shapes do not match the schema.")
            if not np.all(np.isfinite(np.asarray(value))):
                raise ValueError("Factual observations must be finite.")
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "values", canonical)
        object.__setattr__(
            self,
            "factual_id",
            canonical_fingerprint({"schema_id": schema.schema_id, "values": canonical}),
        )


class AbductionResult(StrictModule, NonTrainableState):
    status: CounterfactualStatus = eqx.field(static=True)
    noises: tuple[Array, ...]
    factual_id: str = eqx.field(static=True)
    baseline_id: str = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    abduction_id: str = eqx.field(static=True)

    @property
    def successful(self) -> bool:
        return self.status is CounterfactualStatus.SUCCESS


class CounterfactualResult(StrictModule, NonTrainableState):
    status: CounterfactualStatus = eqx.field(static=True)
    values: tuple[Array, ...]
    schema: CausalSchema = eqx.field(static=True)
    factual_id: str = eqx.field(static=True)
    abduction_id: str = eqx.field(static=True)
    regime_id: str = eqx.field(static=True)
    reason: str = eqx.field(static=True)

    @property
    def successful(self) -> bool:
        return self.status is CounterfactualStatus.SUCCESS

    def value(self, name: str, /) -> Array:
        return self.values[self.schema.index(name)]


class CompiledFiniteSCM(StrictModule):
    factor_graph: DiscreteFactorGraph
    schema: CausalSchema = eqx.field(static=True)
    regime_id: str = eqx.field(static=True)
    compilation_id: str = eqx.field(static=True)


class SCMPosteriorBinding(StrictModule):
    scm_plan_id: str = eqx.field(static=True)
    posterior_problem: PosteriorProblem
    binding_id: str = eqx.field(static=True)


def baseline_regime(plan: SCMPlan) -> MechanismRegime:
    return MechanismRegime(baseline=plan, interventions=InterventionSet(()))


def build_intervention_regime(
    plan: SCMPlan,
    interventions: Iterable[Intervention],
) -> MechanismRegime:
    return MechanismRegime(
        baseline=plan,
        interventions=InterventionSet(interventions),
    )


def sample_scm(
    regime: MechanismRegime,
    key: PRNGKeyArray,
    n_samples: int,
) -> SCMResult:
    """Sample a fixed mechanism regime in topological order."""
    count = int(n_samples)
    if count < 1:
        raise ValueError("n_samples must be positive.")
    replacements = {item.variable: item for item in regime.interventions.interventions}
    values_by_name: dict[str, Array] = {}
    status = jnp.asarray(int(SCMStatus.SUCCESS), dtype=jnp.int32)
    for node in regime.graph.topological_order:
        intervention = replacements.get(node)
        mechanism = (
            regime.baseline.mechanism(node)
            if intervention is None
            else _replacement_mechanism(intervention)
        )
        if isinstance(intervention, PerfectIntervention):
            variable = regime.graph.schema.variable(node)
            value = jnp.asarray(intervention.value)
            if value.shape == variable.event_shape:
                value = jnp.broadcast_to(value, (count,) + variable.event_shape)
            elif value.shape != (count,) + variable.event_shape:
                raise ValueError("Perfect intervention value has an incompatible shape.")
        elif mechanism is None:
            raise TypeError("Intervention does not define an executable replacement.")
        else:
            parent_values = tuple(values_by_name[parent] for parent in mechanism.parents)
            node_key = jax.random.fold_in(key, regime.graph.schema.index(node))
            value = _sample_mechanism(mechanism, node_key, parent_values, count)
        expected = (count,) + regime.graph.schema.variable(node).event_shape
        if value.shape != expected:
            raise ValueError(
                f"Mechanism {node!r} returned shape {value.shape}; expected {expected}."
            )
        status = jnp.where(
            jnp.all(jnp.isfinite(value)),
            status,
            jnp.asarray(int(SCMStatus.NONFINITE), dtype=jnp.int32),
        )
        values_by_name[node] = value
    values = tuple(values_by_name[name] for name in regime.graph.schema.names)
    return SCMResult(
        values=values,
        schema=regime.graph.schema,
        status=status,
        regime_id=regime.regime_id,
        n_samples=count,
    )


def abduct_factual(
    plan: SCMPlan,
    factual: FactualObservation,
) -> AbductionResult:
    if factual.schema.schema_id != plan.graph.schema.schema_id:
        raise ValueError("Factual observation and SCM schemas must match.")
    values = {
        name: factual.values[index] for index, name in enumerate(factual.schema.names)
    }
    noises: list[Array] = []
    for mechanism in plan.mechanisms:
        parent_values = tuple(values[parent] for parent in mechanism.parents)
        if isinstance(mechanism, InvertibleNoiseMechanism):
            noise = mechanism.abduct(parent_values, values[mechanism.output])
        elif isinstance(mechanism, DeterministicMechanism):
            predicted = mechanism.evaluate(parent_values)
            if not np.allclose(
                np.asarray(predicted),
                np.asarray(values[mechanism.output]),
                rtol=1e-8,
                atol=1e-10,
            ):
                return _abduction_failure(
                    plan,
                    factual,
                    "Factual value violates a deterministic structural mechanism.",
                )
            noise = jnp.zeros((factual.values[0].shape[0], 0))
        else:
            return _abduction_failure(
                plan,
                factual,
                f"Mechanism {mechanism.output!r} lacks an abductive capability.",
            )
        noises.append(noise)
    abduction_id = canonical_fingerprint(
        {
            "baseline_id": plan.plan_id,
            "factual_id": factual.factual_id,
            "noises": tuple(noises),
        }
    )
    return AbductionResult(
        status=CounterfactualStatus.SUCCESS,
        noises=tuple(noises),
        factual_id=factual.factual_id,
        baseline_id=plan.plan_id,
        reason="All stochastic mechanisms supplied exact inverse-noise abduction.",
        abduction_id=abduction_id,
    )


def evaluate_counterfactual(
    regime: MechanismRegime,
    factual: FactualObservation,
    abduction: AbductionResult,
) -> CounterfactualResult:
    if not abduction.successful:
        return _counterfactual_failure(
            regime,
            factual,
            abduction,
            CounterfactualStatus.ABDUCTION_UNSUPPORTED,
            abduction.reason,
        )
    if (
        abduction.baseline_id != regime.baseline.plan_id
        or abduction.factual_id != factual.factual_id
    ):
        raise ValueError("Abduction evidence is stale for this factual SCM query.")
    if any(
        not isinstance(intervention, PerfectIntervention)
        for intervention in regime.interventions.interventions
    ):
        return _counterfactual_failure(
            regime,
            factual,
            abduction,
            CounterfactualStatus.INTERVENTION_UNSUPPORTED,
            "Counterfactual execution currently supports perfect interventions only.",
        )
    count = int(factual.values[0].shape[0])
    replacements = {item.variable: item for item in regime.interventions.interventions}
    noise_by_output = {
        mechanism.output: noise
        for mechanism, noise in zip(
            regime.baseline.mechanisms, abduction.noises, strict=True
        )
    }
    values: dict[str, Array] = {}
    for node in regime.graph.topological_order:
        intervention = replacements.get(node)
        variable = regime.graph.schema.variable(node)
        if isinstance(intervention, PerfectIntervention):
            value = jnp.asarray(intervention.value)
            if value.shape == variable.event_shape:
                value = jnp.broadcast_to(value, (count,) + variable.event_shape)
        else:
            mechanism = regime.baseline.mechanism(node)
            parents = tuple(values[parent] for parent in mechanism.parents)
            if isinstance(mechanism, InvertibleNoiseMechanism):
                value = mechanism.evaluate(parents, noise_by_output[node])
            elif isinstance(mechanism, DeterministicMechanism):
                value = mechanism.evaluate(parents)
            else:
                return _counterfactual_failure(
                    regime,
                    factual,
                    abduction,
                    CounterfactualStatus.ABDUCTION_UNSUPPORTED,
                    f"Mechanism {node!r} lacks exact abductive evaluation.",
                )
        if value.shape != (count,) + variable.event_shape:
            return _counterfactual_failure(
                regime,
                factual,
                abduction,
                CounterfactualStatus.INVALID_FACTUAL,
                f"Counterfactual mechanism {node!r} returned an invalid shape.",
            )
        values[node] = value
    ordered = tuple(values[name] for name in regime.graph.schema.names)
    if not all(np.all(np.isfinite(np.asarray(value))) for value in ordered):
        return _counterfactual_failure(
            regime,
            factual,
            abduction,
            CounterfactualStatus.NONFINITE,
            "Counterfactual execution produced non-finite values.",
        )
    return CounterfactualResult(
        status=CounterfactualStatus.SUCCESS,
        values=ordered,
        schema=regime.graph.schema,
        factual_id=factual.factual_id,
        abduction_id=abduction.abduction_id,
        regime_id=regime.regime_id,
        reason="Counterfactual evaluated with the abducted exogenous state.",
    )


def compile_finite_scm(regime: MechanismRegime) -> CompiledFiniteSCM:
    schema = regime.graph.schema
    cardinalities = []
    for variable in schema.variables:
        if variable.cardinality is None or variable.event_shape:
            raise ValueError("Finite SCM compilation requires scalar finite variables.")
        cardinalities.append(variable.cardinality)
    group = DiscreteVariableGroup(
        "causal",
        shape=(len(schema.variables),),
        num_states=jnp.asarray(cardinalities, dtype=jnp.int32),
    )
    replacements = {item.variable: item for item in regime.interventions.interventions}
    factors: list[DenseTableFactorGroup] = []
    for node in regime.graph.topological_order:
        intervention = replacements.get(node)
        if isinstance(intervention, PerfectIntervention):
            child_cardinality = _finite_cardinality(schema, node)
            value = int(np.asarray(intervention.value))
            if value < 0 or value >= child_cardinality:
                raise ValueError("Perfect finite intervention leaves the child support.")
            table = np.zeros((child_cardinality,), dtype=float)
            table[value] = 1.0
            parents: tuple[str, ...] = ()
        else:
            mechanism = (
                regime.baseline.mechanism(node)
                if intervention is None
                else _replacement_mechanism(intervention)
            )
            if not isinstance(mechanism, FiniteConditionalMechanism):
                raise TypeError(
                    "Finite SCM compilation requires finite conditional replacements."
                )
            parents = mechanism.parents
            table = np.asarray(mechanism.probabilities)
        scope_names = parents + (node,)
        selections = tuple(
            VariableSelection(group, jnp.asarray([schema.index(name)], dtype=jnp.int32))
            for name in scope_names
        )
        log_table = np.where(table > 0, np.log(table), -np.inf)
        factors.append(
            DenseTableFactorGroup(
                selections,
                jnp.asarray(log_table[None, ...]),
            )
        )
    factor_graph = DiscreteFactorGraph((group,), tuple(factors))
    compilation_id = canonical_fingerprint(
        {
            "regime_id": regime.regime_id,
            "factor_graph_structure_id": factor_graph.structure_id,
            "mechanism_ids": regime.baseline.mechanism_ids,
        }
    )
    return CompiledFiniteSCM(
        factor_graph=factor_graph,
        schema=schema,
        regime_id=regime.regime_id,
        compilation_id=compilation_id,
    )


def finite_observed_law_from_scm(
    compiled: CompiledFiniteSCM,
    *,
    maximum_assignments: int = 1_000_000,
) -> FiniteObservedLaw:
    cardinalities = tuple(
        int(value) for value in np.asarray(compiled.factor_graph.cardinalities)
    )
    total = int(np.prod(cardinalities, dtype=np.int64))
    if total > maximum_assignments:
        raise ValueError("Finite SCM law exceeds maximum_assignments.")
    assignments = enumerate_assignments(compiled.factor_graph.cardinalities)
    log_scores = factor_graph_log_score(compiled.factor_graph, assignments)
    log_normalizer = jax.scipy.special.logsumexp(log_scores)
    probabilities = jnp.exp(log_scores - log_normalizer).reshape(cardinalities)
    return FiniteObservedLaw(schema=compiled.schema, probabilities=probabilities)


def bind_scm_posterior(
    plan: SCMPlan,
    posterior_problem: PosteriorProblem,
    *,
    posterior_semantic_id: str,
    posterior_numeric_id: str,
) -> SCMPosteriorBinding:
    semantic_id = str(posterior_semantic_id).strip()
    numeric_id = str(posterior_numeric_id).strip()
    if not semantic_id or not numeric_id:
        raise ValueError("Posterior semantic and numeric IDs must be non-empty.")
    binding_id = canonical_fingerprint(
        {
            "scm_plan_id": plan.plan_id,
            "posterior_semantic_id": semantic_id,
            "posterior_numeric_id": numeric_id,
        }
    )
    return SCMPosteriorBinding(
        scm_plan_id=plan.plan_id,
        posterior_problem=posterior_problem,
        binding_id=binding_id,
    )


def _replacement_mechanism(intervention: Intervention) -> Mechanism | None:
    if isinstance(intervention, PerfectIntervention):
        return None
    if isinstance(intervention, ExternalStochasticIntervention):
        return intervention.mechanism
    if isinstance(intervention, SoftMechanismIntervention):
        return intervention.mechanism
    if isinstance(intervention, PolicyIntervention):
        return intervention.mechanism
    raise TypeError("Unsupported intervention type.")


def _sample_mechanism(
    mechanism: Mechanism,
    key: PRNGKeyArray,
    parents: tuple[Array, ...],
    n_samples: int,
) -> Array:
    if isinstance(mechanism, DeterministicMechanism):
        return mechanism.evaluate(parents)
    if isinstance(mechanism, StochasticMechanism):
        return mechanism.sample(key, parents, n_samples)
    if isinstance(mechanism, InvertibleNoiseMechanism):
        return mechanism.evaluate(parents, mechanism.sample_noise(key, n_samples))
    if isinstance(mechanism, FiniteConditionalMechanism):
        if parents:
            indices = tuple(jnp.asarray(parent, dtype=jnp.int32) for parent in parents)
            probabilities = mechanism.probabilities[indices]
        else:
            probabilities = jnp.broadcast_to(
                mechanism.probabilities,
                (n_samples, mechanism.probabilities.shape[-1]),
            )
        return jax.random.categorical(key, jnp.log(probabilities), axis=-1).astype(
            jnp.int32
        )
    raise TypeError("Unsupported mechanism type.")


def _finite_cardinality(schema: CausalSchema, name: str) -> int:
    cardinality = schema.variable(name).cardinality
    if cardinality is None:
        raise ValueError(f"Variable {name!r} is not finite categorical.")
    return cardinality


def _validate_mechanism_schema(schema: CausalSchema, mechanism: Mechanism) -> None:
    schema.index(mechanism.output)
    for parent in mechanism.parents:
        schema.index(parent)
    if isinstance(mechanism, FiniteConditionalMechanism):
        child = schema.variable(mechanism.output)
        parent_variables = tuple(schema.variable(parent) for parent in mechanism.parents)
        if child.cardinality is None or any(
            variable.cardinality is None for variable in parent_variables
        ):
            raise ValueError(
                "Finite mechanisms require finite parent and child variables."
            )
        expected = tuple(
            _finite_cardinality(schema, variable.name) for variable in parent_variables
        )
        expected += (_finite_cardinality(schema, child.name),)
        if mechanism.probabilities.shape != expected:
            raise ValueError(
                f"Finite mechanism {mechanism.output!r} shape "
                f"{mechanism.probabilities.shape} does not match {expected}."
            )


def _abduction_failure(
    plan: SCMPlan,
    factual: FactualObservation,
    reason: str,
) -> AbductionResult:
    return AbductionResult(
        status=CounterfactualStatus.ABDUCTION_UNSUPPORTED,
        noises=(),
        factual_id=factual.factual_id,
        baseline_id=plan.plan_id,
        reason=reason,
        abduction_id=canonical_fingerprint(
            {
                "status": int(CounterfactualStatus.ABDUCTION_UNSUPPORTED),
                "baseline_id": plan.plan_id,
                "factual_id": factual.factual_id,
                "reason": reason,
            }
        ),
    )


def _counterfactual_failure(
    regime: MechanismRegime,
    factual: FactualObservation,
    abduction: AbductionResult,
    status: CounterfactualStatus,
    reason: str,
) -> CounterfactualResult:
    return CounterfactualResult(
        status=status,
        values=(),
        schema=regime.graph.schema,
        factual_id=factual.factual_id,
        abduction_id=abduction.abduction_id,
        regime_id=regime.regime_id,
        reason=reason,
    )


__all__ = [
    "AbductionResult",
    "CompiledFiniteSCM",
    "CounterfactualResult",
    "CounterfactualStatus",
    "DeterministicMechanism",
    "ExternalStochasticIntervention",
    "FactualObservation",
    "FiniteConditionalMechanism",
    "InterventionSet",
    "InvertibleNoiseMechanism",
    "MechanismKind",
    "MechanismRegime",
    "PerfectIntervention",
    "PolicyIntervention",
    "SCMPlan",
    "SCMPosteriorBinding",
    "SCMResult",
    "SCMStatus",
    "SoftMechanismIntervention",
    "StochasticMechanism",
    "abduct_factual",
    "baseline_regime",
    "bind_scm_posterior",
    "build_intervention_regime",
    "compile_finite_scm",
    "evaluate_counterfactual",
    "finite_observed_law_from_scm",
    "sample_scm",
]
