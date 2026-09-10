#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
import enum
import itertools
from collections import deque
from collections.abc import Iterable, Sequence

import equinox as eqx
import jax
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._core import CausalDataset, CausalSchema, VariableObservability, VariableScale
from ._graph import (
    CausalCPDAG,
    CausalDAG,
    CausalPAG,
    CausalPDAG,
    complete_dag_equivalence_class,
    EndpointMark,
    verified_cpdag,
)


class CIStatus(enum.IntEnum):
    SUCCESS = 0
    INSUFFICIENT_DATA = 1
    DEGENERATE = 2
    SPARSE_CELLS = 3
    NONFINITE = 4
    UNSUPPORTED = 5


class DiscoveryStatus(enum.StrEnum):
    SUCCESS = "success"
    INCOMPLETE = "incomplete"
    CI_FAILED = "ci_failed"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    AMBIGUOUS = "ambiguous"


class GraphFalsificationStatus(enum.StrEnum):
    REJECTED = "rejected"
    NOT_REJECTED = "not_rejected"
    INCONCLUSIVE = "inconclusive"
    NOT_APPLICABLE = "not_applicable"


class ConditionalIndependenceResult(StrictModule, NonTrainableState):
    status: CIStatus = eqx.field(static=True)
    independent: bool = eqx.field(static=True)
    statistic: float = eqx.field(static=True)
    p_value: float = eqx.field(static=True)
    effective_samples: int = eqx.field(static=True)
    degrees_of_freedom: int = eqx.field(static=True)
    left: str = eqx.field(static=True)
    right: str = eqx.field(static=True)
    conditioned: tuple[str, ...] = eqx.field(static=True)
    test_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    @property
    def successful(self) -> bool:
        return self.status is CIStatus.SUCCESS


class AbstractConditionalIndependenceTest(StrictModule):
    test_id: str = eqx.field(static=True)
    alpha: float = eqx.field(static=True)

    @abc.abstractmethod
    def test(
        self,
        dataset: CausalDataset,
        left: str,
        right: str,
        conditioned: Sequence[str] = (),
        *,
        sample_indices: Array | None = None,
        key: PRNGKeyArray | None = None,
    ) -> ConditionalIndependenceResult:
        raise NotImplementedError


class FisherZTest(AbstractConditionalIndependenceTest):
    minimum_variance: float = eqx.field(static=True)

    def __init__(self, *, alpha: float = 0.05, minimum_variance: float = 1e-12) -> None:
        _validate_alpha(alpha)
        if minimum_variance <= 0:
            raise ValueError("minimum_variance must be positive.")
        object.__setattr__(self, "alpha", float(alpha))
        object.__setattr__(self, "minimum_variance", float(minimum_variance))
        object.__setattr__(
            self,
            "test_id",
            canonical_fingerprint(
                {
                    "kind": "fisher_z",
                    "alpha": float(alpha),
                    "minimum_variance": float(minimum_variance),
                }
            ),
        )

    def test(
        self,
        dataset: CausalDataset,
        left: str,
        right: str,
        conditioned: Sequence[str] = (),
        *,
        sample_indices: Array | None = None,
        key: PRNGKeyArray | None = None,
    ) -> ConditionalIndependenceResult:
        del key
        names = (left, right) + tuple(conditioned)
        _validate_ci_variables(dataset.schema, names, continuous=True)
        matrix, count = _ci_matrix(dataset, names, sample_indices)
        conditioning_size = len(conditioned)
        if count <= conditioning_size + 3:
            return _ci_result(
                self,
                CIStatus.INSUFFICIENT_DATA,
                left,
                right,
                conditioned,
                statistic=np.nan,
                p_value=np.nan,
                effective_samples=count,
                degrees_of_freedom=count - conditioning_size - 3,
            )
        if not np.all(np.isfinite(matrix)):
            return _ci_result(
                self,
                CIStatus.NONFINITE,
                left,
                right,
                conditioned,
                statistic=np.nan,
                p_value=np.nan,
                effective_samples=count,
                degrees_of_freedom=count - conditioning_size - 3,
            )
        x = matrix[:, 0]
        y = matrix[:, 1]
        if conditioning_size:
            z = np.column_stack((np.ones((count,)), matrix[:, 2:]))
            if np.linalg.matrix_rank(z) < z.shape[1]:
                return _ci_result(
                    self,
                    CIStatus.DEGENERATE,
                    left,
                    right,
                    conditioned,
                    statistic=np.nan,
                    p_value=np.nan,
                    effective_samples=count,
                    degrees_of_freedom=count - conditioning_size - 3,
                )
            x = x - z @ np.linalg.lstsq(z, x, rcond=None)[0]
            y = y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
        x_variance = float(np.dot(x - x.mean(), x - x.mean()))
        y_variance = float(np.dot(y - y.mean(), y - y.mean()))
        if x_variance <= self.minimum_variance or y_variance <= self.minimum_variance:
            return _ci_result(
                self,
                CIStatus.DEGENERATE,
                left,
                right,
                conditioned,
                statistic=np.nan,
                p_value=np.nan,
                effective_samples=count,
                degrees_of_freedom=count - conditioning_size - 3,
            )
        correlation = float(
            np.dot(x - x.mean(), y - y.mean()) / np.sqrt(x_variance * y_variance)
        )
        if not -1.0 < correlation < 1.0:
            return _ci_result(
                self,
                CIStatus.DEGENERATE,
                left,
                right,
                conditioned,
                statistic=np.nan,
                p_value=np.nan,
                effective_samples=count,
                degrees_of_freedom=count - conditioning_size - 3,
            )
        statistic = float(
            np.arctanh(correlation) * np.sqrt(count - conditioning_size - 3)
        )
        p_value = float(2.0 * jax.scipy.special.ndtr(-abs(statistic)))
        return _ci_result(
            self,
            CIStatus.SUCCESS,
            left,
            right,
            conditioned,
            statistic=statistic,
            p_value=p_value,
            effective_samples=count,
            degrees_of_freedom=count - conditioning_size - 3,
        )


class GSquareTest(AbstractConditionalIndependenceTest):
    minimum_expected_count: float = eqx.field(static=True)

    def __init__(
        self, *, alpha: float = 0.05, minimum_expected_count: float = 5.0
    ) -> None:
        _validate_alpha(alpha)
        if minimum_expected_count <= 0:
            raise ValueError("minimum_expected_count must be positive.")
        object.__setattr__(self, "alpha", float(alpha))
        object.__setattr__(self, "minimum_expected_count", float(minimum_expected_count))
        object.__setattr__(
            self,
            "test_id",
            canonical_fingerprint(
                {
                    "kind": "g_square",
                    "alpha": float(alpha),
                    "minimum_expected_count": float(minimum_expected_count),
                }
            ),
        )

    def test(
        self,
        dataset: CausalDataset,
        left: str,
        right: str,
        conditioned: Sequence[str] = (),
        *,
        sample_indices: Array | None = None,
        key: PRNGKeyArray | None = None,
    ) -> ConditionalIndependenceResult:
        del key
        names = (left, right) + tuple(conditioned)
        _validate_ci_variables(dataset.schema, names, continuous=False)
        matrix, count = _ci_matrix(dataset, names, sample_indices)
        left_cardinality = _finite_cardinality(dataset.schema, left)
        right_cardinality = _finite_cardinality(dataset.schema, right)
        conditioned_cards = tuple(
            _finite_cardinality(dataset.schema, name) for name in conditioned
        )
        assignments = (
            itertools.product(*(range(cardinality) for cardinality in conditioned_cards))
            if conditioned_cards
            else [()]
        )
        statistic = 0.0
        degrees = 0
        sparse = False
        for assignment in assignments:
            mask = np.ones((count,), dtype=bool)
            for offset, value in enumerate(assignment):
                mask &= matrix[:, offset + 2] == value
            table = np.zeros((left_cardinality, right_cardinality), dtype=float)
            np.add.at(
                table,
                (matrix[mask, 0].astype(int), matrix[mask, 1].astype(int)),
                1.0,
            )
            total = table.sum()
            if total <= 0:
                continue
            expected = (
                table.sum(axis=1, keepdims=True)
                * table.sum(axis=0, keepdims=True)
                / total
            )
            positive_expected = expected > 0
            sparse |= bool(
                np.any(expected[positive_expected] < self.minimum_expected_count)
            )
            positive_observed = table > 0
            statistic += float(
                2.0
                * np.sum(
                    table[positive_observed]
                    * np.log(table[positive_observed] / expected[positive_observed])
                )
            )
            nonzero_rows = int(np.sum(table.sum(axis=1) > 0))
            nonzero_columns = int(np.sum(table.sum(axis=0) > 0))
            degrees += max(nonzero_rows - 1, 0) * max(nonzero_columns - 1, 0)
        if degrees <= 0:
            status = CIStatus.DEGENERATE
            p_value = np.nan
        elif sparse:
            status = CIStatus.SPARSE_CELLS
            p_value = np.nan
        else:
            status = CIStatus.SUCCESS
            p_value = float(jax.scipy.special.gammaincc(0.5 * degrees, 0.5 * statistic))
        return _ci_result(
            self,
            status,
            left,
            right,
            conditioned,
            statistic=statistic,
            p_value=p_value,
            effective_samples=count,
            degrees_of_freedom=degrees,
        )


class KernelConditionalIndependenceTest(AbstractConditionalIndependenceTest):
    bandwidth: float = eqx.field(static=True)
    ridge: float = eqx.field(static=True)
    permutations: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        bandwidth: float = 1.0,
        ridge: float = 1e-3,
        permutations: int = 199,
    ) -> None:
        _validate_alpha(alpha)
        if bandwidth <= 0 or ridge <= 0 or int(permutations) < 19:
            raise ValueError(
                "Kernel CI bandwidth/ridge must be positive and permutations >= 19."
            )
        object.__setattr__(self, "alpha", float(alpha))
        object.__setattr__(self, "bandwidth", float(bandwidth))
        object.__setattr__(self, "ridge", float(ridge))
        object.__setattr__(self, "permutations", int(permutations))
        object.__setattr__(
            self,
            "test_id",
            canonical_fingerprint(
                {
                    "kind": "kernel_ci",
                    "alpha": float(alpha),
                    "bandwidth": float(bandwidth),
                    "ridge": float(ridge),
                    "permutations": int(permutations),
                }
            ),
        )

    def test(
        self,
        dataset: CausalDataset,
        left: str,
        right: str,
        conditioned: Sequence[str] = (),
        *,
        sample_indices: Array | None = None,
        key: PRNGKeyArray | None = None,
    ) -> ConditionalIndependenceResult:
        if key is None:
            raise ValueError("Kernel CI requires an explicit JAX key.")
        names = (left, right) + tuple(conditioned)
        _validate_ci_variables(dataset.schema, names, continuous=True)
        matrix, count = _ci_matrix(dataset, names, sample_indices)
        if count < 5:
            return _ci_result(
                self,
                CIStatus.INSUFFICIENT_DATA,
                left,
                right,
                conditioned,
                statistic=np.nan,
                p_value=np.nan,
                effective_samples=count,
                degrees_of_freedom=-1,
            )
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        x_kernel = _rbf_kernel(centered[:, :1], self.bandwidth)
        y_kernel = _rbf_kernel(centered[:, 1:2], self.bandwidth)
        centering = np.eye(count) - np.ones((count, count)) / count
        if conditioned:
            z_kernel = _rbf_kernel(centered[:, 2:], self.bandwidth)
            residual = np.eye(count) - z_kernel @ np.linalg.solve(
                z_kernel + self.ridge * np.eye(count),
                np.eye(count),
            )
            x_kernel = residual @ x_kernel @ residual.T
            y_kernel = residual @ y_kernel @ residual.T
        x_kernel = centering @ x_kernel @ centering
        y_kernel = centering @ y_kernel @ centering
        statistic = float(np.sum(x_kernel * y_kernel) / (count * count))
        permutation_keys = jax.random.split(key, self.permutations)
        permuted = np.empty((self.permutations,), dtype=float)
        for index, permutation_key in enumerate(permutation_keys):
            permutation = np.asarray(jax.random.permutation(permutation_key, count))
            shuffled = y_kernel[np.ix_(permutation, permutation)]
            permuted[index] = np.sum(x_kernel * shuffled) / (count * count)
        p_value = float((1 + np.sum(permuted >= statistic)) / (self.permutations + 1))
        return _ci_result(
            self,
            CIStatus.SUCCESS,
            left,
            right,
            conditioned,
            statistic=statistic,
            p_value=p_value,
            effective_samples=count,
            degrees_of_freedom=-1,
        )


class DiscoveryBackgroundKnowledge(StrictModule, NonTrainableState):
    required_adjacencies: tuple[tuple[str, str], ...] = eqx.field(static=True)
    forbidden_adjacencies: tuple[tuple[str, str], ...] = eqx.field(static=True)
    required_orientations: tuple[tuple[str, str], ...] = eqx.field(static=True)
    forbidden_orientations: tuple[tuple[str, str], ...] = eqx.field(static=True)
    knowledge_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        schema: CausalSchema,
        required_adjacencies: Iterable[tuple[str, str]] = (),
        forbidden_adjacencies: Iterable[tuple[str, str]] = (),
        required_orientations: Iterable[tuple[str, str]] = (),
        forbidden_orientations: Iterable[tuple[str, str]] = (),
    ) -> None:
        forbidden_adj = _canonical_pairs(schema, forbidden_adjacencies)
        required_dir = _canonical_directed(schema, required_orientations)
        forbidden_dir = _canonical_directed(schema, forbidden_orientations)
        required_adj = _canonical_pairs(
            schema,
            tuple(required_adjacencies) + tuple(required_dir),
        )
        if set(required_adj) & set(forbidden_adj):
            raise ValueError(
                "Background knowledge requires and forbids the same adjacency."
            )
        if set(required_dir) & set(forbidden_dir):
            raise ValueError(
                "Background knowledge requires and forbids the same orientation."
            )
        if any((target, source) in set(required_dir) for source, target in required_dir):
            raise ValueError(
                "Background knowledge contains opposing required orientations."
            )
        for source, target in required_dir:
            if _pair(schema, source, target) in forbidden_adj:
                raise ValueError("A required orientation has a forbidden adjacency.")
        object.__setattr__(self, "required_adjacencies", required_adj)
        object.__setattr__(self, "forbidden_adjacencies", forbidden_adj)
        object.__setattr__(self, "required_orientations", required_dir)
        object.__setattr__(self, "forbidden_orientations", forbidden_dir)
        object.__setattr__(
            self,
            "knowledge_id",
            canonical_fingerprint(
                {
                    "required_adjacencies": required_adj,
                    "forbidden_adjacencies": forbidden_adj,
                    "required_orientations": required_dir,
                    "forbidden_orientations": forbidden_dir,
                }
            ),
        )


class DiscoveryResourcePolicy(StrictModule, NonTrainableState):
    maximum_ci_tests: int = eqx.field(static=True)
    maximum_conditioning_depth: int = eqx.field(static=True)
    maximum_extensions: int = eqx.field(static=True)
    maximum_score_candidates: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_ci_tests: int = 100_000,
        maximum_conditioning_depth: int = 4,
        maximum_extensions: int = 65_536,
        maximum_score_candidates: int = 100_000,
    ) -> None:
        values = tuple(
            int(value)
            for value in (
                maximum_ci_tests,
                maximum_conditioning_depth,
                maximum_extensions,
                maximum_score_candidates,
            )
        )
        if values[0] < 1 or values[1] < 0 or values[2] < 1 or values[3] < 1:
            raise ValueError("Discovery resource limits must be positive/non-negative.")
        object.__setattr__(self, "maximum_ci_tests", values[0])
        object.__setattr__(self, "maximum_conditioning_depth", values[1])
        object.__setattr__(self, "maximum_extensions", values[2])
        object.__setattr__(self, "maximum_score_candidates", values[3])
        object.__setattr__(
            self,
            "policy_id",
            canonical_fingerprint(
                {
                    "maximum_ci_tests": values[0],
                    "maximum_conditioning_depth": values[1],
                    "maximum_extensions": values[2],
                    "maximum_score_candidates": values[3],
                }
            ),
        )


class SeparationEvidence(StrictModule, NonTrainableState):
    left: str = eqx.field(static=True)
    right: str = eqx.field(static=True)
    conditioned: tuple[str, ...] = eqx.field(static=True)
    ci_result_id: str = eqx.field(static=True)


class GraphFalsificationResult(StrictModule, NonTrainableState):
    status: GraphFalsificationStatus = eqx.field(static=True)
    graph_id: str = eqx.field(static=True)
    data_id: str = eqx.field(static=True)
    ci_result_ids: tuple[str, ...] = eqx.field(static=True)
    tested_implications: int = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


class OrientationEvidence(StrictModule, NonTrainableState):
    source: str = eqx.field(static=True)
    target: str = eqx.field(static=True)
    rule: str = eqx.field(static=True)
    premises: tuple[str, ...] = eqx.field(static=True)


class DiscoveryResult(StrictModule, NonTrainableState):
    status: DiscoveryStatus = eqx.field(static=True)
    graph: CausalCPDAG | CausalPDAG | CausalPAG = eqx.field(static=True)
    separation_evidence: tuple[SeparationEvidence, ...] = eqx.field(static=True)
    orientation_evidence: tuple[OrientationEvidence, ...] = eqx.field(static=True)
    ci_tests: int = eqx.field(static=True)
    data_id: str = eqx.field(static=True)
    test_id: str = eqx.field(static=True)
    knowledge_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    @property
    def successful(self) -> bool:
        return self.status is DiscoveryStatus.SUCCESS


class PCStablePlan(StrictModule):
    ci_test: AbstractConditionalIndependenceTest
    knowledge: DiscoveryBackgroundKnowledge = eqx.field(static=True)
    resources: DiscoveryResourcePolicy = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        ci_test: AbstractConditionalIndependenceTest,
        knowledge: DiscoveryBackgroundKnowledge,
        resources: DiscoveryResourcePolicy | None = None,
    ) -> None:
        canonical_resources = (
            DiscoveryResourcePolicy() if resources is None else resources
        )
        object.__setattr__(self, "ci_test", ci_test)
        object.__setattr__(self, "knowledge", knowledge)
        object.__setattr__(self, "resources", canonical_resources)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "pc_stable",
                    "test_id": ci_test.test_id,
                    "knowledge_id": knowledge.knowledge_id,
                    "policy_id": canonical_resources.policy_id,
                }
            ),
        )


class ConservativeFCIPlan(StrictModule):
    ci_test: AbstractConditionalIndependenceTest
    knowledge: DiscoveryBackgroundKnowledge = eqx.field(static=True)
    resources: DiscoveryResourcePolicy = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        ci_test: AbstractConditionalIndependenceTest,
        knowledge: DiscoveryBackgroundKnowledge,
        resources: DiscoveryResourcePolicy | None = None,
    ) -> None:
        canonical_resources = (
            DiscoveryResourcePolicy() if resources is None else resources
        )
        object.__setattr__(self, "ci_test", ci_test)
        object.__setattr__(self, "knowledge", knowledge)
        object.__setattr__(self, "resources", canonical_resources)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "conservative_fci",
                    "test_id": ci_test.test_id,
                    "knowledge_id": knowledge.knowledge_id,
                    "policy_id": canonical_resources.policy_id,
                }
            ),
        )


class GESPlan(StrictModule, NonTrainableState):
    penalty_discount: float = eqx.field(static=True)
    minimum_improvement: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    resources: DiscoveryResourcePolicy = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        penalty_discount: float = 1.0,
        minimum_improvement: float = 1e-10,
        maximum_steps: int = 100,
        resources: DiscoveryResourcePolicy | None = None,
    ) -> None:
        canonical_resources = (
            DiscoveryResourcePolicy() if resources is None else resources
        )
        if penalty_discount <= 0 or minimum_improvement < 0 or int(maximum_steps) < 1:
            raise ValueError("GES penalties/steps are invalid.")
        object.__setattr__(self, "penalty_discount", float(penalty_discount))
        object.__setattr__(self, "minimum_improvement", float(minimum_improvement))
        object.__setattr__(self, "maximum_steps", int(maximum_steps))
        object.__setattr__(self, "resources", canonical_resources)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "ges",
                    "penalty_discount": float(penalty_discount),
                    "minimum_improvement": float(minimum_improvement),
                    "maximum_steps": int(maximum_steps),
                    "policy_id": canonical_resources.policy_id,
                }
            ),
        )


def discover_pc_stable(
    dataset: CausalDataset,
    plan: PCStablePlan,
    *,
    sample_indices: Array | None = None,
    key: PRNGKeyArray | None = None,
) -> DiscoveryResult:
    adjacency, separators, evidence, tests, status, reason = _learn_skeleton(
        dataset,
        plan.ci_test,
        plan.knowledge,
        plan.resources,
        sample_indices=sample_indices,
        key=key,
        exhaustive_candidates=False,
    )
    directed, undirected, orientations, conflict = _orient_pc(
        dataset.schema,
        adjacency,
        separators,
        plan.knowledge,
    )
    pdag = CausalPDAG(
        schema=dataset.schema,
        directed_edges=directed,
        undirected_edges=undirected,
    )
    if status is DiscoveryStatus.SUCCESS and not conflict:
        if 2 ** len(undirected) > plan.resources.maximum_extensions:
            status = DiscoveryStatus.RESOURCE_EXHAUSTED
            reason = "CPDAG verification exceeds maximum_extensions."
            graph: CausalCPDAG | CausalPDAG = pdag
        else:
            completed = verified_cpdag(
                pdag,
                maximum_extensions=plan.resources.maximum_extensions,
            )
            if completed is None:
                status = DiscoveryStatus.AMBIGUOUS
                reason = "Orientation closure did not produce a completed PDAG."
                graph = pdag
            else:
                graph = completed
    else:
        if conflict and status is DiscoveryStatus.SUCCESS:
            status = DiscoveryStatus.AMBIGUOUS
            reason = "Finite-sample orientation evidence is inconsistent."
        graph = pdag
    return _discovery_result(
        status=status,
        graph=graph,
        evidence=evidence,
        orientations=orientations,
        tests=tests,
        dataset=dataset,
        test=plan.ci_test,
        knowledge=plan.knowledge,
        resources=plan.resources,
        reason=reason,
    )


def discover_conservative_fci(
    dataset: CausalDataset,
    plan: ConservativeFCIPlan,
    *,
    sample_indices: Array | None = None,
    key: PRNGKeyArray | None = None,
) -> DiscoveryResult:
    adjacency, separators, evidence, tests, status, reason = _learn_skeleton(
        dataset,
        plan.ci_test,
        plan.knowledge,
        plan.resources,
        sample_indices=sample_indices,
        key=key,
        exhaustive_candidates=True,
    )
    endpoints: dict[tuple[str, str], list[EndpointMark]] = {
        pair: [EndpointMark.CIRCLE, EndpointMark.CIRCLE] for pair in sorted(adjacency)
    }
    orientations: list[OrientationEvidence] = []
    names = dataset.schema.names
    for middle in names:
        neighbors = [
            node for node in names if _pair(dataset.schema, node, middle) in adjacency
        ]
        for left, right in itertools.combinations(neighbors, 2):
            if _pair(dataset.schema, left, right) in adjacency:
                continue
            separator = separators.get(_pair(dataset.schema, left, right), ())
            if middle not in separator:
                _set_endpoint_arrow(dataset.schema, endpoints, left, middle)
                _set_endpoint_arrow(dataset.schema, endpoints, right, middle)
                orientations.extend(
                    (
                        OrientationEvidence(
                            left, middle, "unshielded_collider", (right,)
                        ),
                        OrientationEvidence(
                            right, middle, "unshielded_collider", (left,)
                        ),
                    )
                )
    endpoint_edges = tuple(
        (left, marks[0], right, marks[1])
        for (left, right), marks in sorted(
            endpoints.items(),
            key=lambda item: (
                dataset.schema.index(item[0][0]),
                dataset.schema.index(item[0][1]),
            ),
        )
    )
    provenance = canonical_fingerprint(
        {
            "algorithm": "conservative_fci",
            "data_id": dataset.data_id,
            "test_id": plan.ci_test.test_id,
            "evidence": [item.ci_result_id for item in evidence],
        }
    )
    graph = CausalPAG(
        schema=dataset.schema,
        endpoint_edges=endpoint_edges,
        provenance_id=provenance,
    )
    return _discovery_result(
        status=status,
        graph=graph,
        evidence=evidence,
        orientations=orientations,
        tests=tests,
        dataset=dataset,
        test=plan.ci_test,
        knowledge=plan.knowledge,
        resources=plan.resources,
        reason=(
            "Conservative FCI returned a sound partial ancestral graph; "
            "unresolved endpoints remain circles."
            if status is DiscoveryStatus.SUCCESS
            else reason
        ),
    )


def falsify_dag_local_markov(
    dataset: CausalDataset,
    graph: CausalDAG,
    ci_test: AbstractConditionalIndependenceTest,
    *,
    maximum_tests: int = 10_000,
    sample_indices: Array | None = None,
    key: PRNGKeyArray | None = None,
) -> GraphFalsificationResult:
    """Challenge DAG local-Markov implications without validating the graph."""
    if graph.schema.schema_id != dataset.schema.schema_id:
        raise ValueError("Graph and dataset schemas must match.")
    if any(
        variable.observability is not VariableObservability.OBSERVED
        for variable in graph.schema.variables
    ):
        raise ValueError("Local-Markov falsification requires observed graph variables.")
    ci_ids: list[str] = []
    status = GraphFalsificationStatus.NOT_REJECTED
    reason = "Tested local-Markov implications were not rejected."
    tested = 0
    for node in graph.schema.names:
        parents = set(graph.parents(node))
        descendants_of_node = _directed_descendants(graph, node)
        candidates = [
            other
            for other in graph.schema.names
            if other not in parents | descendants_of_node | {node}
        ]
        for other in candidates:
            tested += 1
            if tested > int(maximum_tests):
                status = GraphFalsificationStatus.INCONCLUSIVE
                reason = "Local-Markov falsification exceeded maximum_tests."
                return _graph_falsification_result(
                    status,
                    graph,
                    dataset,
                    ci_ids,
                    tested,
                    reason,
                )
            query_key = None if key is None else jax.random.fold_in(key, tested)
            result = ci_test.test(
                dataset,
                node,
                other,
                tuple(name for name in graph.schema.names if name in parents),
                sample_indices=sample_indices,
                key=query_key,
            )
            ci_ids.append(result.result_id)
            if not result.successful:
                status = GraphFalsificationStatus.INCONCLUSIVE
                reason = f"CI implication test failed with {result.status.name}."
                return _graph_falsification_result(
                    status,
                    graph,
                    dataset,
                    ci_ids,
                    tested,
                    reason,
                )
            if not result.independent:
                status = GraphFalsificationStatus.REJECTED
                reason = (
                    f"Observed dependence of {node!r} and {other!r} given its "
                    "parents rejects a DAG local-Markov implication."
                )
                return _graph_falsification_result(
                    status,
                    graph,
                    dataset,
                    ci_ids,
                    tested,
                    reason,
                )
    if tested == 0:
        status = GraphFalsificationStatus.NOT_APPLICABLE
        reason = "The DAG has no nontrivial local-Markov implication to test."
    return _graph_falsification_result(
        status,
        graph,
        dataset,
        ci_ids,
        tested,
        reason,
    )


def discover_ges(
    dataset: CausalDataset,
    plan: GESPlan,
    *,
    sample_indices: Array | None = None,
) -> DiscoveryResult:
    _validate_discovery_dataset(dataset)
    for variable in dataset.schema.variables:
        if variable.scale is not VariableScale.CONTINUOUS:
            raise ValueError("Gaussian BIC GES requires continuous variables.")
        active = _active_indices(dataset, sample_indices)
        if not np.all(np.asarray(dataset.observed_mask(variable.name))[active]):
            raise ValueError("GES requires complete observations for every variable.")
    names = dataset.schema.names
    if len(names) > 8:
        raise ValueError("Exact equivalence-class GES is bounded to eight variables.")
    indices = _active_indices(dataset, sample_indices)
    matrix = np.column_stack([np.asarray(dataset.value(name))[indices] for name in names])
    if not np.all(np.isfinite(matrix)):
        raise ValueError("GES requires finite complete data.")
    current = CausalCPDAG(
        schema=dataset.schema,
        directed_edges=(),
        undirected_edges=(),
        maximum_extensions=plan.resources.maximum_extensions,
    )
    current_score = _gaussian_bic_score(
        matrix, current.extensions[0], plan.penalty_discount
    )
    candidates_evaluated = 0
    extension_resource_limited = False
    orientations: list[OrientationEvidence] = []
    for phase in ("forward", "backward"):
        for step in range(plan.maximum_steps):
            best: tuple[float, CausalCPDAG, tuple[str, str]] | None = None
            candidate_classes: dict[str, tuple[float, CausalCPDAG, tuple[str, str]]] = {}
            for extension in current.extensions:
                skeleton = {
                    _pair(dataset.schema, source, target)
                    for source, target in extension.directed_edges
                }
                if phase == "forward":
                    raw_candidates = (
                        (source, target)
                        for source in names
                        for target in names
                        if source != target
                        and _pair(dataset.schema, source, target) not in skeleton
                    )
                else:
                    raw_candidates = iter(extension.directed_edges)
                for source, target in raw_candidates:
                    candidates_evaluated += 1
                    if candidates_evaluated > plan.resources.maximum_score_candidates:
                        return _discovery_result(
                            status=DiscoveryStatus.RESOURCE_EXHAUSTED,
                            graph=current,
                            evidence=(),
                            orientations=orientations,
                            tests=0,
                            dataset=dataset,
                            test=None,
                            knowledge=None,
                            resources=plan.resources,
                            reason="GES exceeded maximum_score_candidates.",
                        )
                    edges = list(extension.directed_edges)
                    if phase == "forward":
                        edges.append((source, target))
                    else:
                        edges.remove((source, target))
                    if not _acyclic(names, edges):
                        continue
                    if 2 ** len(edges) > plan.resources.maximum_extensions:
                        extension_resource_limited = True
                        continue
                    dag = CausalDAG(schema=dataset.schema, directed_edges=edges)
                    cpdag = complete_dag_equivalence_class(
                        dag,
                        maximum_extensions=plan.resources.maximum_extensions,
                    )
                    if cpdag.graph_id in candidate_classes:
                        continue
                    score = _gaussian_bic_score(matrix, dag, plan.penalty_discount)
                    candidate_classes[cpdag.graph_id] = (score, cpdag, (source, target))
            for candidate in candidate_classes.values():
                if best is None or candidate[0] > best[0]:
                    best = candidate
            if best is None or best[0] <= current_score + plan.minimum_improvement:
                break
            current_score, current, operation = best
            orientations.append(
                OrientationEvidence(
                    operation[0],
                    operation[1],
                    f"ges_{phase}_{step}",
                    (f"score={current_score:.17g}",),
                )
            )
    terminal_status = (
        DiscoveryStatus.RESOURCE_EXHAUSTED
        if extension_resource_limited
        else DiscoveryStatus.SUCCESS
    )
    return _discovery_result(
        status=terminal_status,
        graph=current,
        evidence=(),
        orientations=orientations,
        tests=0,
        dataset=dataset,
        test=None,
        knowledge=None,
        resources=plan.resources,
        reason=(
            "Some equivalence-class candidates exceeded maximum_extensions."
            if extension_resource_limited
            else "Exact bounded equivalence-class greedy search completed."
        ),
    )


def _directed_descendants(graph: CausalDAG, source: str) -> set[str]:
    children = {name: set() for name in graph.schema.names}
    for parent, child in graph.directed_edges:
        children[parent].add(child)
    result: set[str] = set()
    queue = deque(children[source])
    while queue:
        node = queue.popleft()
        if node in result:
            continue
        result.add(node)
        queue.extend(children[node] - result)
    return result


def _graph_falsification_result(
    status: GraphFalsificationStatus,
    graph: CausalDAG,
    dataset: CausalDataset,
    ci_result_ids: Sequence[str],
    tested_implications: int,
    reason: str,
) -> GraphFalsificationResult:
    identifiers = tuple(ci_result_ids)
    return GraphFalsificationResult(
        status=status,
        graph_id=graph.graph_id,
        data_id=dataset.data_id,
        ci_result_ids=identifiers,
        tested_implications=int(tested_implications),
        reason=reason,
        result_id=canonical_fingerprint(
            {
                "status": status.value,
                "graph_id": graph.graph_id,
                "data_id": dataset.data_id,
                "ci_result_ids": identifiers,
                "tested_implications": int(tested_implications),
                "reason": reason,
            }
        ),
    )


def _learn_skeleton(
    dataset: CausalDataset,
    test: AbstractConditionalIndependenceTest,
    knowledge: DiscoveryBackgroundKnowledge,
    resources: DiscoveryResourcePolicy,
    *,
    sample_indices: Array | None,
    key: PRNGKeyArray | None,
    exhaustive_candidates: bool,
) -> tuple[
    set[tuple[str, str]],
    dict[tuple[str, str], tuple[str, ...]],
    tuple[SeparationEvidence, ...],
    int,
    DiscoveryStatus,
    str,
]:
    _validate_discovery_dataset(dataset)
    names = dataset.schema.names
    adjacency = {
        _pair(dataset.schema, left, right)
        for left, right in itertools.combinations(names, 2)
        if _pair(dataset.schema, left, right) not in knowledge.forbidden_adjacencies
    }
    adjacency.update(knowledge.required_adjacencies)
    separators: dict[tuple[str, str], tuple[str, ...]] = {}
    evidence: list[SeparationEvidence] = []
    tests = 0
    status = DiscoveryStatus.SUCCESS
    reason = "Constraint-based skeleton search completed."
    for depth in range(resources.maximum_conditioning_depth + 1):
        snapshot = set(adjacency)
        removals: dict[tuple[str, str], tuple[str, ...]] = {}
        for left, right in sorted(
            snapshot,
            key=lambda pair: (
                dataset.schema.index(pair[0]),
                dataset.schema.index(pair[1]),
            ),
        ):
            if (left, right) in knowledge.required_adjacencies:
                continue
            if exhaustive_candidates:
                pools = [tuple(node for node in names if node not in {left, right})]
            else:
                left_neighbors = tuple(
                    node
                    for node in names
                    if node != right and _pair(dataset.schema, left, node) in snapshot
                )
                right_neighbors = tuple(
                    node
                    for node in names
                    if node != left and _pair(dataset.schema, right, node) in snapshot
                )
                pools = [left_neighbors, right_neighbors]
            tested_sets: set[tuple[str, ...]] = set()
            separated = False
            for pool in pools:
                if len(pool) < depth:
                    continue
                for conditioned in itertools.combinations(pool, depth):
                    canonical_conditioned = tuple(
                        node for node in names if node in set(conditioned)
                    )
                    if canonical_conditioned in tested_sets:
                        continue
                    tested_sets.add(canonical_conditioned)
                    tests += 1
                    if tests > resources.maximum_ci_tests:
                        return (
                            adjacency,
                            separators,
                            tuple(evidence),
                            tests,
                            DiscoveryStatus.RESOURCE_EXHAUSTED,
                            "Skeleton search exceeded maximum_ci_tests.",
                        )
                    query_key = None if key is None else jax.random.fold_in(key, tests)
                    result = test.test(
                        dataset,
                        left,
                        right,
                        canonical_conditioned,
                        sample_indices=sample_indices,
                        key=query_key,
                    )
                    if not result.successful:
                        return (
                            adjacency,
                            separators,
                            tuple(evidence),
                            tests,
                            DiscoveryStatus.CI_FAILED,
                            f"CI test failed with status {result.status.name}.",
                        )
                    if result.independent:
                        removals[(left, right)] = canonical_conditioned
                        evidence.append(
                            SeparationEvidence(
                                left=left,
                                right=right,
                                conditioned=canonical_conditioned,
                                ci_result_id=result.result_id,
                            )
                        )
                        separated = True
                        break
                if separated:
                    break
        adjacency.difference_update(removals)
        separators.update(removals)
        if not removals and all(
            sum(node in edge for edge in adjacency) <= depth for node in names
        ):
            break
    return adjacency, separators, tuple(evidence), tests, status, reason


def _orient_pc(
    schema: CausalSchema,
    adjacency: set[tuple[str, str]],
    separators: dict[tuple[str, str], tuple[str, ...]],
    knowledge: DiscoveryBackgroundKnowledge,
) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
    tuple[OrientationEvidence, ...],
    bool,
]:
    directed = set(knowledge.required_orientations)
    undirected = set(adjacency)
    evidence: list[OrientationEvidence] = [
        OrientationEvidence(source, target, "background_knowledge", ())
        for source, target in knowledge.required_orientations
    ]
    conflict = False
    for source, target in tuple(directed):
        undirected.discard(_pair(schema, source, target))
    for left, right in tuple(undirected):
        left_tier = schema.variable(left).temporal_tier
        right_tier = schema.variable(right).temporal_tier
        if left_tier is None or right_tier is None or left_tier == right_tier:
            continue
        source, target = (left, right) if left_tier < right_tier else (right, left)
        conflict |= not _orient_edge(schema, directed, undirected, source, target)
        evidence.append(OrientationEvidence(source, target, "temporal_tier", ()))
    names = schema.names
    for middle in names:
        neighbors = [node for node in names if _pair(schema, node, middle) in adjacency]
        for left, right in itertools.combinations(neighbors, 2):
            if _pair(schema, left, right) in adjacency:
                continue
            if middle not in separators.get(_pair(schema, left, right), ()):
                for source in (left, right):
                    conflict |= not _orient_edge(
                        schema, directed, undirected, source, middle
                    )
                    evidence.append(
                        OrientationEvidence(
                            source,
                            middle,
                            "unshielded_collider",
                            (left, middle, right),
                        )
                    )
    changed = True
    while changed and not conflict:
        changed = False
        for left, right in tuple(undirected):
            for source, target in ((left, right), (right, left)):
                # Meek R1: a -> source - target and a not adjacent target.
                witnesses = [
                    parent
                    for parent, child in directed
                    if child == source and _pair(schema, parent, target) not in adjacency
                ]
                if witnesses:
                    conflict |= not _orient_edge(
                        schema, directed, undirected, source, target
                    )
                    evidence.append(
                        OrientationEvidence(source, target, "meek_r1", tuple(witnesses))
                    )
                    changed = True
                    break
                # Meek R2: source -> middle -> target.
                middle_nodes = {
                    child for parent, child in directed if parent == source
                } & {parent for parent, child in directed if child == target}
                if middle_nodes:
                    conflict |= not _orient_edge(
                        schema, directed, undirected, source, target
                    )
                    evidence.append(
                        OrientationEvidence(
                            source,
                            target,
                            "meek_r2",
                            tuple(sorted(middle_nodes)),
                        )
                    )
                    changed = True
                    break
                # Meek R3: source - c -> target and source - d -> target,
                # with c and d nonadjacent, implies source -> target.
                candidate_parents = [
                    node
                    for node in names
                    if (node, target) in directed
                    and _pair(schema, source, node) in undirected
                ]
                r3_witness: tuple[str, str] | None = None
                for first, second in itertools.combinations(candidate_parents, 2):
                    if _pair(schema, first, second) not in adjacency:
                        r3_witness = (first, second)
                        break
                if r3_witness is not None:
                    conflict |= not _orient_edge(
                        schema, directed, undirected, source, target
                    )
                    evidence.append(
                        OrientationEvidence(
                            source,
                            target,
                            "meek_r3",
                            r3_witness,
                        )
                    )
                    changed = True
                    break
    if set(directed) & set(knowledge.forbidden_orientations):
        conflict = True
    index = {name: position for position, name in enumerate(names)}
    return (
        tuple(sorted(directed, key=lambda edge: (index[edge[0]], index[edge[1]]))),
        tuple(sorted(undirected, key=lambda edge: (index[edge[0]], index[edge[1]]))),
        tuple(evidence),
        conflict,
    )


def _orient_edge(
    schema: CausalSchema,
    directed: set[tuple[str, str]],
    undirected: set[tuple[str, str]],
    source: str,
    target: str,
) -> bool:
    if (target, source) in directed:
        return False
    pair = _pair(schema, source, target)
    if pair not in undirected and (source, target) not in directed:
        return False
    undirected.discard(pair)
    directed.add((source, target))
    return True


def _set_endpoint_arrow(
    schema: CausalSchema,
    endpoints: dict[tuple[str, str], list[EndpointMark]],
    source: str,
    target: str,
) -> None:
    pair = _pair(schema, source, target)
    position = 1 if pair[1] == target else 0
    endpoints[pair][position] = EndpointMark.ARROW


def _ci_matrix(
    dataset: CausalDataset,
    names: Sequence[str],
    sample_indices: Array | None,
) -> tuple[np.ndarray, int]:
    indices = _active_indices(dataset, sample_indices)
    observed = np.ones((indices.size,), dtype=bool)
    columns = []
    for name in names:
        observed &= np.asarray(dataset.observed_mask(name))[indices]
        columns.append(np.asarray(dataset.value(name))[indices])
    if not np.all(observed):
        raise ValueError(
            "CI tests require complete observations for all query variables."
        )
    matrix = np.column_stack(columns)
    return matrix, int(matrix.shape[0])


def _active_indices(dataset: CausalDataset, sample_indices: Array | None) -> np.ndarray:
    if sample_indices is None:
        candidates = np.arange(dataset.n_samples, dtype=np.int32)
    else:
        candidates = np.asarray(sample_indices, dtype=np.int32)
        if (
            candidates.ndim != 1
            or np.any(candidates < 0)
            or np.any(candidates >= dataset.n_samples)
        ):
            raise ValueError("sample_indices are invalid for this dataset.")
    return candidates[np.asarray(dataset.sample_mask)[candidates]]


def _finite_cardinality(schema: CausalSchema, name: str) -> int:
    cardinality = schema.variable(name).cardinality
    if cardinality is None:
        raise ValueError(f"Variable {name!r} is not finite categorical.")
    return cardinality


def _validate_ci_variables(
    schema: CausalSchema,
    names: Sequence[str],
    *,
    continuous: bool,
) -> None:
    if len(set(names)) != len(names):
        raise ValueError("CI query variables must be distinct.")
    for name in names:
        variable = schema.variable(name)
        if (
            variable.observability is not VariableObservability.OBSERVED
            or variable.event_shape
        ):
            raise ValueError("CI tests require observed scalar variables.")
        finite = variable.scale in {
            VariableScale.BINARY,
            VariableScale.CATEGORICAL,
            VariableScale.ORDINAL,
        }
        if continuous == finite:
            expected = "continuous" if continuous else "finite categorical"
            raise ValueError(f"This CI test requires {expected} variables.")


def _validate_discovery_dataset(dataset: CausalDataset) -> None:
    for variable in dataset.schema.variables:
        if variable.observability is not VariableObservability.OBSERVED:
            raise ValueError("Discovery schemas may contain observed variables only.")
        if variable.event_shape:
            raise ValueError("Discovery currently requires scalar variables.")


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie in (0, 1).")


def _ci_result(
    test: AbstractConditionalIndependenceTest,
    status: CIStatus,
    left: str,
    right: str,
    conditioned: Sequence[str],
    *,
    statistic: float,
    p_value: float,
    effective_samples: int,
    degrees_of_freedom: int,
) -> ConditionalIndependenceResult:
    independent = status is CIStatus.SUCCESS and p_value > test.alpha
    canonical_conditioned = tuple(conditioned)
    payload = {
        "status": int(status),
        "independent": independent,
        "statistic": statistic,
        "p_value": p_value,
        "effective_samples": effective_samples,
        "degrees_of_freedom": degrees_of_freedom,
        "left": left,
        "right": right,
        "conditioned": canonical_conditioned,
        "test_id": test.test_id,
    }
    return ConditionalIndependenceResult(
        status=status,
        independent=independent,
        statistic=float(statistic),
        p_value=float(p_value),
        effective_samples=int(effective_samples),
        degrees_of_freedom=int(degrees_of_freedom),
        left=left,
        right=right,
        conditioned=canonical_conditioned,
        test_id=test.test_id,
        result_id=canonical_fingerprint(payload),
    )


def _rbf_kernel(values: np.ndarray, bandwidth: float) -> np.ndarray:
    squared = np.sum((values[:, None, :] - values[None, :, :]) ** 2, axis=-1)
    return np.exp(-0.5 * squared / (bandwidth * bandwidth))


def _pair(schema: CausalSchema, left: str, right: str) -> tuple[str, str]:
    return (left, right) if schema.index(left) < schema.index(right) else (right, left)


def _canonical_pairs(
    schema: CausalSchema,
    edges: Iterable[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    canonical = {_pair(schema, left, right) for left, right in edges}
    if any(left == right for left, right in canonical):
        raise ValueError("Background knowledge self-edges are invalid.")
    return tuple(
        sorted(canonical, key=lambda edge: (schema.index(edge[0]), schema.index(edge[1])))
    )


def _canonical_directed(
    schema: CausalSchema,
    edges: Iterable[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    canonical = set(edges)
    for source, target in canonical:
        schema.index(source)
        schema.index(target)
        if source == target:
            raise ValueError("Background knowledge self-orientations are invalid.")
    return tuple(
        sorted(canonical, key=lambda edge: (schema.index(edge[0]), schema.index(edge[1])))
    )


def _acyclic(names: Sequence[str], edges: Sequence[tuple[str, str]]) -> bool:
    indegree = {name: 0 for name in names}
    children = {name: [] for name in names}
    for source, target in edges:
        indegree[target] += 1
        children[source].append(target)
    queue = deque(name for name in names if indegree[name] == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    return visited == len(names)


def _gaussian_bic_score(
    matrix: np.ndarray,
    graph: CausalDAG,
    penalty_discount: float,
) -> float:
    n_samples = matrix.shape[0]
    score = 0.0
    for target_index, target in enumerate(graph.schema.names):
        parent_indices = [graph.schema.index(parent) for parent in graph.parents(target)]
        design = (
            np.ones((n_samples, 1))
            if not parent_indices
            else np.column_stack((np.ones((n_samples,)), matrix[:, parent_indices]))
        )
        outcome = matrix[:, target_index]
        residual = outcome - design @ np.linalg.lstsq(design, outcome, rcond=None)[0]
        variance = float(np.dot(residual, residual) / n_samples)
        if not np.isfinite(variance) or variance <= 0:
            return -np.inf
        parameters = design.shape[1] + 1
        score += -0.5 * (
            n_samples * (np.log(2.0 * np.pi * variance) + 1.0)
            + penalty_discount * parameters * np.log(n_samples)
        )
    return score


def _discovery_result(
    *,
    status: DiscoveryStatus,
    graph: CausalCPDAG | CausalPDAG | CausalPAG,
    evidence: Sequence[SeparationEvidence],
    orientations: Sequence[OrientationEvidence],
    tests: int,
    dataset: CausalDataset,
    test: AbstractConditionalIndependenceTest | None,
    knowledge: DiscoveryBackgroundKnowledge | None,
    resources: DiscoveryResourcePolicy,
    reason: str,
) -> DiscoveryResult:
    test_id = "score-based" if test is None else test.test_id
    knowledge_id = "none" if knowledge is None else knowledge.knowledge_id
    payload = {
        "status": status.value,
        "graph_id": graph.graph_id,
        "separation_evidence": [item.ci_result_id for item in evidence],
        "orientation_evidence": [
            (item.source, item.target, item.rule, item.premises) for item in orientations
        ],
        "ci_tests": tests,
        "data_id": dataset.data_id,
        "test_id": test_id,
        "knowledge_id": knowledge_id,
        "policy_id": resources.policy_id,
        "reason": reason,
    }
    return DiscoveryResult(
        status=status,
        graph=graph,
        separation_evidence=tuple(evidence),
        orientation_evidence=tuple(orientations),
        ci_tests=tests,
        data_id=dataset.data_id,
        test_id=test_id,
        knowledge_id=knowledge_id,
        policy_id=resources.policy_id,
        reason=reason,
        result_id=canonical_fingerprint(payload),
    )


__all__ = [
    "AbstractConditionalIndependenceTest",
    "CIStatus",
    "ConditionalIndependenceResult",
    "ConservativeFCIPlan",
    "DiscoveryBackgroundKnowledge",
    "DiscoveryResourcePolicy",
    "DiscoveryResult",
    "DiscoveryStatus",
    "FisherZTest",
    "GESPlan",
    "GSquareTest",
    "GraphFalsificationResult",
    "GraphFalsificationStatus",
    "KernelConditionalIndependenceTest",
    "OrientationEvidence",
    "PCStablePlan",
    "SeparationEvidence",
    "discover_conservative_fci",
    "discover_ges",
    "discover_pc_stable",
    "falsify_dag_local_markov",
]
