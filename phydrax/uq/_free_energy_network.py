#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Covariance-aware free-energy edges, cycles and network inference."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg._dense_pseudoinverse import (
    apply_pseudoinverse,
    factor_pseudoinverse,
    materialize_pseudoinverse,
)
from ..linalg._policies import RankPolicy
from ._free_energy import FreeEnergyResult, FreeEnergyStatus


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty canonical string.")
    return value


def _real_array(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if jnp.iscomplexobj(array):
        raise TypeError(f"{name} must be real-valued.")
    if not jnp.issubdtype(array.dtype, jnp.floating):
        array = array.astype(float)
    return array


def _covariance_tolerance(value: np.ndarray, /) -> float:
    scale = max(float(np.max(np.abs(value), initial=0.0)), 1.0)
    return 256.0 * np.finfo(value.dtype).eps * scale


class FreeEnergyEdgeObservation(StrictModule, NonTrainableState):
    """One directed free-energy difference and its joint-influence coordinates."""

    value: Array
    variance: Array
    influence_values: Array
    independent_variance: Array
    source_state_id: str = eqx.field(static=True)
    destination_state_id: str = eqx.field(static=True)
    observation_id: str = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)
    selection_id: str = eqx.field(static=True)
    influence_basis_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        value: ArrayLike,
        variance: ArrayLike,
        /,
        *,
        source_state_id: str,
        destination_state_id: str,
        observation_id: str,
        dataset_id: str,
        selection_id: str,
        influence_values: ArrayLike | None = None,
        influence_basis_id: str | None = None,
    ):
        estimate = _real_array(value, "value").reshape(())
        variance_ = _real_array(variance, "variance").reshape(())
        source = _identifier(source_state_id, "source_state_id")
        destination = _identifier(destination_state_id, "destination_state_id")
        if source == destination:
            raise ValueError("An edge must connect two distinct state identities.")
        if not bool(jnp.isfinite(estimate)):
            raise ValueError("Edge values must be finite.")
        if not bool(jnp.isfinite(variance_) & (variance_ >= 0.0)):
            raise ValueError("Edge variance must be finite and non-negative.")
        if influence_values is None and influence_basis_id is not None:
            raise ValueError("influence_basis_id requires explicit influence_values.")
        if influence_values is None:
            influence = jnp.zeros((0,), dtype=estimate.dtype)
            basis = None
        else:
            influence = _real_array(influence_values, "influence_values").reshape((-1,))
            if not bool(jnp.all(jnp.isfinite(influence))):
                raise ValueError("Edge influence values must be finite.")
            basis = _identifier(influence_basis_id, "influence_basis_id")
        represented = jnp.sum(influence**2)
        scale = max(float(np.asarray(variance_)), 1.0)
        tolerance = 256.0 * np.finfo(np.asarray(variance_).dtype).eps * scale
        if float(np.asarray(represented - variance_)) > tolerance:
            raise ValueError(
                "Edge influence variance cannot exceed the declared variance."
            )
        independent = jnp.maximum(variance_ - represented, 0.0)
        self.value = estimate
        self.variance = variance_
        self.influence_values = influence
        self.independent_variance = independent
        self.source_state_id = source
        self.destination_state_id = destination
        self.observation_id = _identifier(observation_id, "observation_id")
        self.dataset_id = _identifier(dataset_id, "dataset_id")
        self.selection_id = _identifier(selection_id, "selection_id")
        self.influence_basis_id = basis

    @classmethod
    def from_result(
        cls,
        result: FreeEnergyResult,
        source_state_id: str,
        destination_state_id: str,
        /,
        *,
        observation_id: str | None = None,
    ) -> "FreeEnergyEdgeObservation":
        """Project one successful joint result onto a directed network edge."""

        if not isinstance(result, FreeEnergyResult):
            raise TypeError("result must be FreeEnergyResult.")
        if not bool(result.successful):
            raise ValueError(
                "Only statistically and numerically successful results form edges."
            )
        source = _identifier(source_state_id, "source_state_id")
        destination = _identifier(destination_state_id, "destination_state_id")
        if source not in result.state_ids or destination not in result.state_ids:
            raise ValueError("Edge states must occur in the source result.")
        source_index = result.state_ids.index(source)
        destination_index = result.state_ids.index(destination)
        influence = (
            result.influence_values[destination_index]
            - result.influence_values[source_index]
        )
        variance = result.standard_errors[source_index, destination_index] ** 2
        identity = observation_id or canonical_fingerprint(
            {
                "kind": "free-energy-edge-observation",
                "analysis_id": result.analysis_id,
                "source_state_id": source,
                "destination_state_id": destination,
            }
        )
        return cls(
            result.differences[source_index, destination_index],
            variance,
            source_state_id=source,
            destination_state_id=destination,
            observation_id=identity,
            dataset_id=result.dataset_id,
            selection_id=result.selection_id,
            influence_values=influence,
            influence_basis_id=result.analysis_id,
        )


class FreeEnergyNetworkPlan(StrictModule, NonTrainableState):
    """State topology, fixed gauge and numerical policy for network GLS."""

    state_ids: tuple[str, ...] = eqx.field(static=True)
    reference_state_id: str = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    maximum_cycle_z_score: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        state_ids: Sequence[str],
        /,
        *,
        reference_state_id: str,
        rank_tolerance: float = 1.0e-12,
        maximum_cycle_z_score: float = 5.0,
    ):
        states = tuple(_identifier(value, "state_id") for value in state_ids)
        if len(states) < 2 or len(set(states)) != len(states):
            raise ValueError(
                "Network state_ids must contain at least two unique entries."
            )
        reference = _identifier(reference_state_id, "reference_state_id")
        if reference not in states:
            raise ValueError("reference_state_id must occur in state_ids.")
        tolerance = float(rank_tolerance)
        cycle_threshold = float(maximum_cycle_z_score)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("rank_tolerance must be finite and non-negative.")
        if not math.isfinite(cycle_threshold) or cycle_threshold <= 0.0:
            raise ValueError("maximum_cycle_z_score must be finite and positive.")
        self.state_ids = states
        self.reference_state_id = reference
        self.rank_tolerance = tolerance
        self.maximum_cycle_z_score = cycle_threshold
        self.plan_id = canonical_fingerprint(
            {
                "kind": "free-energy-network-plan",
                "state_ids": states,
                "reference_state_id": reference,
                "rank_tolerance": tolerance.hex(),
                "maximum_cycle_z_score": cycle_threshold.hex(),
                "estimator": "singular-covariance-aware-generalized-least-squares",
            }
        )

    def analyze(
        self,
        observations: Sequence[FreeEnergyEdgeObservation],
        /,
        *,
        joint_covariance: ArrayLike | None = None,
    ) -> "FreeEnergyNetworkResult":
        return analyze_free_energy_network(
            self,
            observations,
            joint_covariance=joint_covariance,
        )


class FreeEnergyNetworkResult(StrictModule, NonTrainableState):
    """Gauge-fixed GLS network with edge and cycle closure evidence."""

    free_energies: Array
    covariance: Array
    differences: Array
    standard_errors: Array
    edge_values: Array
    edge_residuals: Array
    joint_edge_covariance: Array
    cycle_matrix: Array
    cycle_residuals: Array
    cycle_covariance: Array
    cycle_standard_scores: Array
    design_rank: Array
    covariance_rank: Array
    numerical_status: Array
    statistical_status: Array
    plan: FreeEnergyNetworkPlan
    observation_ids: tuple[str, ...] = eqx.field(static=True)
    analysis_id: str = eqx.field(static=True)

    def __init__(
        self,
        free_energies: ArrayLike,
        covariance: ArrayLike,
        edge_values: ArrayLike,
        edge_residuals: ArrayLike,
        joint_edge_covariance: ArrayLike,
        cycle_matrix: ArrayLike,
        cycle_residuals: ArrayLike,
        cycle_covariance: ArrayLike,
        cycle_standard_scores: ArrayLike,
        design_rank: ArrayLike,
        covariance_rank: ArrayLike,
        numerical_status: int | FreeEnergyStatus,
        statistical_status: int | FreeEnergyStatus,
        plan: FreeEnergyNetworkPlan,
        /,
        *,
        observation_ids: Sequence[str],
    ):
        if not isinstance(plan, FreeEnergyNetworkPlan):
            raise TypeError("plan must be FreeEnergyNetworkPlan.")
        free = _real_array(free_energies, "free_energies").reshape((-1,))
        covariance_ = _real_array(covariance, "covariance")
        values = _real_array(edge_values, "edge_values").reshape((-1,))
        residuals = _real_array(edge_residuals, "edge_residuals").reshape((-1,))
        edge_covariance = _real_array(joint_edge_covariance, "joint_edge_covariance")
        cycles = _real_array(cycle_matrix, "cycle_matrix")
        cycle_residual = _real_array(cycle_residuals, "cycle_residuals").reshape((-1,))
        cycle_covariance_ = _real_array(cycle_covariance, "cycle_covariance")
        cycle_scores = _real_array(
            cycle_standard_scores, "cycle_standard_scores"
        ).reshape((-1,))
        state_count = len(plan.state_ids)
        edge_count = values.size
        cycle_count = cycles.shape[0]
        if free.shape != (state_count,) or covariance_.shape != (
            state_count,
            state_count,
        ):
            raise ValueError(
                "Network state estimates and covariance do not match the plan."
            )
        if residuals.shape != values.shape or edge_covariance.shape != (
            edge_count,
            edge_count,
        ):
            raise ValueError("Network edge arrays do not share one edge order.")
        if cycles.shape != (cycle_count, edge_count):
            raise ValueError("cycle_matrix must act on the ordered edge vector.")
        if (
            cycle_residual.shape != (cycle_count,)
            or cycle_covariance_.shape != (cycle_count, cycle_count)
            or cycle_scores.shape != (cycle_count,)
        ):
            raise ValueError(
                "Cycle residuals, scores and covariance do not match cycle_matrix."
            )
        arrays = (
            free,
            covariance_,
            values,
            residuals,
            edge_covariance,
            cycles,
            cycle_residual,
            cycle_covariance_,
            cycle_scores,
        )
        if not all(bool(jnp.all(jnp.isfinite(value))) for value in arrays):
            raise ValueError(
                "Network result arrays must be finite; failures use status bits."
            )
        covariance_ = 0.5 * (covariance_ + covariance_.T)
        edge_covariance = 0.5 * (edge_covariance + edge_covariance.T)
        cycle_covariance_ = 0.5 * (cycle_covariance_ + cycle_covariance_.T)
        reference_index = plan.state_ids.index(plan.reference_state_id)
        tolerance = _covariance_tolerance(np.asarray(covariance_))
        if abs(float(np.asarray(free[reference_index]))) > tolerance:
            raise ValueError("Network free energies must use the plan's fixed gauge.")
        if np.max(np.abs(np.asarray(covariance_)[reference_index])) > tolerance:
            raise ValueError("Network covariance must be zero along the fixed gauge.")
        if float(np.min(np.linalg.eigvalsh(np.asarray(covariance_)))) < -tolerance:
            raise ValueError("Network covariance must be positive semidefinite.")
        if float(np.min(np.linalg.eigvalsh(np.asarray(edge_covariance)))) < -tolerance:
            raise ValueError("Joint edge covariance must be positive semidefinite.")
        if (
            cycle_count
            and float(np.min(np.linalg.eigvalsh(np.asarray(cycle_covariance_))))
            < -tolerance
        ):
            raise ValueError("Cycle covariance must be positive semidefinite.")
        expected_cycle_residual = cycles @ values
        expected_cycle_covariance = cycles @ edge_covariance @ cycles.T
        expected_cycle_scores = jnp.abs(expected_cycle_residual) / jnp.sqrt(
            jnp.maximum(
                jnp.maximum(jnp.diag(expected_cycle_covariance), 0.0),
                jnp.finfo(values.dtype).tiny,
            )
        )
        if not np.allclose(
            np.asarray(cycle_residual),
            np.asarray(expected_cycle_residual),
            atol=tolerance,
            rtol=1.0e-6,
        ) or not np.allclose(
            np.asarray(cycle_covariance_),
            np.asarray(expected_cycle_covariance),
            atol=tolerance,
            rtol=1.0e-6,
        ):
            raise ValueError("Cycle evidence must be derived from the joint edge law.")
        if not np.allclose(
            np.asarray(cycle_scores),
            np.asarray(expected_cycle_scores),
            atol=tolerance,
            rtol=1.0e-6,
        ):
            raise ValueError("Cycle standard scores do not match cycle covariance.")
        design_rank_array = jnp.asarray(design_rank, dtype=jnp.int32).reshape(())
        covariance_rank_array = jnp.asarray(covariance_rank, dtype=jnp.int32).reshape(())
        design_rank_value = int(np.asarray(design_rank_array))
        covariance_rank_value = int(np.asarray(covariance_rank_array))
        if design_rank_value < 0 or design_rank_value > state_count - 1:
            raise ValueError("design_rank is incompatible with the fixed gauge.")
        if covariance_rank_value < 0 or covariance_rank_value > edge_count:
            raise ValueError("covariance_rank is incompatible with the edge covariance.")
        difference = free[None, :] - free[:, None]
        diagonal = jnp.diag(covariance_)
        errors = jnp.sqrt(
            jnp.maximum(diagonal[:, None] + diagonal[None, :] - 2.0 * covariance_, 0.0)
        )
        ids = tuple(_identifier(value, "observation_id") for value in observation_ids)
        if len(ids) != edge_count or len(set(ids)) != len(ids):
            raise ValueError("observation_ids must uniquely identify every ordered edge.")
        numerical_value = int(numerical_status)
        statistical_value = int(statistical_status)
        known_status = sum(int(value) for value in FreeEnergyStatus if value)
        if (
            numerical_value < 0
            or statistical_value < 0
            or numerical_value & ~known_status
            or statistical_value & ~known_status
        ):
            raise ValueError("Network status contains unknown bits.")
        numerical = jnp.asarray(numerical_value, dtype=jnp.int32)
        statistical = jnp.asarray(statistical_value, dtype=jnp.int32)
        analysis_id = canonical_fingerprint(
            {
                "kind": "free-energy-network-result",
                "plan_id": plan.plan_id,
                "observation_ids": ids,
                "numerical_status": int(np.asarray(numerical)),
                "statistical_status": int(np.asarray(statistical)),
                "arrays": array_tree_fingerprint(
                    {
                        "free_energies": free,
                        "covariance": covariance_,
                        "edge_values": values,
                        "edge_residuals": residuals,
                        "joint_edge_covariance": edge_covariance,
                        "cycle_matrix": cycles,
                        "cycle_residuals": cycle_residual,
                        "cycle_covariance": cycle_covariance_,
                        "cycle_standard_scores": cycle_scores,
                    }
                ),
            }
        )
        self.free_energies = free
        self.covariance = covariance_
        self.differences = difference
        self.standard_errors = errors
        self.edge_values = values
        self.edge_residuals = residuals
        self.joint_edge_covariance = edge_covariance
        self.cycle_matrix = cycles
        self.cycle_residuals = cycle_residual
        self.cycle_covariance = cycle_covariance_
        self.cycle_standard_scores = cycle_scores
        self.design_rank = design_rank_array
        self.covariance_rank = covariance_rank_array
        self.numerical_status = numerical
        self.statistical_status = statistical
        self.plan = plan
        self.observation_ids = ids
        self.analysis_id = analysis_id

    @property
    def successful(self) -> Array:
        return (self.numerical_status == int(FreeEnergyStatus.SUCCESS)) & (
            self.statistical_status == int(FreeEnergyStatus.SUCCESS)
        )


def _incidence(
    plan: FreeEnergyNetworkPlan,
    observations: Sequence[FreeEnergyEdgeObservation],
    /,
) -> np.ndarray:
    lookup = {state: index for index, state in enumerate(plan.state_ids)}
    incidence = np.zeros((len(observations), len(plan.state_ids)), dtype=float)
    for edge_index, observation in enumerate(observations):
        if (
            observation.source_state_id not in lookup
            or observation.destination_state_id not in lookup
        ):
            raise ValueError("Every edge state must occur in the network plan.")
        incidence[edge_index, lookup[observation.source_state_id]] = -1.0
        incidence[edge_index, lookup[observation.destination_state_id]] = 1.0
    return incidence


def _topologically_connected(incidence: np.ndarray, /) -> bool:
    factors = factor_pseudoinverse(jnp.asarray(incidence), RankPolicy())
    return int(np.asarray(factors.rank)) == incidence.shape[1] - 1


def _fundamental_cycle_matrix(incidence: np.ndarray, /) -> np.ndarray:
    edge_count, state_count = incidence.shape
    source = np.argmin(incidence, axis=1)
    destination = np.argmax(incidence, axis=1)
    parent = list(range(state_count))

    def root(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    tree_edges: list[int] = []
    non_tree_edges: list[int] = []
    for edge in range(edge_count):
        source_root = root(int(source[edge]))
        destination_root = root(int(destination[edge]))
        if source_root == destination_root:
            non_tree_edges.append(edge)
        else:
            parent[destination_root] = source_root
            tree_edges.append(edge)
    adjacency: list[list[tuple[int, int, float]]] = [[] for _ in range(state_count)]
    for edge in tree_edges:
        source_ = int(source[edge])
        destination_ = int(destination[edge])
        adjacency[source_].append((destination_, edge, 1.0))
        adjacency[destination_].append((source_, edge, -1.0))
    cycles: list[np.ndarray] = []
    for closing_edge in non_tree_edges:
        start = int(destination[closing_edge])
        target = int(source[closing_edge])
        queue: deque[int] = deque([start])
        predecessor: dict[int, tuple[int, int, float] | None] = {start: None}
        while queue and target not in predecessor:
            node = queue.popleft()
            for neighbor, edge, sign in adjacency[node]:
                if neighbor not in predecessor:
                    predecessor[neighbor] = (node, edge, sign)
                    queue.append(neighbor)
        if target not in predecessor:
            continue
        cycle = np.zeros((edge_count,), dtype=float)
        cycle[closing_edge] = 1.0
        node = target
        while node != start:
            record = predecessor[node]
            if record is None:
                raise RuntimeError("Cycle predecessor chain ended unexpectedly.")
            previous, edge, sign = record
            cycle[edge] = sign
            node = previous
        cycles.append(cycle)
    return np.stack(cycles) if cycles else np.zeros((0, edge_count), dtype=float)


def _joint_covariance_from_influences(
    observations: Sequence[FreeEnergyEdgeObservation],
    /,
) -> Array:
    bases = tuple(observation.influence_basis_id for observation in observations)
    if any(value is None for value in bases) or len(set(bases)) != 1:
        raise ValueError(
            "joint_covariance is required unless every edge shares one influence basis."
        )
    sizes = {observation.influence_values.size for observation in observations}
    if len(sizes) != 1:
        raise ValueError("Shared-basis edge influences must have one coordinate size.")
    influence = jnp.stack(
        tuple(observation.influence_values for observation in observations)
    )
    independent = jnp.asarray(
        [observation.independent_variance for observation in observations]
    )
    return influence @ influence.T + jnp.diag(independent)


def analyze_free_energy_network(
    plan: FreeEnergyNetworkPlan,
    observations: Sequence[FreeEnergyEdgeObservation],
    /,
    *,
    joint_covariance: ArrayLike | None = None,
) -> FreeEnergyNetworkResult:
    """Fit a network by GLS without discarding cross-edge covariance."""

    if not isinstance(plan, FreeEnergyNetworkPlan):
        raise TypeError("plan must be FreeEnergyNetworkPlan.")
    edges = tuple(observations)
    if len(edges) < 1 or any(
        not isinstance(edge, FreeEnergyEdgeObservation) for edge in edges
    ):
        raise TypeError("observations must be a non-empty sequence of edge observations.")
    observation_ids = tuple(edge.observation_id for edge in edges)
    if len(set(observation_ids)) != len(observation_ids):
        raise ValueError("Edge observation identities must be unique.")
    incidence_host = _incidence(plan, edges)
    incidence = jnp.asarray(incidence_host)
    values = jnp.asarray([edge.value for edge in edges])
    covariance = (
        _joint_covariance_from_influences(edges)
        if joint_covariance is None
        else _real_array(joint_covariance, "joint_covariance")
    )
    edge_count = len(edges)
    if covariance.shape != (edge_count, edge_count):
        raise ValueError("joint_covariance must be square over ordered observations.")
    covariance = 0.5 * (covariance + covariance.T)
    host_covariance = np.asarray(covariance)
    tolerance = _covariance_tolerance(host_covariance)
    if not np.all(np.isfinite(host_covariance)):
        raise ValueError("joint_covariance must be finite.")
    if float(np.min(np.linalg.eigvalsh(host_covariance))) < -tolerance:
        raise ValueError("joint_covariance must be positive semidefinite.")
    declared_variance = np.asarray([edge.variance for edge in edges])
    if not np.allclose(
        np.diag(host_covariance), declared_variance, atol=tolerance, rtol=1e-6
    ):
        raise ValueError("joint_covariance diagonal must equal declared edge variances.")
    reference = plan.state_ids.index(plan.reference_state_id)
    kept = [index for index in range(len(plan.state_ids)) if index != reference]
    design = incidence[:, jnp.asarray(kept, dtype=jnp.int32)]
    covariance_factors = factor_pseudoinverse(
        covariance,
        RankPolicy(relative_cutoff=plan.rank_tolerance),
        hermitian=True,
    )
    covariance_rank = int(np.asarray(covariance_factors.rank))
    precision = materialize_pseudoinverse(covariance_factors)
    null_constraints = jnp.conj(covariance_factors.left_vectors[:, covariance_rank:].T)
    information = design.T @ precision @ design
    constraint_design = null_constraints @ design
    state_size = len(kept)
    nullity = null_constraints.shape[0]
    kkt = jnp.block(
        [
            [information, jnp.conj(constraint_design.T)],
            [constraint_design, jnp.zeros((nullity, nullity), dtype=design.dtype)],
        ]
    )
    right_hand_side = jnp.concatenate(
        (design.T @ precision @ values, null_constraints @ values)
    )
    kkt_factors = factor_pseudoinverse(
        kkt,
        RankPolicy(relative_cutoff=plan.rank_tolerance),
        hermitian=True,
    )
    solution = apply_pseudoinverse(kkt_factors, right_hand_side)[:state_size]
    map_right = jnp.concatenate((design.T @ precision, null_constraints), axis=0)
    linear_map = apply_pseudoinverse(kkt_factors, map_right)[:state_size]
    reduced_covariance = linear_map @ covariance @ linear_map.T
    reduced_covariance = 0.5 * (reduced_covariance + reduced_covariance.T)
    free = jnp.zeros((len(plan.state_ids),), dtype=values.dtype)
    free = free.at[jnp.asarray(kept, dtype=jnp.int32)].set(solution)
    state_covariance = jnp.zeros(
        (len(plan.state_ids), len(plan.state_ids)), dtype=values.dtype
    )
    kept_array = jnp.asarray(kept, dtype=jnp.int32)
    state_covariance = state_covariance.at[kept_array[:, None], kept_array[None, :]].set(
        reduced_covariance
    )
    residuals = values - incidence @ free
    cycle_matrix_host = _fundamental_cycle_matrix(incidence_host)
    cycle_matrix = jnp.asarray(cycle_matrix_host, dtype=values.dtype)
    cycle_residuals = cycle_matrix @ values
    cycle_covariance = cycle_matrix @ covariance @ cycle_matrix.T
    cycle_variance = jnp.maximum(jnp.diag(cycle_covariance), 0.0)
    cycle_standard_scores = jnp.abs(cycle_residuals) / jnp.sqrt(
        jnp.maximum(cycle_variance, jnp.finfo(values.dtype).tiny)
    )
    positive = covariance_factors.singular_values[:covariance_rank]
    weighted_design = (
        (jnp.conj(covariance_factors.left_vectors[:, :covariance_rank].T) @ design)
        / jnp.sqrt(positive)[:, None]
        if covariance_rank > 0
        else jnp.zeros((0, state_size), dtype=design.dtype)
    )
    rank_design = jnp.concatenate((weighted_design, constraint_design), axis=0)
    design_factors = factor_pseudoinverse(
        rank_design,
        RankPolicy(relative_cutoff=plan.rank_tolerance),
    )
    design_rank = int(np.asarray(design_factors.rank))
    numerical_status = FreeEnergyStatus.SUCCESS
    statistical_status = FreeEnergyStatus.SUCCESS
    if design_rank < state_size:
        numerical_status |= FreeEnergyStatus.RANK_DEFICIENT
    if not _topologically_connected(incidence_host):
        statistical_status |= FreeEnergyStatus.DISCONNECTED
    exact_cycle_conflict = (np.asarray(cycle_variance) <= tolerance) & (
        np.abs(np.asarray(cycle_residuals)) > tolerance
    )
    if bool(np.any(exact_cycle_conflict)) or bool(
        jnp.any(cycle_standard_scores > plan.maximum_cycle_z_score)
    ):
        statistical_status |= FreeEnergyStatus.CYCLE_INCONSISTENT
    if not bool(jnp.all(jnp.isfinite(free))) or not bool(
        jnp.all(jnp.isfinite(state_covariance))
    ):
        numerical_status |= FreeEnergyStatus.NONFINITE
        free = jnp.where(jnp.isfinite(free), free, 0.0)
        state_covariance = jnp.where(
            jnp.isfinite(state_covariance), state_covariance, 0.0
        )
    return FreeEnergyNetworkResult(
        free,
        state_covariance,
        values,
        residuals,
        covariance,
        cycle_matrix,
        cycle_residuals,
        cycle_covariance,
        cycle_standard_scores,
        jnp.asarray(design_rank),
        jnp.asarray(covariance_rank),
        numerical_status,
        statistical_status,
        plan,
        observation_ids=observation_ids,
    )


__all__ = [
    "FreeEnergyEdgeObservation",
    "FreeEnergyNetworkPlan",
    "FreeEnergyNetworkResult",
    "analyze_free_energy_network",
]
