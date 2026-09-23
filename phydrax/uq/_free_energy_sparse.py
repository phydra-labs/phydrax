#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit pairwise analysis for sparsely cross-evaluated reduced potentials."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._free_energy import (
    _dataset_fingerprint,
    _dimensionless_unit,
    _floating_array,
    _identifier,
    _identifiers,
    _index_array,
    _sampling_qualification,
    _state_bias_ids,
    _validate_sample_identity,
    bennett_acceptance_ratio,
    FreeEnergyResult,
    FreeEnergySelectionPlan,
    ReducedWorkDataset,
)
from ._free_energy_network import (
    analyze_free_energy_network,
    FreeEnergyEdgeObservation,
    FreeEnergyNetworkPlan,
    FreeEnergyNetworkResult,
)


class SparseReducedPotentialDataset(StrictModule, NonTrainableState):
    """Authenticated sparse state-by-sample reduced-potential coverage."""

    values: Array
    coverage: Array
    sample_active: Array
    origin_state: Array
    chain_index: Array
    draw_index: Array
    repeat_index: Array
    dependence_group_index: Array
    inverse_temperatures: Array
    state_ids: tuple[str, ...] = eqx.field(static=True)
    potential_ids: tuple[str, ...] = eqx.field(static=True)
    bias_ids: tuple[str | None, ...] = eqx.field(static=True)
    measure_id: str = eqx.field(static=True)
    producer_id: str = eqx.field(static=True)
    run_id: str = eqx.field(static=True)
    unit_system_id: str | None = eqx.field(static=True)
    unit_id: str = eqx.field(static=True)
    reduced_convention_id: str = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)
    sampling_exact: bool = eqx.field(static=True)
    sampling_bias_bound: float = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        coverage: ArrayLike,
        sample_active: ArrayLike,
        origin_state: ArrayLike,
        chain_index: ArrayLike,
        draw_index: ArrayLike,
        repeat_index: ArrayLike,
        dependence_group_index: ArrayLike,
        /,
        *,
        state_ids: Sequence[str],
        potential_ids: Sequence[str],
        inverse_temperatures: ArrayLike,
        measure_id: str,
        producer_id: str,
        run_id: str,
        reduced_convention_id: str,
        qualification_id: str,
        sampling_exact: bool,
        sampling_bias_bound: float,
        bias_ids: Sequence[str | None] = (),
        unit_system_id: str | None = None,
        unit_id: str,
    ):
        potential = _floating_array(values, "values")
        if potential.ndim != 2 or potential.shape[0] < 2 or potential.shape[1] < 1:
            raise ValueError(
                "values must have shape (states, capacity) with at least two states."
            )
        state_count, capacity = potential.shape
        covered = jnp.asarray(coverage, dtype=jnp.bool_)
        active = jnp.asarray(sample_active, dtype=jnp.bool_)
        origin = _index_array(origin_state, "origin_state")
        chain = _index_array(chain_index, "chain_index")
        draw = _index_array(draw_index, "draw_index")
        repeat = _index_array(repeat_index, "repeat_index")
        dependence = _index_array(dependence_group_index, "dependence_group_index")
        beta = _floating_array(inverse_temperatures, "inverse_temperatures")
        if covered.shape != potential.shape:
            raise ValueError("coverage must have the same shape as values.")
        for name, array in (
            ("sample_active", active),
            ("origin_state", origin),
            ("chain_index", chain),
            ("draw_index", draw),
            ("repeat_index", repeat),
            ("dependence_group_index", dependence),
        ):
            if array.shape != (capacity,):
                raise ValueError(f"{name} must have shape (capacity,).")
        if beta.shape != (state_count,) or not bool(
            jnp.all(jnp.isfinite(beta) & (beta > 0.0))
        ):
            raise ValueError(
                "inverse_temperatures must contain one finite positive beta per state."
            )
        host_active = np.asarray(active)
        host_origin = np.asarray(origin)
        if np.any(
            (host_origin[host_active] < 0) | (host_origin[host_active] >= state_count)
        ):
            raise ValueError("Every active sample must name one valid origin state.")
        safe_origin = jnp.clip(origin, 0, state_count - 1)
        origin_covered = covered[safe_origin, jnp.arange(capacity)]
        if not bool(jnp.all(jnp.where(active, origin_covered, True))):
            raise ValueError("Every active sample must cover its origin-state potential.")
        canonical_coverage = covered & active[None, :]
        if not bool(
            jnp.all(jnp.where(canonical_coverage, jnp.isfinite(potential), True))
        ):
            raise ValueError("Covered active reduced potentials must be finite.")
        _validate_sample_identity(
            host_active,
            host_origin,
            np.asarray(chain),
            np.asarray(draw),
            np.asarray(repeat),
            np.asarray(dependence),
        )
        states = _identifiers(state_ids, "state_id", count=state_count, unique=True)
        potentials = _identifiers(potential_ids, "potential_id", count=state_count)
        biases = _state_bias_ids(bias_ids, state_count)
        measure = _identifier(measure_id, "measure_id")
        producer = _identifier(producer_id, "producer_id")
        run = _identifier(run_id, "run_id")
        convention = _identifier(reduced_convention_id, "reduced_convention_id")
        qualification, exact, bias_bound = _sampling_qualification(
            qualification_id,
            sampling_exact,
            sampling_bias_bound,
        )
        unit_system = (
            None
            if unit_system_id is None
            else _identifier(unit_system_id, "unit_system_id")
        )
        unit = _dimensionless_unit(unit_id)
        canonical_values = jnp.where(canonical_coverage, potential, 0.0)
        canonical_origin = jnp.where(active, origin, -1)
        canonical_chain = jnp.where(active, chain, -1)
        canonical_draw = jnp.where(active, draw, -1)
        canonical_repeat = jnp.where(active, repeat, -1)
        canonical_dependence = jnp.where(active, dependence, -1)
        metadata = {
            "state_ids": states,
            "potential_ids": potentials,
            "bias_ids": biases,
            "measure_id": measure,
            "producer_id": producer,
            "run_id": run,
            "unit_system_id": unit_system,
            "unit_id": unit,
            "reduced_convention_id": convention,
            "qualification_id": qualification,
            "sampling_exact": exact,
            "sampling_bias_bound": bias_bound.hex(),
        }
        arrays = {
            "values": canonical_values,
            "coverage": canonical_coverage,
            "sample_active": active,
            "origin_state": canonical_origin,
            "chain_index": canonical_chain,
            "draw_index": canonical_draw,
            "repeat_index": canonical_repeat,
            "dependence_group_index": canonical_dependence,
            "inverse_temperatures": beta,
        }
        self.values = canonical_values
        self.coverage = canonical_coverage
        self.sample_active = active
        self.origin_state = canonical_origin
        self.chain_index = canonical_chain
        self.draw_index = canonical_draw
        self.repeat_index = canonical_repeat
        self.dependence_group_index = canonical_dependence
        self.inverse_temperatures = beta
        self.state_ids = states
        self.potential_ids = potentials
        self.bias_ids = biases
        self.measure_id = measure
        self.producer_id = producer
        self.run_id = run
        self.unit_system_id = unit_system
        self.unit_id = unit
        self.reduced_convention_id = convention
        self.qualification_id = qualification
        self.sampling_exact = exact
        self.sampling_bias_bound = bias_bound
        self.dataset_id = _dataset_fingerprint(
            "sparse-reduced-potential-dataset", metadata, arrays
        )

    @property
    def state_counts(self) -> Array:
        safe_origin = jnp.where(self.sample_active, self.origin_state, 0)
        return jnp.bincount(
            safe_origin,
            weights=self.sample_active.astype(jnp.int32),
            length=len(self.state_ids),
        ).astype(jnp.int32)


class SparsePairwiseFreeEnergyNetworkResult(StrictModule, NonTrainableState):
    """Pairwise BAR edge evidence and its diagonal-covariance network fit."""

    edge_results: tuple[FreeEnergyResult, ...]
    edge_observations: tuple[FreeEnergyEdgeObservation, ...]
    network_result: FreeEnergyNetworkResult
    edge_state_ids: tuple[tuple[str, str], ...] = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)
    analysis_id: str = eqx.field(static=True)

    def __init__(
        self,
        edge_results: Sequence[FreeEnergyResult],
        edge_observations: Sequence[FreeEnergyEdgeObservation],
        network_result: FreeEnergyNetworkResult,
        /,
        *,
        edge_state_ids: Sequence[tuple[str, str]],
        dataset_id: str,
    ):
        results = tuple(edge_results)
        observations = tuple(edge_observations)
        edges = tuple(
            (str(source), str(destination)) for source, destination in edge_state_ids
        )
        if not results or len(results) != len(observations) or len(results) != len(edges):
            raise ValueError("Sparse pairwise result components must align by edge.")
        if any(not isinstance(value, FreeEnergyResult) for value in results):
            raise TypeError("edge_results must contain FreeEnergyResult values.")
        if any(
            not isinstance(value, FreeEnergyEdgeObservation) for value in observations
        ):
            raise TypeError(
                "edge_observations must contain FreeEnergyEdgeObservation values."
            )
        if not isinstance(network_result, FreeEnergyNetworkResult):
            raise TypeError("network_result must be FreeEnergyNetworkResult.")
        identity = _identifier(dataset_id, "dataset_id")
        analysis_id = canonical_fingerprint(
            {
                "kind": "sparse-pairwise-free-energy-network-result",
                "dataset_id": identity,
                "edges": edges,
                "edge_analyzes": tuple(value.analysis_id for value in results),
                "network_analysis": network_result.analysis_id,
            }
        )
        self.edge_results = results
        self.edge_observations = observations
        self.network_result = network_result
        self.edge_state_ids = edges
        self.dataset_id = identity
        self.analysis_id = analysis_id

    @property
    def successful(self) -> Array:
        edges_successful = jnp.all(
            jnp.asarray([value.successful for value in self.edge_results])
        )
        return edges_successful & self.network_result.successful


def _edge_indices(
    dataset: SparseReducedPotentialDataset,
    edges: Sequence[tuple[str, str]],
    /,
) -> tuple[tuple[int, int], ...]:
    lookup = {state_id: index for index, state_id in enumerate(dataset.state_ids)}
    result: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for edge in edges:
        if len(edge) != 2:
            raise ValueError("Each sparse edge must contain source and destination IDs.")
        source_id, destination_id = edge
        if source_id not in lookup or destination_id not in lookup:
            raise ValueError("Every sparse edge state must occur in dataset.state_ids.")
        source, destination = lookup[source_id], lookup[destination_id]
        if source == destination or (source, destination) in seen:
            raise ValueError("Sparse edges must be distinct oriented state pairs.")
        seen.add((source, destination))
        result.append((source, destination))
    if not result:
        raise ValueError("At least one explicit sparse edge is required.")
    return tuple(result)


def sparse_pairwise_free_energy_network(
    dataset: SparseReducedPotentialDataset,
    edges: Sequence[tuple[str, str]],
    /,
    *,
    reference_state_id: str,
    selection_plan: FreeEnergySelectionPlan | None = None,
    maximum_iterations: int = 128,
    tolerance: float = 1.0e-10,
    network_rank_tolerance: float = 1.0e-12,
    maximum_cycle_z_score: float = 5.0,
    key: ArrayLike | None = None,
) -> SparsePairwiseFreeEnergyNetworkResult:
    """Fit explicit sparse BAR edges, then combine them by diagonal-covariance GLS."""

    if not isinstance(dataset, SparseReducedPotentialDataset):
        raise TypeError("dataset must be SparseReducedPotentialDataset.")
    selection = FreeEnergySelectionPlan() if selection_plan is None else selection_plan
    if not isinstance(selection, FreeEnergySelectionPlan):
        raise TypeError("selection_plan must be FreeEnergySelectionPlan or None.")
    indexed_edges = _edge_indices(dataset, edges)
    edge_ids = tuple(
        (dataset.state_ids[source], dataset.state_ids[destination])
        for source, destination in indexed_edges
    )
    used_groups: list[set[tuple[int, int]]] = []
    eligible_masks: list[np.ndarray] = []
    for source, destination in indexed_edges:
        origin = np.asarray(dataset.origin_state)
        active = np.asarray(dataset.sample_active)
        coverage = np.asarray(dataset.coverage)
        eligible = (
            active
            & coverage[source]
            & coverage[destination]
            & ((origin == source) | (origin == destination))
        )
        groups = set(
            zip(
                np.asarray(dataset.repeat_index)[eligible].tolist(),
                np.asarray(dataset.dependence_group_index)[eligible].tolist(),
            )
        )
        if not groups:
            raise ValueError(
                "A sparse edge has no jointly covered source/destination samples."
            )
        if any(groups & previous for previous in used_groups):
            raise ValueError(
                "Sparse edges share dependence groups; use joint dense free-energy analysis."
            )
        used_groups.append(groups)
        eligible_masks.append(eligible)
    edge_keys = (
        (None,) * len(indexed_edges)
        if key is None
        else tuple(jr.split(jnp.asarray(key), len(indexed_edges)))
    )
    edge_results: list[FreeEnergyResult] = []
    observations: list[FreeEnergyEdgeObservation] = []
    for edge_number, ((source, destination), eligible, edge_key) in enumerate(
        zip(indexed_edges, eligible_masks, edge_keys, strict=True)
    ):
        inverse_temperatures = np.asarray(dataset.inverse_temperatures)[
            [source, destination]
        ]
        if not np.all(inverse_temperatures == inverse_temperatures[0]):
            raise ValueError(
                "Sparse pairwise reduced work requires one shared inverse temperature."
            )
        active = jnp.asarray(eligible)
        source_origin = dataset.origin_state == source
        destination_origin = dataset.origin_state == destination
        work = jnp.where(
            source_origin,
            dataset.values[destination] - dataset.values[source],
            dataset.values[source] - dataset.values[destination],
        )
        source_state = jnp.where(source_origin, 0, 1)
        destination_state = jnp.where(destination_origin, 0, 1)
        work_id = canonical_fingerprint(
            {
                "kind": "sparse-pairwise-reduced-work",
                "dataset_id": dataset.dataset_id,
                "source_state_id": dataset.state_ids[source],
                "destination_state_id": dataset.state_ids[destination],
            }
        )
        work_dataset = ReducedWorkDataset(
            work,
            active,
            active,
            source_state,
            destination_state,
            dataset.chain_index,
            dataset.draw_index,
            dataset.repeat_index,
            dataset.dependence_group_index,
            state_ids=(dataset.state_ids[source], dataset.state_ids[destination]),
            potential_ids=(
                dataset.potential_ids[source],
                dataset.potential_ids[destination],
            ),
            measure_ids=(dataset.measure_id, dataset.measure_id),
            producer_id=dataset.producer_id,
            run_id=dataset.run_id,
            work_id=work_id,
            work_kind="equilibrium-difference",
            inverse_temperature=float(inverse_temperatures[0]),
            qualification_id=dataset.qualification_id,
            sampling_exact=dataset.sampling_exact,
            sampling_bias_bound=dataset.sampling_bias_bound,
            bias_ids=(dataset.bias_ids[source], dataset.bias_ids[destination]),
            unit_system_id=dataset.unit_system_id,
            unit_id=dataset.unit_id,
        )
        result = bennett_acceptance_ratio(
            work_dataset,
            selection,
            maximum_iterations=maximum_iterations,
            tolerance=tolerance,
            key=edge_key,
        )
        if not bool(result.successful):
            raise ValueError(
                "A sparse edge did not produce qualified BAR evidence; inspect edge coverage and status."
            )
        observation = FreeEnergyEdgeObservation.from_result(
            result,
            result.state_ids[0],
            result.state_ids[1],
            observation_id=canonical_fingerprint(
                {
                    "kind": "sparse-free-energy-edge",
                    "dataset_id": dataset.dataset_id,
                    "edge_number": edge_number,
                    "source_state_id": result.state_ids[0],
                    "destination_state_id": result.state_ids[1],
                }
            ),
        )
        edge_results.append(result)
        observations.append(observation)
    edge_covariance = jnp.diag(
        jnp.asarray([observation.variance for observation in observations])
    )
    network_plan = FreeEnergyNetworkPlan(
        dataset.state_ids,
        reference_state_id=reference_state_id,
        rank_tolerance=network_rank_tolerance,
        maximum_cycle_z_score=maximum_cycle_z_score,
    )
    network = analyze_free_energy_network(
        network_plan,
        observations,
        joint_covariance=edge_covariance,
    )
    return SparsePairwiseFreeEnergyNetworkResult(
        edge_results,
        observations,
        network,
        edge_state_ids=edge_ids,
        dataset_id=dataset.dataset_id,
    )


__all__ = [
    "SparsePairwiseFreeEnergyNetworkResult",
    "SparseReducedPotentialDataset",
    "sparse_pairwise_free_energy_network",
]
