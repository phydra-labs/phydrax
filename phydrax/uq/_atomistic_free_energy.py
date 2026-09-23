#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Strict bridges from native atomistic records to authenticated UQ datasets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import jax.numpy as jnp
import numpy as np

from .._fingerprint import canonical_fingerprint
from ._free_energy import ReducedPotentialDataset, ReducedWorkDataset


def _exact(value: str, expected: str, name: str, /) -> None:
    if value != expected:
        raise ValueError(f"{name} does not match the atomistic record.")


def _exact_tuple(values, expected, name: str, /) -> tuple:
    resolved = tuple(expected)
    if tuple(values) != resolved:
        raise ValueError(f"{name} do not match the atomistic record.")
    return resolved


def reduced_potential_dataset_from_alchemical_evaluation(
    evaluation,
    sample_active,
    origin_state,
    chain_index,
    draw_index,
    repeat_index,
    dependence_group_index,
    /,
    *,
    state_ids: Sequence[str],
    potential_ids: Sequence[str],
    bias_ids: Sequence[str | None],
    control_ids: Sequence[str],
    inverse_temperatures,
    reduced_convention_id: str,
    qualification_id: str,
    sampling_exact: bool,
    sampling_bias_bound: float,
    measure_id: str,
    unit_system_id: str,
    prepared_id: str,
    thermodynamic_table_id: str,
    schedule_id: str,
    run_id: str,
    unit_id: str,
) -> ReducedPotentialDataset:
    """Authenticate controlled-Hamiltonian cross evaluations for MBAR."""

    from ..atomistic._alchemical import AlchemicalReducedPotentialEvaluation

    if not isinstance(evaluation, AlchemicalReducedPotentialEvaluation):
        raise TypeError("evaluation must be AlchemicalReducedPotentialEvaluation.")
    states = _exact_tuple(evaluation.state_ids, state_ids, "state_ids")
    potentials = _exact_tuple(evaluation.potential_ids, potential_ids, "potential_ids")
    biases = _exact_tuple(evaluation.bias_ids, bias_ids, "bias_ids")
    controls = _exact_tuple(evaluation.control_ids, control_ids, "control_ids")
    _exact(evaluation.measure_id, measure_id, "measure_id")
    _exact(evaluation.unit_system_id, unit_system_id, "unit_system_id")
    _exact(evaluation.prepared_id, prepared_id, "prepared_id")
    _exact(evaluation.schedule_id, schedule_id, "schedule_id")
    _exact(
        evaluation.thermodynamic_table_id,
        thermodynamic_table_id,
        "thermodynamic_table_id",
    )
    if unit_id != "1":
        raise ValueError("Alchemical reduced potentials must use unit_id='1'.")
    beta = np.asarray(inverse_temperatures, dtype=np.float64)
    if beta.shape != np.asarray(
        evaluation.inverse_temperatures
    ).shape or not np.array_equal(beta, np.asarray(evaluation.inverse_temperatures)):
        raise ValueError("inverse_temperatures do not match the alchemical evaluation.")
    values = jnp.asarray(evaluation.values)
    coverage = jnp.asarray(evaluation.coverage, dtype=jnp.bool_)
    successful = jnp.asarray(evaluation.successful, dtype=jnp.bool_)
    if (
        values.ndim != 2
        or coverage.shape != values.shape
        or successful.shape != values.shape
    ):
        raise ValueError(
            "Alchemical reduced-potential arrays must share shape (states,samples)."
        )
    control_values = jnp.asarray(evaluation.controls)
    if control_values.ndim != 2 or control_values.shape != (
        len(states),
        len(controls),
    ):
        raise ValueError("Alchemical controls do not match state/control identities.")
    if values.shape[0] != len(states):
        raise ValueError("Alchemical reduced-potential rows do not match state_ids.")
    if not np.array_equal(np.asarray(coverage), np.asarray(successful)):
        raise ValueError("Alchemical coverage and successful evidence disagree.")
    return ReducedPotentialDataset(
        values,
        coverage,
        sample_active,
        origin_state,
        chain_index,
        draw_index,
        repeat_index,
        dependence_group_index,
        state_ids=states,
        potential_ids=potentials,
        inverse_temperatures=beta,
        reduced_convention_id=reduced_convention_id,
        qualification_id=qualification_id,
        sampling_exact=sampling_exact,
        sampling_bias_bound=sampling_bias_bound,
        measure_id=measure_id,
        producer_id=prepared_id,
        run_id=run_id,
        bias_ids=biases,
        unit_system_id=unit_system_id,
        unit_id=unit_id,
    )


def reduced_potential_dataset_from_multistate(result) -> ReducedPotentialDataset:
    """Flatten one committed multistate segment in canonical draw-major order."""

    from ..atomistic.sampling._multistate import AtomisticMultistateSegmentResult

    if not isinstance(result, AtomisticMultistateSegmentResult):
        raise TypeError("result must be AtomisticMultistateSegmentResult.")
    if not result.successful:
        raise ValueError("Multistate segment must be successful before UQ conversion.")
    values = jnp.asarray(result.reduced_potentials)
    coverage = jnp.asarray(result.coverage, dtype=jnp.bool_)
    active = jnp.asarray(result.sample_active, dtype=jnp.bool_)
    if values.ndim != 3:
        raise ValueError(
            "Segment reduced_potentials must have shape (capacity,states,replicas)."
        )
    capacity, state_count, replica_count = values.shape
    expected_matrix = (capacity, state_count, replica_count)
    expected_samples = (capacity, replica_count)
    if coverage.shape != expected_matrix:
        raise ValueError("Segment coverage must match reduced_potentials.")
    for name, array in (
        ("sample_active", active),
        ("origin_state", result.origin_state),
        ("chain_index", result.chain_index),
        ("draw_index", result.draw_index),
        ("repeat_index", result.repeat_index),
        ("dependence_group_index", result.dependence_group_index),
        ("sams_adapting", result.sams_adapting),
    ):
        if jnp.asarray(array).shape != expected_samples:
            raise ValueError(f"Segment {name} must have shape (capacity,replicas).")
    if bool(jnp.any(active & jnp.asarray(result.sams_adapting, dtype=jnp.bool_))):
        raise ValueError(
            "SAMS adaptation draws must be inactive for free-energy analysis."
        )
    if len(result.state_ids) != state_count or len(result.potential_ids) != state_count:
        raise ValueError(
            "Segment state/potential identities do not match the state axis."
        )
    if len(result.bias_ids) != state_count:
        raise ValueError("Segment bias identities do not match the state axis.")
    if result.unit_id != "1":
        raise ValueError("Multistate reduced potentials must use unit_id='1'.")
    if result.producer_id != result.runtime_id:
        raise ValueError("Segment producer_id must identify its prepared runtime.")
    unit_system_id = result.units.unit_system_id
    flattened_values = values.transpose((1, 0, 2)).reshape((state_count, -1))
    flattened_coverage = coverage.transpose((1, 0, 2)).reshape((state_count, -1))
    return ReducedPotentialDataset(
        flattened_values,
        flattened_coverage,
        active.reshape((-1,)),
        jnp.asarray(result.origin_state).reshape((-1,)),
        jnp.asarray(result.chain_index).reshape((-1,)),
        jnp.asarray(result.draw_index).reshape((-1,)),
        jnp.asarray(result.repeat_index).reshape((-1,)),
        jnp.asarray(result.dependence_group_index).reshape((-1,)),
        state_ids=result.state_ids,
        potential_ids=result.potential_ids,
        inverse_temperatures=result.inverse_temperatures,
        reduced_convention_id=result.reduced_convention_id,
        qualification_id=result.qualification_id,
        sampling_exact=result.sampling_exact,
        sampling_bias_bound=result.sampling_bias_bound,
        measure_id=result.measure_id,
        producer_id=result.producer_id,
        run_id=result.run_id,
        bias_ids=result.bias_ids,
        unit_system_id=unit_system_id,
        unit_id=result.unit_id,
    )


def reduced_work_dataset_from_alchemical_switching(
    record,
    /,
    *,
    direction: Literal["forward", "reverse", "both"],
    source_origin_id: int,
    destination_origin_id: int,
    state_ids: Sequence[str],
    potential_ids: Sequence[str],
    measure_ids: Sequence[str],
    unit_system_id: str,
    unit_id: str,
    producer_id: str,
    run_id: str,
    schedule_id: str,
    hamiltonian_id: str,
    thermodynamic_table_id: str,
    bias_ids: Sequence[str | None],
) -> ReducedWorkDataset:
    """Convert one or both authenticated switching orientations to FEP/BAR data."""

    from ..atomistic.free_energy._switching import AlchemicalSwitchingRecord

    if not isinstance(record, AlchemicalSwitchingRecord):
        raise TypeError("record must be AlchemicalSwitchingRecord.")
    biases = tuple(bias_ids)
    if biases != (None, None):
        raise ValueError("The current switching protocol is bias-free.")
    if direction not in ("forward", "reverse", "both"):
        raise ValueError("direction must be 'forward', 'reverse', or 'both'.")
    states = _exact_tuple(
        (record.source_state_id, record.destination_state_id), state_ids, "state_ids"
    )
    potentials = _exact_tuple(
        (record.source_potential_id, record.destination_potential_id),
        potential_ids,
        "potential_ids",
    )
    measures = _exact_tuple(record.measure_ids, measure_ids, "measure_ids")
    if measures[0] != measures[1]:
        raise ValueError(
            "Nonequilibrium switching requires one common phase-space measure."
        )
    _exact(record.unit_system_id, unit_system_id, "unit_system_id")
    _exact(record.unit_id, unit_id, "unit_id")
    _exact(record.producer_id, producer_id, "producer_id")
    _exact(record.run_id, run_id, "run_id")
    _exact(record.schedule_id, schedule_id, "schedule_id")
    _exact(record.hamiltonian_id, hamiltonian_id, "hamiltonian_id")
    _exact(
        record.thermodynamic_table_id,
        thermodynamic_table_id,
        "thermodynamic_table_id",
    )
    if record.work_kind != "nonequilibrium-switching":
        raise ValueError("Switching record has an unsupported work_kind.")
    if (
        record.forward_orientation != "source-to-destination"
        or record.reverse_orientation != "destination-to-source"
    ):
        raise ValueError("Switching record orientations are not canonical.")
    complete = bool(
        np.all(np.asarray(record.forward_coverage))
        and np.all(np.asarray(record.reverse_coverage))
    )
    if not record.successful or not complete:
        raise ValueError("Switching record must own complete runtime-qualified coverage.")
    forward_count = jnp.asarray(record.forward_work).size
    reverse_count = jnp.asarray(record.reverse_work).size
    if (
        record.forward_lineage.sample_count != forward_count
        or record.reverse_lineage.sample_count != reverse_count
        or len(record.forward_final_states) != forward_count
        or len(record.reverse_final_states) != reverse_count
    ):
        raise ValueError("Switching work, lineage, and final-state counts must agree.")
    source_origin = int(source_origin_id)
    destination_origin = int(destination_origin_id)
    if source_origin < 0 or destination_origin < 0 or source_origin == destination_origin:
        raise ValueError(
            "Switching endpoint origin identities must be distinct and non-negative."
        )
    if not np.all(np.asarray(record.forward_lineage.origin_ids) == source_origin):
        raise ValueError("Forward switching lineage has the wrong origin identity.")
    if not np.all(np.asarray(record.reverse_lineage.origin_ids) == destination_origin):
        raise ValueError("Reverse switching lineage has the wrong origin identity.")

    def oriented(work, coverage, lineage, source, destination):
        covered = jnp.asarray(coverage, dtype=jnp.bool_)
        active = jnp.ones(covered.shape, dtype=jnp.bool_)
        return (
            jnp.asarray(work),
            covered,
            active,
            jnp.full(covered.shape, source, dtype=jnp.int32),
            jnp.full(covered.shape, destination, dtype=jnp.int32),
            lineage.chain_ids,
            lineage.draw_indices,
            lineage.repeat_ids,
            lineage.dependence_ids,
        )

    forward = oriented(
        record.forward_work, record.forward_coverage, record.forward_lineage, 0, 1
    )
    reverse = oriented(
        record.reverse_work, record.reverse_coverage, record.reverse_lineage, 1, 0
    )
    if direction == "both":
        arrays = tuple(
            jnp.concatenate((left, right)) for left, right in zip(forward, reverse)
        )
        ordered_states = states
        ordered_potentials = potentials
        ordered_measures = measures
        ordered_biases = biases
    elif direction == "forward":
        arrays = forward
        ordered_states = states
        ordered_potentials = potentials
        ordered_measures = measures
        ordered_biases = biases
    else:
        work, coverage, active, _, _, chain, draw, repeat, dependence = reverse
        arrays = (
            work,
            coverage,
            active,
            jnp.zeros(active.shape, dtype=jnp.int32),
            jnp.ones(active.shape, dtype=jnp.int32),
            chain,
            draw,
            repeat,
            dependence,
        )
        ordered_states = (states[1], states[0])
        ordered_potentials = (potentials[1], potentials[0])
        ordered_measures = (measures[1], measures[0])
        ordered_biases = (biases[1], biases[0])
    work_id = (
        record.work_id
        if direction == "both"
        else canonical_fingerprint(
            {
                "kind": "oriented-alchemical-switching-work",
                "record_id": record.work_id,
                "direction": direction,
            }
        )
    )
    return ReducedWorkDataset(
        *arrays,
        state_ids=ordered_states,
        potential_ids=ordered_potentials,
        qualification_id=record.qualification_id,
        sampling_exact=record.sampling_exact,
        sampling_bias_bound=record.sampling_bias_bound,
        measure_ids=ordered_measures,
        inverse_temperature=record.inverse_temperature,
        producer_id=record.producer_id,
        run_id=record.run_id,
        work_id=work_id,
        work_kind="nonequilibrium-switching",
        bias_ids=ordered_biases,
        unit_system_id=record.unit_system_id,
        unit_id=record.unit_id,
    )


__all__ = [
    "reduced_potential_dataset_from_alchemical_evaluation",
    "reduced_potential_dataset_from_multistate",
    "reduced_work_dataset_from_alchemical_switching",
]
