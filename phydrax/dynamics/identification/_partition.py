#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...data_utils import CasePartitionManifest, PartitionName
from .._trajectory import TrajectoryData


class TrajectoryDataPartition(StrictModule, NonTrainableState):
    """Trajectory subsets bound to one shared physical-case partition."""

    train: TrajectoryData
    validation: TrajectoryData
    test: TrajectoryData
    partition: CasePartitionManifest
    partition_id: str = eqx.field(static=True)

    def __init__(
        self,
        train: TrajectoryData,
        validation: TrajectoryData,
        test: TrajectoryData,
        partition: CasePartitionManifest,
        /,
    ):
        if not isinstance(partition, CasePartitionManifest):
            raise TypeError("partition must be a CasePartitionManifest.")
        expected = tuple(
            len(partition.subset_ids(name)) for name in ("train", "validation", "test")
        )
        actual = (train.num_cases, validation.num_cases, test.num_cases)
        if actual != expected:
            raise ValueError("Trajectory subsets do not match the partition membership.")
        self.train = train
        self.validation = validation
        self.test = test
        self.partition = partition
        self.partition_id = partition.partition_id


def _case_first(values: Array, num_cases: int, case_rank: int, /) -> Array:
    return jnp.reshape(values, (num_cases,) + values.shape[case_rank:])


def _take_cases(
    data: TrajectoryData,
    indices: Sequence[int],
    partition_name: PartitionName,
    partition_id: str,
    /,
) -> TrajectoryData:
    positions = jnp.asarray(tuple(indices), dtype=jnp.int32)
    case_rank = len(data.case_shape)
    num_cases = data.num_cases
    coordinates = _case_first(data.coordinates, num_cases, case_rank)[positions]
    states = _case_first(data.states, num_cases, case_rank)[positions]
    sample_valid = _case_first(data.sample_valid, num_cases, case_rank)[positions]
    transition_valid = _case_first(data.transition_valid, num_cases, case_rank)[positions]
    reset_mask = _case_first(data.reset_mask, num_cases, case_rank)[positions]
    weights = _case_first(data.weights, num_cases, case_rank)[positions]
    inputs = (
        None
        if data.inputs is None
        else _case_first(data.inputs, num_cases, case_rank)[positions]
    )
    input_valid = (
        None
        if data.input_valid is None
        else _case_first(data.input_valid, num_cases, case_rank)[positions]
    )
    derivatives = (
        None
        if data.derivatives is None
        else _case_first(data.derivatives, num_cases, case_rank)[positions]
    )
    derivative_valid = (
        None
        if data.derivative_valid is None
        else _case_first(data.derivative_valid, num_cases, case_rank)[positions]
    )
    return TrajectoryData(
        coordinates,
        states,
        state_layout=data.state_layout,
        sample_valid=sample_valid,
        transition_valid=transition_valid,
        reset_mask=reset_mask,
        weights=weights,
        inputs=inputs,
        input_layout=data.input_layout,
        input_valid=input_valid,
        input_alignment=(
            "transitions" if data.input_alignment is None else data.input_alignment
        ),
        derivatives=derivatives,
        derivative_valid=derivative_valid,
        case_axes=("case",),
        case_axis_roles=("case",),
        coordinate_id=data.coordinate_id,
        coordinate_kind=data.coordinate_kind,
        source_id=f"{data.source_id}:{partition_id}:{partition_name}",
    )


def partition_trajectory_data(
    data: TrajectoryData,
    ordered_case_ids: Sequence[str],
    partition: CasePartitionManifest,
    /,
) -> TrajectoryDataPartition:
    """Apply a shared case partition without re-fitting or re-splitting data."""
    if not isinstance(data, TrajectoryData):
        raise TypeError("data must be TrajectoryData.")
    if not isinstance(partition, CasePartitionManifest):
        raise TypeError("partition must be a CasePartitionManifest.")
    ordered = tuple(str(case_id) for case_id in ordered_case_ids)
    if len(ordered) != data.num_cases:
        raise ValueError("ordered_case_ids must assign one ID to every trajectory case.")
    subsets = tuple(
        _take_cases(
            data,
            partition.indices(ordered, name),
            name,
            partition.partition_id,
        )
        for name in ("train", "validation", "test")
    )
    return TrajectoryDataPartition(*subsets, partition)


__all__ = ["TrajectoryDataPartition", "partition_trajectory_data"]
