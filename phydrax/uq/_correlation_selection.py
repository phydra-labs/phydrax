#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host-only sequence and synchronous-block selection invariants.

The initial-positive sequence is the established free-energy convention, not
scalar-observable IMS. Window closure is additional evidence: existing
free-energy consumers intentionally do not require it.

The bounded host IPS kernel intentionally retains its established loop/reduction
order: its cyclomatic complexity is 19, but extracting each lag or sequence into
forwarding helpers would obscure numerical ownership without removing an
invariant. The touched ratio preparation and qualification phases are separately
decomposed; this source-faithful kernel is not an alternate runtime substrate.
"""

from __future__ import annotations

import math

import numpy as np

from ..typing import Dim, HostBool, HostFloat, HostFloat64, HostInt32, HostInteger


class ObservationDim(Dim):
    """Flattened aligned observations."""


class StratumDim(Dim):
    """Declared observation strata."""


def correlation_inefficiency(
    values: HostFloat[ObservationDim],
    retained: HostBool[ObservationDim],
    strata: HostInteger[ObservationDim],
    chain: HostInteger[ObservationDim],
    draw: HostInteger[ObservationDim],
    repeat: HostInteger[ObservationDim],
    dependence: HostInteger[ObservationDim],
    stratum_count: int,
    maximum_lag: int | None,
    /,
) -> tuple[HostFloat64[StratumDim], HostBool[StratumDim], HostBool[StratumDim]]:
    result = np.ones((stratum_count,), dtype=np.float64)
    resolved = np.ones((stratum_count,), dtype=np.bool_)
    closed = np.ones((stratum_count,), dtype=np.bool_)
    for stratum in range(stratum_count):
        selected_state = retained & (strata == stratum)
        if not np.any(selected_state):
            resolved[stratum] = False
            closed[stratum] = False
            continue
        group_keys = sorted(
            set(zip(repeat[selected_state].tolist(), dependence[selected_state].tolist()))
        )
        for repeat_id, dependence_id in group_keys:
            selected = (
                selected_state & (repeat == repeat_id) & (dependence == dependence_id)
            )
            sequences: list[HostFloat[ObservationDim]] = []
            for chain_id in sorted(set(chain[selected].tolist())):
                indices = np.nonzero(selected & (chain == chain_id))[0]
                indices = indices[np.argsort(draw[indices], kind="stable")]
                if indices.size:
                    sequences.append(values[indices])
            sample_count = sum(sequence.size for sequence in sequences)
            if sample_count < 2:
                resolved[stratum] = False
                closed[stratum] = False
                continue
            concatenated = np.concatenate(sequences)
            mean = float(np.mean(concatenated))
            variance_numerator = sum(
                float(np.sum((sequence - mean) ** 2)) for sequence in sequences
            )
            variance = variance_numerator / sample_count
            scale = max(float(np.max(np.abs(concatenated), initial=0.0)), 1.0)
            variance_floor = 128.0 * np.finfo(concatenated.dtype).eps * scale * scale
            if not math.isfinite(variance):
                resolved[stratum] = False
                closed[stratum] = False
                continue
            if variance <= variance_floor:
                closed[stratum] &= variance == 0.0
                continue
            available_lag = max(sequence.size for sequence in sequences) - 1
            if available_lag < 1:
                resolved[stratum] = False
                closed[stratum] = False
                continue
            lag_limit = (
                available_lag if maximum_lag is None else min(available_lag, maximum_lag)
            )
            correlations: list[float] = []
            correlation_valid = True
            for lag in range(1, lag_limit + 1):
                numerator = 0.0
                count = 0
                for sequence in sequences:
                    if sequence.size > lag:
                        numerator += float(
                            np.sum((sequence[:-lag] - mean) * (sequence[lag:] - mean))
                        )
                        count += sequence.size - lag
                if count == 0:
                    correlation_valid = False
                    break
                correlation = numerator / (count * variance)
                if not math.isfinite(correlation):
                    correlation_valid = False
                    break
                correlations.append(correlation)
            if not correlation_valid:
                resolved[stratum] = False
                closed[stratum] = False
                continue
            included = 0.0
            terminated = False
            for index in range(0, len(correlations), 2):
                pair_sum = correlations[index]
                if index + 1 < len(correlations):
                    pair_sum += correlations[index + 1]
                if pair_sum <= 0.0:
                    terminated = True
                    break
                included += pair_sum
            closed[stratum] &= terminated
            result[stratum] = max(result[stratum], max(1.0, 1.0 + 2.0 * included))
    return result, resolved, closed


def synchronous_block_indices(
    retained: HostBool[ObservationDim],
    repeat: HostInteger[ObservationDim],
    dependence: HostInteger[ObservationDim],
    draw: HostInteger[ObservationDim],
    block_length: int,
    /,
) -> tuple[HostInt32[ObservationDim], HostInt32[ObservationDim], int]:
    """Assign canonical group/draw-rank blocks without merging stream boundaries.

    Completeness is an estimator contract: free energy retains partial blocks;
    ratio analysis first removes partial groups/blocks before calling this owner.
    """
    block_keys: list[tuple[int, int, int]] = []
    group_keys = sorted(
        set(zip(repeat[retained].tolist(), dependence[retained].tolist()))
    )
    group_lookup = {key: index for index, key in enumerate(group_keys)}
    group_index = np.full(retained.shape, -1, dtype=np.int32)
    observation_block_key: dict[int, tuple[int, int, int]] = {}
    for key in group_keys:
        selected = retained & (repeat == key[0]) & (dependence == key[1])
        unique_draws = sorted(set(int(value) for value in draw[selected]))
        draw_rank = {value: index for index, value in enumerate(unique_draws)}
        group_index[selected] = group_lookup[key]
        for index in np.nonzero(selected)[0]:
            block_key = (key[0], key[1], draw_rank[int(draw[index])] // block_length)
            observation_block_key[int(index)] = block_key
            block_keys.append(block_key)
    unique_blocks = sorted(set(block_keys))
    block_lookup = {key: index for index, key in enumerate(unique_blocks)}
    block_index = np.full(retained.shape, -1, dtype=np.int32)
    for index, key in observation_block_key.items():
        block_index[index] = block_lookup[key]
    return block_index, group_index, len(unique_blocks)
