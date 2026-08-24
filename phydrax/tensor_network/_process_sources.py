#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jaxtyping import ArrayLike

from ._causal_process import CausalProcessTensor, CombLegSpec
from ._local_lindblad import prepare_local_lindblad_channel


def causal_process_from_lindblad(
    hamiltonian: ArrayLike,
    jump_operators: ArrayLike,
    initial_density: ArrayLike,
    /,
    *,
    step_size: ArrayLike,
    slot_count: int,
    process_id: str = "lindblad-causal-process",
) -> CausalProcessTensor:
    prepared = prepare_local_lindblad_channel(hamiltonian, jump_operators, step_size)
    dimension = jnp.asarray(hamiltonian).shape[0]
    return CausalProcessTensor(
        CombLegSpec(dimension, 1, int(slot_count)),
        initial_density,
        tuple(prepared.kraus for _ in range(int(slot_count))),
        process_id=process_id,
    )


def causal_process_from_unitaries(
    unitaries: Sequence[ArrayLike],
    initial_density: ArrayLike,
    /,
    *,
    process_id: str = "unitary-causal-process",
) -> CausalProcessTensor:
    values = tuple(jnp.asarray(unitary) for unitary in unitaries)
    if not values:
        raise ValueError("At least one unitary is required.")
    dimension = values[0].shape[0]
    if any(value.shape != (dimension, dimension) for value in values):
        raise ValueError("Unitary process dimensions must match.")
    return CausalProcessTensor(
        CombLegSpec(dimension, 1, len(values)),
        initial_density,
        tuple(value[None, ...] for value in values),
        process_id=process_id,
    )


__all__ = ["causal_process_from_lindblad", "causal_process_from_unitaries"]
