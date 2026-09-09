#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np

from ..._array_archive import (
    pack_array_tree,
    read_array_archive,
    unpack_array_tree,
    write_array_archive,
)
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ._model import ClimateDrivers, PreparedReducedClimate, ReducedClimateState


_FORMAT = "phydrax-reduced-climate-checkpoint"


def _runtime_contract(
    climate: PreparedReducedClimate, drivers: ClimateDrivers, /
) -> dict:
    return {
        "prepared_id": climate.prepared_id,
        "parameters": array_tree_fingerprint(climate.plan),
        "drivers": array_tree_fingerprint(drivers),
    }


def _validate_state(
    climate: PreparedReducedClimate, state: ReducedClimateState, /
) -> None:
    if (
        not isinstance(state, ReducedClimateState)
        or state.prepared_id != climate.prepared_id
    ):
        raise ValueError(
            "Checkpoint requires a state from the exact prepared climate runtime."
        )
    if (
        state.boxes.shape != climate.plan.gases.fractions.shape
        or state.temperature.shape != climate.plan.energy.capacities.shape
    ):
        raise ValueError("Checkpoint state dimensions do not match the runtime.")
    if (
        state.cumulative_emissions.shape != (3,)
        or state.cumulative_sink.shape != (3,)
        or any(
            value.shape != ()
            for value in (
                state.cumulative_forcing_energy,
                state.cumulative_outgoing_energy,
                state.time,
                state.step_index,
            )
        )
    ):
        raise ValueError("Checkpoint budget and clock arrays have incompatible shapes.")
    if not np.issubdtype(state.step_index.dtype, np.integer):
        raise ValueError("Checkpoint step index must be an integer.")
    if not all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree.leaves(state)):
        raise ValueError("Checkpoint state must be finite.")
    index = int(state.step_index)
    if not 0 <= index <= climate.step_count or not np.isclose(
        float(state.time),
        climate.start_time + index * climate.step_size,
        rtol=0.0,
        atol=1.0e-5 * climate.step_size,
    ):
        raise ValueError("Checkpoint must lie on an accepted prepared grid boundary.")
    if np.any(np.asarray(climate.plan.gases.concentration(state.boxes)) <= 0.0):
        raise ValueError("Checkpoint absolute concentrations must remain positive.")


def write_reduced_climate_checkpoint(
    path: str | Path,
    climate: PreparedReducedClimate,
    state: ReducedClimateState,
    drivers: ClimateDrivers,
    /,
) -> Path:
    """Atomically archive an accepted state, binding the full driver trajectory.

    Native array archive checksums protect arrays; a content identity also
    binds model coefficients, numerical clock and scenario. No pickle or new
    serialization framework is introduced.
    """
    _validate_state(climate, state)
    arrays: dict[str, object] = {}
    specification = pack_array_tree("state", state, arrays)
    contract = _runtime_contract(climate, drivers)
    payload = canonical_fingerprint(
        {
            "contract": contract,
            "state": specification,
            "arrays": array_tree_fingerprint(arrays),
        }
    )
    return write_array_archive(
        path,
        manifest={
            "format": _FORMAT,
            "contract": contract,
            "state": specification,
            "payload_id": payload,
        },
        arrays=arrays,
    )


def read_reduced_climate_checkpoint(
    path: str | Path,
    climate: PreparedReducedClimate,
    template: ReducedClimateState,
    drivers: ClimateDrivers,
    /,
) -> ReducedClimateState:
    """Restore only when the numerical model and complete scenario match."""
    _validate_state(climate, template)
    manifest, arrays = read_array_archive(path)
    if (
        set(manifest) != {"format", "contract", "state", "payload_id", "arrays"}
        or manifest["format"] != _FORMAT
    ):
        raise ValueError("Not a canonical native reduced climate checkpoint.")
    contract = _runtime_contract(climate, drivers)
    if manifest["contract"] != contract:
        raise ValueError("Climate checkpoint model, clock or scenario does not match.")
    payload = canonical_fingerprint(
        {
            "contract": contract,
            "state": manifest["state"],
            "arrays": array_tree_fingerprint(arrays),
        }
    )
    if manifest["payload_id"] != payload:
        raise ValueError("Climate checkpoint payload identity is corrupt.")
    restored = unpack_array_tree(manifest["state"], arrays, template)
    _validate_state(climate, restored)
    return restored


__all__ = ["read_reduced_climate_checkpoint", "write_reduced_climate_checkpoint"]
