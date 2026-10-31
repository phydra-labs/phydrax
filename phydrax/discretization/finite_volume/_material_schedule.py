from __future__ import annotations

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry._volume_material_map import VolumeMaterialChain
from ...typing import checked
from .._cell_mesh import CellMesh


@final
class MaterialChainSchedule(StrictModule, NonTrainableState):
    """Padded sparse edit actions and reusable live-cell slots for one chain."""

    source_row_slots: Array
    target_row_slots: Array
    source_row_valid: Array
    target_row_valid: Array
    edge_source_slots: Array
    edge_target_slots: Array
    edge_weight_mantissas: Array
    edge_weight_exponents: Array
    edge_valid: Array
    initial_slots: Array
    final_slots: Array
    source_count: int = eqx.field(static=True)
    target_count: int = eqx.field(static=True)
    slot_count: int = eqx.field(static=True)
    map_count: int = eqx.field(static=True)
    maximum_rows: int = eqx.field(static=True)
    maximum_edges: int = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        source_row_slots: ArrayLike,
        target_row_slots: ArrayLike,
        source_row_valid: ArrayLike,
        target_row_valid: ArrayLike,
        edge_source_slots: ArrayLike,
        edge_target_slots: ArrayLike,
        edge_weight_mantissas: ArrayLike,
        edge_weight_exponents: ArrayLike,
        edge_valid: ArrayLike,
        initial_slots: ArrayLike,
        final_slots: ArrayLike,
        /,
        *,
        source_count: int,
        target_count: int,
        slot_count: int,
        schedule_id: str,
    ) -> None:
        source_rows = jnp.asarray(source_row_slots, dtype=jnp.int64)
        target_rows = jnp.asarray(target_row_slots, dtype=jnp.int64)
        source_mask = jnp.asarray(source_row_valid, dtype=jnp.bool_)
        target_mask = jnp.asarray(target_row_valid, dtype=jnp.bool_)
        edge_sources = jnp.asarray(edge_source_slots, dtype=jnp.int64)
        edge_targets = jnp.asarray(edge_target_slots, dtype=jnp.int64)
        mantissas = jnp.asarray(edge_weight_mantissas)
        exponents = jnp.asarray(edge_weight_exponents, dtype=jnp.int64)
        edge_mask = jnp.asarray(edge_valid, dtype=jnp.bool_)
        initial = jnp.asarray(initial_slots, dtype=jnp.int64)
        final = jnp.asarray(final_slots, dtype=jnp.int64)
        maps, rows = source_rows.shape
        if (
            maps <= 0
            or target_rows.shape != source_rows.shape
            or source_mask.shape != source_rows.shape
            or target_mask.shape != target_rows.shape
            or edge_sources.ndim != 2
            or edge_sources.shape[0] != maps
            or edge_targets.shape != edge_sources.shape
            or mantissas.shape != edge_sources.shape
            or exponents.shape != edge_sources.shape
            or edge_mask.shape != edge_sources.shape
            or initial.shape != (source_count,)
            or final.shape != (target_count,)
            or min(source_count, target_count, slot_count) <= 0
            or not isinstance(schedule_id, str)
            or not schedule_id
        ):
            raise ValueError(
                "Material chain schedule banks have inconsistent shapes or endpoints."
            )
        self.source_row_slots = source_rows
        self.target_row_slots = target_rows
        self.source_row_valid = source_mask
        self.target_row_valid = target_mask
        self.edge_source_slots = edge_sources
        self.edge_target_slots = edge_targets
        self.edge_weight_mantissas = mantissas
        self.edge_weight_exponents = exponents
        self.edge_valid = edge_mask
        self.initial_slots = initial
        self.final_slots = final
        self.source_count = source_count
        self.target_count = target_count
        self.slot_count = slot_count
        self.map_count = maps
        self.maximum_rows = rows
        self.maximum_edges = edge_sources.shape[1]
        self.schedule_id = schedule_id


def _mesh_cell_ids(mesh: CellMesh, /) -> np.ndarray:
    ids = np.concatenate(
        tuple(np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks)
    )
    if ids.ndim != 1 or np.unique(ids).size != ids.size:
        raise ValueError(
            "Material chain endpoint cell IDs must be unique and one-dimensional."
        )
    return ids


def _ids(value: ArrayLike, name: str, /) -> np.ndarray:
    raw = np.asarray(jax.device_get(value))
    if raw.dtype.kind not in "iu" or raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional integer cell-ID array.")
    result = raw.astype(np.int64, copy=False)
    if np.unique(result).size != result.size:
        raise ValueError(f"{name} must contain unique cell IDs.")
    return result


@checked
def prepare_material_chain_schedule(
    chain: VolumeMaterialChain,
    source_ids: ArrayLike,
    target_ids: ArrayLike,
    /,
) -> MaterialChainSchedule:
    """Prepare one compact sparse edit schedule, preserving each edit in order."""
    if not isinstance(chain, VolumeMaterialChain):
        raise TypeError("chain must be a VolumeMaterialChain.")
    source = _ids(source_ids, "source_ids")
    target = _ids(target_ids, "target_ids")
    if not chain.maps:
        raise ValueError("A material chain schedule requires at least one edit.")
    if not np.array_equal(
        source, _mesh_cell_ids(chain.maps[0].source_mesh)
    ) or not np.array_equal(target, _mesh_cell_ids(chain.maps[-1].target_mesh)):
        raise ValueError(
            "Material chain schedule IDs must bind its exact endpoint meshes."
        )

    active_slots = {int(identity): row for row, identity in enumerate(source.tolist())}
    source_rows_by_map: list[np.ndarray] = []
    target_rows_by_map: list[np.ndarray] = []
    edge_sources_by_map: list[np.ndarray] = []
    edge_targets_by_map: list[np.ndarray] = []
    edge_mantissas_by_map: list[np.ndarray] = []
    edge_exponents_by_map: list[np.ndarray] = []
    slot_count = source.size

    for material_map in chain.maps:
        map_source_ids = _mesh_cell_ids(material_map.source_mesh)
        map_target_ids = _mesh_cell_ids(material_map.target_mesh)
        if set(map_source_ids.tolist()) != set(active_slots):
            raise ValueError(
                "Material chain edit source IDs do not match its predecessor."
            )
        source_row_slots = np.asarray(
            [active_slots[int(identity)] for identity in map_source_ids], dtype=np.int64
        )
        map_target_id_set = set(map_target_ids.tolist())
        retained = {
            int(identity): int(source_row_slots[row])
            for row, identity in enumerate(map_source_ids.tolist())
            if int(identity) in map_target_id_set
        }
        used = set(retained.values())
        reusable = [int(slot) for slot in source_row_slots if int(slot) not in used]
        reuse_index = 0
        next_slots: dict[int, int] = {}
        for identity in map_target_ids.tolist():
            value = int(identity)
            if value in retained:
                next_slots[value] = retained[value]
            elif reuse_index < len(reusable):
                next_slots[value] = reusable[reuse_index]
                reuse_index += 1
            else:
                next_slots[value] = slot_count
                slot_count += 1
        target_row_slots = np.asarray(
            [next_slots[int(identity)] for identity in map_target_ids], dtype=np.int64
        )
        source_rows = np.asarray(jax.device_get(material_map.source_rows), dtype=np.int64)
        target_rows = np.asarray(jax.device_get(material_map.target_rows), dtype=np.int64)
        weights = np.asarray(
            jax.device_get(material_map.extensive_weights), dtype=np.float64
        )
        if (
            source_rows.ndim != 1
            or target_rows.shape != source_rows.shape
            or weights.shape != source_rows.shape
            or source_rows.size == 0
            or np.any((source_rows < 0) | (source_rows >= map_source_ids.size))
            or np.any((target_rows < 0) | (target_rows >= map_target_ids.size))
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
            or np.any(np.bincount(source_rows, minlength=map_source_ids.size) == 0)
            or np.any(np.bincount(target_rows, minlength=map_target_ids.size) == 0)
        ):
            raise ValueError(
                "Material chain edit has incomplete or invalid sparse weights."
            )
        mantissas, exponents = np.frexp(weights)
        edge_sources_by_map.append(source_row_slots[source_rows])
        edge_targets_by_map.append(target_row_slots[target_rows])
        edge_mantissas_by_map.append(mantissas.astype(np.float64, copy=False))
        edge_exponents_by_map.append(exponents.astype(np.int64, copy=False))
        source_rows_by_map.append(source_row_slots)
        target_rows_by_map.append(target_row_slots)
        active_slots = next_slots

    maximum_rows = max(
        max(row.size for row in source_rows_by_map),
        max(row.size for row in target_rows_by_map),
    )
    maximum_edges = max(row.size for row in edge_sources_by_map)
    map_count = len(chain.maps)
    source_row_bank = np.zeros((map_count, maximum_rows), dtype=np.int64)
    target_row_bank = np.zeros_like(source_row_bank)
    source_valid = np.zeros((map_count, maximum_rows), dtype=np.bool_)
    target_valid = np.zeros_like(source_valid)
    edge_source_bank = np.zeros((map_count, maximum_edges), dtype=np.int64)
    edge_target_bank = np.zeros_like(edge_source_bank)
    mantissa_bank = np.zeros((map_count, maximum_edges), dtype=np.float64)
    exponent_bank = np.zeros((map_count, maximum_edges), dtype=np.int64)
    edge_valid = np.zeros((map_count, maximum_edges), dtype=np.bool_)
    for index in range(map_count):
        source_size = source_rows_by_map[index].size
        target_size = target_rows_by_map[index].size
        edge_size = edge_sources_by_map[index].size
        source_row_bank[index, :source_size] = source_rows_by_map[index]
        source_valid[index, :source_size] = True
        target_row_bank[index, :target_size] = target_rows_by_map[index]
        target_valid[index, :target_size] = True
        edge_source_bank[index, :edge_size] = edge_sources_by_map[index]
        edge_target_bank[index, :edge_size] = edge_targets_by_map[index]
        mantissa_bank[index, :edge_size] = edge_mantissas_by_map[index]
        exponent_bank[index, :edge_size] = edge_exponents_by_map[index]
        edge_valid[index, :edge_size] = True

    schedule_id = canonical_fingerprint(
        {
            "kind": "volume-material-chain-schedule",
            "chain": chain.chain_id,
            "source_ids": source.tolist(),
            "target_ids": target.tolist(),
            "source_row_slots": source_row_bank,
            "target_row_slots": target_row_bank,
            "edge_source_slots": edge_source_bank,
            "edge_target_slots": edge_target_bank,
            "edge_weight_mantissas": mantissa_bank,
            "edge_weight_exponents": exponent_bank,
        }
    )
    return MaterialChainSchedule(
        source_row_bank,
        target_row_bank,
        source_valid,
        target_valid,
        edge_source_bank,
        edge_target_bank,
        mantissa_bank,
        exponent_bank,
        edge_valid,
        np.arange(source.size, dtype=np.int64),
        np.asarray(
            [active_slots[int(identity)] for identity in target.tolist()],
            dtype=np.int64,
        ),
        source_count=source.size,
        target_count=target.size,
        slot_count=slot_count,
        schedule_id=schedule_id,
    )


__all__ = ["MaterialChainSchedule", "prepare_material_chain_schedule"]
