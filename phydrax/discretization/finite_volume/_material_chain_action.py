from __future__ import annotations

from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._numerics._compensated import compensated_sum_chunks
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry._volume_material_map import VolumeMaterialChain
from ._material_schedule import (
    _mesh_cell_ids,
    MaterialChainSchedule,
    prepare_material_chain_schedule,
)


_ZERO_EXPONENT = -(1 << 60)


def _multiply(
    left_mantissa: Array,
    left_exponent: Array,
    right_mantissa: Array,
    right_exponent: Array,
    /,
) -> tuple[Array, Array]:
    product, shift = jnp.frexp(left_mantissa * right_mantissa)
    exponent = left_exponent + right_exponent + shift.astype(jnp.int64)
    return product, jnp.where(product == 0.0, _ZERO_EXPONENT, exponent)


def _map_action(
    schedule: MaterialChainSchedule,
    values: Array,
    /,
    *,
    transpose: bool,
) -> Array:
    source_rows = schedule.target_count if transpose else schedule.source_count
    target_rows = schedule.source_count if transpose else schedule.target_count
    if values.ndim == 0 or values.shape[0] != source_rows:
        raise ValueError(
            f"Material chain values must start with {source_rows} endpoint rows; got {values.shape}."
        )
    payload = tuple(values.shape[1:])
    original_dtype = values.dtype
    if jnp.issubdtype(original_dtype, jnp.complexfloating):
        channels = jnp.stack((jnp.real(values), jnp.imag(values)), axis=-1)
        channels = channels.reshape((source_rows, -1))
    else:
        channels = values.reshape((source_rows, -1))
    compute_dtype = jnp.result_type(channels.dtype, jnp.float64)
    channels = channels.astype(compute_dtype)
    input_mantissas, input_exponents = jnp.frexp(channels)
    input_exponents = input_exponents.astype(jnp.int64)
    slots = schedule.slot_count
    state_size = slots + schedule.maximum_rows
    channel_count = channels.shape[1]
    state_mantissas = jnp.zeros((state_size, channel_count), dtype=compute_dtype)
    state_exponents = jnp.full(
        (state_size, channel_count), _ZERO_EXPONENT, dtype=jnp.int64
    )
    input_slots = schedule.final_slots if transpose else schedule.initial_slots
    state_mantissas = state_mantissas.at[input_slots].set(input_mantissas)
    state_exponents = state_exponents.at[input_slots].set(
        jnp.where(input_mantissas == 0.0, _ZERO_EXPONENT, input_exponents)
    )

    def action(index: Array, state: tuple[Array, Array]) -> tuple[Array, Array]:
        mantissas, exponents = state
        map_index = schedule.map_count - 1 - index if transpose else index
        edge_sources = schedule.edge_source_slots[map_index]
        edge_targets = schedule.edge_target_slots[map_index]
        valid = schedule.edge_valid[map_index]
        source_slots = edge_targets if transpose else edge_sources
        target_slots = edge_sources if transpose else edge_targets
        safe_sources = jnp.where(valid, source_slots, 0)
        safe_targets = jnp.where(valid, target_slots, 0)
        source_mantissas = mantissas[safe_sources]
        source_exponents = exponents[safe_sources]
        weighted_mantissas, weighted_exponents = _multiply(
            source_mantissas,
            source_exponents,
            schedule.edge_weight_mantissas[map_index, :, None].astype(compute_dtype),
            schedule.edge_weight_exponents[map_index, :, None],
        )
        contribution_valid = valid[:, None] & (weighted_mantissas != 0.0)
        weighted_exponents = jnp.where(
            contribution_valid, weighted_exponents, _ZERO_EXPONENT
        )
        maximum = jnp.full((state_size, channel_count), _ZERO_EXPONENT, dtype=jnp.int64)
        maximum = maximum.at[safe_targets].max(weighted_exponents)
        shift = weighted_exponents - maximum[safe_targets]
        aligned = jnp.ldexp(weighted_mantissas, shift)
        aligned = jnp.where(contribution_valid, aligned, 0.0)
        sums = jnp.zeros((state_size, channel_count), dtype=compute_dtype)
        sums = sums.at[safe_targets].add(aligned)
        output_mantissas, output_shifts = jnp.frexp(sums)
        output_exponents = maximum + output_shifts.astype(jnp.int64)
        output_exponents = jnp.where(sums == 0.0, _ZERO_EXPONENT, output_exponents)
        row_slots = (
            schedule.source_row_slots[map_index]
            if transpose
            else schedule.target_row_slots[map_index]
        )
        row_valid = (
            schedule.source_row_valid[map_index]
            if transpose
            else schedule.target_row_valid[map_index]
        )
        safe_rows = jnp.where(
            row_valid,
            row_slots,
            slots + jnp.arange(schedule.maximum_rows, dtype=jnp.int64),
        )
        prior_mantissas = mantissas[safe_rows]
        prior_exponents = exponents[safe_rows]
        mantissas = mantissas.at[safe_rows].set(
            jnp.where(row_valid[:, None], output_mantissas[safe_rows], prior_mantissas)
        )
        exponents = exponents.at[safe_rows].set(
            jnp.where(row_valid[:, None], output_exponents[safe_rows], prior_exponents)
        )
        return mantissas, exponents

    state_mantissas, state_exponents = lax.fori_loop(
        0,
        schedule.map_count,
        action,
        (state_mantissas, state_exponents),
    )
    output_slots = schedule.initial_slots if transpose else schedule.final_slots
    result = jnp.ldexp(state_mantissas[output_slots], state_exponents[output_slots])
    if jnp.issubdtype(original_dtype, jnp.complexfloating):
        paired = result.reshape((target_rows, *payload, 2))
        restored = lax.complex(paired[..., 0], paired[..., 1])
        return restored.astype(original_dtype)
    return result.reshape((target_rows, *payload)).astype(original_dtype)


def material_chain_apply(
    schedule: MaterialChainSchedule,
    values: ArrayLike,
    /,
) -> Array:
    """Apply the scheduled chronological extensive-content action."""
    if not isinstance(schedule, MaterialChainSchedule):
        raise TypeError("schedule must be a MaterialChainSchedule.")
    return _map_action(schedule, jnp.asarray(values), transpose=False)


def material_chain_transpose(
    schedule: MaterialChainSchedule,
    cotangents: ArrayLike,
    /,
) -> Array:
    """Apply the algebraic transpose of the scheduled extensive-content action."""
    if not isinstance(schedule, MaterialChainSchedule):
        raise TypeError("schedule must be a MaterialChainSchedule.")
    return _map_action(schedule, jnp.asarray(cotangents), transpose=True)


@final
class MaterialChainConservativeRemap(StrictModule, NonTrainableState):
    """Material-following FV content transfer through one scheduled material chain.

    The chain moves extensive content along the accepted material maps; cell
    averages are converted through the endpoint FV cell volumes. ``source_rows``
    and ``target_rows`` place the chain endpoint cell orders into the FV source
    and target cell orders. ``source_coverage_defects`` is the per-source-cell
    partition-of-unity defect of the composed extensive action, the evidence that
    every unit of source content reaches the target exactly once.
    """

    schedule: MaterialChainSchedule
    source_rows: Array
    target_rows: Array
    source_volumes: Array
    target_volumes: Array
    source_coverage_defects: Array
    coverage_tolerance: float = eqx.field(static=True)
    chain_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        schedule: MaterialChainSchedule,
        source_rows: ArrayLike,
        target_rows: ArrayLike,
        source_volumes: ArrayLike,
        target_volumes: ArrayLike,
        /,
        *,
        chain_id: str,
    ) -> None:
        if not isinstance(schedule, MaterialChainSchedule):
            raise TypeError("schedule must be a MaterialChainSchedule.")
        sources = np.asarray(source_rows, dtype=np.int64)
        targets = np.asarray(target_rows, dtype=np.int64)
        source_volume = np.asarray(source_volumes, dtype=np.float64)
        target_volume = np.asarray(target_volumes, dtype=np.float64)
        if (
            sources.shape != (schedule.source_count,)
            or targets.shape != (schedule.target_count,)
            or source_volume.shape != (schedule.source_count,)
            or target_volume.shape != (schedule.target_count,)
            or not np.array_equal(np.sort(sources), np.arange(sources.size))
            or not np.array_equal(np.sort(targets), np.arange(targets.size))
            or np.any(~np.isfinite(source_volume))
            or np.any(~np.isfinite(target_volume))
            or np.any(source_volume <= 0.0)
            or np.any(target_volume <= 0.0)
            or not isinstance(chain_id, str)
            or not chain_id
        ):
            raise ValueError(
                "Material-chain remap rows and volumes must bind the exact FV endpoints."
            )
        outflow = np.asarray(
            material_chain_transpose(
                schedule, jnp.ones((schedule.target_count,), dtype=jnp.float64)
            ),
            dtype=np.float64,
        )
        defects = outflow - 1.0
        # Each composed edit rounds once per binary64 product and accumulation.
        tolerance = float(
            8.0
            * schedule.map_count
            * max(1, schedule.maximum_edges)
            * np.finfo(np.float64).eps
        )
        self.schedule = schedule
        self.source_rows = jnp.asarray(sources)
        self.target_rows = jnp.asarray(targets)
        self.source_volumes = jnp.asarray(source_volume)
        self.target_volumes = jnp.asarray(target_volume)
        self.source_coverage_defects = jnp.asarray(defects)
        self.coverage_tolerance = tolerance
        self.chain_id = chain_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "material-chain-conservative-remap",
                "schedule": schedule.schedule_id,
                "chain": chain_id,
                "source_rows": sources,
                "target_rows": targets,
                "source_volumes": source_volume,
                "target_volumes": target_volume,
            }
        )

    @property
    def coverage_error(self) -> Array:
        """Largest partition-of-unity defect of the composed extensive action."""
        return jnp.max(jnp.abs(self.source_coverage_defects))

    @property
    def coverage_passed(self) -> Array:
        return self.coverage_error <= self.coverage_tolerance

    def apply_content(self, source_content: ArrayLike, /) -> Array:
        """Move extensive content from FV source rows to FV target rows."""
        values = jnp.asarray(source_content)
        if values.ndim == 0 or values.shape[0] != self.source_volumes.size:
            raise ValueError("source_content must begin with the source cell count.")
        moved = material_chain_apply(self.schedule, values[self.source_rows])
        return (
            jnp.zeros((self.target_volumes.size, *values.shape[1:]), dtype=moved.dtype)
            .at[self.target_rows]
            .set(moved)
        )

    def apply(self, source_cell_averages: ArrayLike, /) -> Array:
        """Transfer material-following cell averages through extensive content."""
        values = jnp.asarray(source_cell_averages)
        if values.ndim == 0 or values.shape[0] != self.source_volumes.size:
            raise ValueError("source_cell_averages must begin with source cell count.")
        trailing = (1,) * (values.ndim - 1)
        content = values * self.source_volumes.astype(values.dtype).reshape(
            (-1, *trailing)
        )
        return self.apply_content(content) / self.target_volumes.astype(
            values.dtype
        ).reshape((-1, *trailing))

    def conservation_defect(
        self, source_cell_averages: ArrayLike, target_cell_averages: ArrayLike, /
    ) -> Array:
        """Compensated target-minus-source total content per field channel."""
        source = jnp.asarray(source_cell_averages)
        target = jnp.asarray(target_cell_averages)
        if source.ndim == 0 or source.shape[0] != self.source_volumes.size:
            raise ValueError("source_cell_averages must begin with source cell count.")
        if target.ndim == 0 or target.shape[0] != self.target_volumes.size:
            raise ValueError("target_cell_averages must begin with target cell count.")
        source_terms = source * self.source_volumes.astype(source.dtype).reshape(
            (-1,) + (1,) * (source.ndim - 1)
        )
        target_terms = target * self.target_volumes.astype(target.dtype).reshape(
            (-1,) + (1,) * (target.ndim - 1)
        )
        return compensated_sum_chunks(
            (target_terms, -source_terms), output_ndim=source.ndim - 1
        )


def prepare_material_chain_conservative_remap(
    chain: VolumeMaterialChain,
    source_cell_ids: ArrayLike,
    target_cell_ids: ArrayLike,
    source_volumes: ArrayLike,
    target_volumes: ArrayLike,
    /,
) -> MaterialChainConservativeRemap:
    """Bind one accepted material chain to FV source/target cell orders."""
    if not isinstance(chain, VolumeMaterialChain):
        raise TypeError("chain must be a VolumeMaterialChain.")
    if not chain.maps:
        raise ValueError("A material-chain remap requires at least one material map.")
    chain_sources = _mesh_cell_ids(chain.maps[0].source_mesh)
    chain_targets = _mesh_cell_ids(chain.maps[-1].target_mesh)
    fv_sources = np.asarray(source_cell_ids, dtype=np.int64)
    fv_targets = np.asarray(target_cell_ids, dtype=np.int64)
    source_row_by_id = {int(value): row for row, value in enumerate(fv_sources)}
    target_row_by_id = {int(value): row for row, value in enumerate(fv_targets)}
    if (
        fv_sources.ndim != 1
        or fv_targets.ndim != 1
        or len(source_row_by_id) != fv_sources.size
        or len(target_row_by_id) != fv_targets.size
        or set(source_row_by_id) != set(chain_sources.tolist())
        or set(target_row_by_id) != set(chain_targets.tolist())
    ):
        raise ValueError(
            "Material-chain remap cells differ from the chain endpoint meshes."
        )
    schedule = prepare_material_chain_schedule(chain, chain_sources, chain_targets)
    return MaterialChainConservativeRemap(
        schedule,
        np.asarray([source_row_by_id[int(value)] for value in chain_sources]),
        np.asarray([target_row_by_id[int(value)] for value in chain_targets]),
        np.asarray(source_volumes, dtype=np.float64),
        np.asarray(target_volumes, dtype=np.float64),
        chain_id=chain.chain_id,
    )


__all__ = [
    "MaterialChainConservativeRemap",
    "material_chain_apply",
    "material_chain_transpose",
    "prepare_material_chain_conservative_remap",
]
