#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._compressible_contracts import CompressibleKineticPopulationState
from ._compressible_rules import CompressibleVelocityRule


KineticStorageLayout = Literal[
    "double-buffer-aos", "double-buffer-soa", "q-blocked", "aa"
]


class ScaledPopulationField(StrictModule):
    mantissa: Array
    scale: Array
    original_dtype: str = eqx.field(static=True)


class CompressibleKineticPrecisionPolicy(StrictModule, NonTrainableState):
    storage_dtype: str = eqx.field(static=True)
    compute_dtype: str = eqx.field(static=True)
    accumulation_dtype: str = eqx.field(static=True)
    dual_dtype: str = eqx.field(static=True)
    block_size: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        storage_dtype: str = "float64",
        compute_dtype: str = "float64",
        accumulation_dtype: str = "float64",
        dual_dtype: str = "float64",
        block_size: int = 256,
    ):
        supported = ("float16", "bfloat16", "float32", "float64")
        if storage_dtype not in supported or dual_dtype not in supported:
            raise ValueError("Unsupported kinetic storage or dual dtype.")
        if compute_dtype not in ("float32", "float64") or accumulation_dtype not in (
            "float32",
            "float64",
        ):
            raise ValueError("Kinetic compute and accumulation must use float32/float64.")
        if np.dtype(compute_dtype).itemsize < np.dtype("float32").itemsize:
            raise ValueError("Kinetic compute precision cannot be below float32.")
        if np.dtype(accumulation_dtype).itemsize < np.dtype(compute_dtype).itemsize:
            raise ValueError("Accumulation precision cannot be below compute precision.")
        size = int(block_size)
        if size < 1:
            raise ValueError("block_size must be positive.")
        self.storage_dtype = storage_dtype
        self.compute_dtype = compute_dtype
        self.accumulation_dtype = accumulation_dtype
        self.dual_dtype = dual_dtype
        self.block_size = size
        self.policy_id = canonical_fingerprint(
            {
                "kind": "compressible-kinetic-precision",
                "storage_dtype": storage_dtype,
                "compute_dtype": compute_dtype,
                "accumulation_dtype": accumulation_dtype,
                "dual_dtype": dual_dtype,
                "block_size": size,
            }
        )

    def encode(self, populations: ArrayLike, /) -> ScaledPopulationField:
        values = jnp.asarray(populations)
        if jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("Kinetic populations must be real-valued.")
        flat = values.reshape((-1, values.shape[-1]))
        padding = (-flat.shape[0]) % self.block_size
        padded = jnp.pad(flat, ((0, padding), (0, 0)))
        blocks = padded.reshape((-1, self.block_size, values.shape[-1]))
        scale = jnp.max(jnp.abs(blocks), axis=(1, 2))
        scale = jnp.where(scale > 0.0, scale, 1.0).astype(self.compute_dtype)
        mantissa = (blocks / scale[:, None, None]).astype(self.storage_dtype)
        return ScaledPopulationField(mantissa, scale, str(values.dtype))

    def decode(
        self,
        encoded: ScaledPopulationField,
        shape: tuple[int, ...],
        /,
    ) -> Array:
        if not isinstance(encoded, ScaledPopulationField):
            raise TypeError("encoded must be a ScaledPopulationField.")
        if not shape or shape[-1] != encoded.mantissa.shape[-1]:
            raise ValueError("Requested decoded shape is incompatible.")
        values = (
            encoded.mantissa.astype(self.compute_dtype) * encoded.scale[:, None, None]
        )
        size = int(np.prod(shape[:-1]))
        return values.reshape((-1, shape[-1]))[:size].reshape(shape)


class KineticStoragePlan(StrictModule, NonTrainableState):
    layout: KineticStorageLayout = eqx.field(static=True)
    q_block_size: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        layout: KineticStorageLayout = "double-buffer-soa",
        /,
        *,
        q_block_size: int = 16,
    ):
        if layout not in ("double-buffer-aos", "double-buffer-soa", "q-blocked", "aa"):
            raise ValueError(f"Unknown kinetic storage layout {layout!r}.")
        block = int(q_block_size)
        if block < 1:
            raise ValueError("q_block_size must be positive.")
        self.layout = layout
        self.q_block_size = block
        self.plan_id = canonical_fingerprint(
            {"kind": "kinetic-storage", "layout": layout, "q_block_size": block}
        )

    @property
    def state_multiplier(self) -> int:
        return 1 if self.layout == "aa" else 2


class KineticWorksetPlan(StrictModule, NonTrainableState):
    cell_batch_size: int = eqx.field(static=True)
    maximum_scratch_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, cell_batch_size: int = 4096, maximum_scratch_bytes: int = 1 << 30):
        batch = int(cell_batch_size)
        scratch = int(maximum_scratch_bytes)
        if batch < 1 or scratch < 1:
            raise ValueError("Workset batch and scratch limits must be positive.")
        self.cell_batch_size = batch
        self.maximum_scratch_bytes = scratch
        self.plan_id = canonical_fingerprint(
            {
                "kind": "kinetic-workset",
                "cell_batch_size": batch,
                "maximum_scratch_bytes": scratch,
            }
        )

    def require(self, *, scalars_per_cell: int, dtype: np.dtype | str) -> int:
        scalar_count = int(scalars_per_cell)
        if scalar_count < 1:
            raise ValueError("scalars_per_cell must be positive.")
        required = self.cell_batch_size * scalar_count * np.dtype(dtype).itemsize
        if required > self.maximum_scratch_bytes:
            raise MemoryError(
                f"Kinetic workset needs {required} scratch bytes, exceeding "
                f"{self.maximum_scratch_bytes}."
            )
        return required


class IntegerLatticeTransportEvidence(StrictModule):
    mass_defect: Array
    minimum_population: Array
    parity: Array
    finite: Array
    successful: Array


class IntegerLatticeTransportPlan(StrictModule, NonTrainableState):
    rule: CompressibleVelocityRule
    grid_shape: tuple[int, ...] = eqx.field(static=True)
    periodic_axes: tuple[bool, ...] = eqx.field(static=True)
    storage: KineticStoragePlan
    transport_id: str = eqx.field(static=True)

    def __init__(
        self,
        rule: CompressibleVelocityRule,
        grid_shape: tuple[int, ...],
        /,
        *,
        periodic_axes: tuple[bool, ...] | None = None,
        storage: KineticStoragePlan | None = None,
    ):
        if not isinstance(rule, CompressibleVelocityRule):
            raise TypeError("rule must be a CompressibleVelocityRule.")
        if not rule.exact_streaming:
            raise ValueError(
                "IntegerLatticeTransportPlan requires exact-streaming velocities."
            )
        shape = tuple(int(value) for value in grid_shape)
        if len(shape) != rule.dimension or any(value < 1 for value in shape):
            raise ValueError("grid_shape must contain one positive extent per dimension.")
        if any(
            shape[axis] <= 2 * rule.maximum_reach[axis] for axis in range(rule.dimension)
        ):
            raise ValueError("Grid extents must exceed twice the maximum lattice reach.")
        periodic = (
            (True,) * rule.dimension if periodic_axes is None else tuple(periodic_axes)
        )
        if len(periodic) != rule.dimension or not all(periodic):
            raise ValueError("This transport plan currently requires periodic axes.")
        selected_storage = KineticStoragePlan() if storage is None else storage
        if not isinstance(selected_storage, KineticStoragePlan):
            raise TypeError("storage must be a KineticStoragePlan.")
        self.rule = rule
        self.grid_shape = shape
        self.periodic_axes = periodic
        self.storage = selected_storage
        self.transport_id = canonical_fingerprint(
            {
                "kind": "integer-lattice-transport",
                "rule": rule.rule_id,
                "grid_shape": list(shape),
                "periodic_axes": list(periodic),
                "storage": selected_storage.plan_id,
            }
        )

    def stream(
        self,
        state: CompressibleKineticPopulationState,
        /,
        *,
        parity: ArrayLike = 0,
    ) -> tuple[CompressibleKineticPopulationState, IntegerLatticeTransportEvidence]:
        if state.spatial_shape != self.grid_shape:
            raise ValueError("State spatial shape does not match the transport grid.")
        velocities = np.asarray(self.rule.velocities, dtype=np.int32)
        streamed_fields = []
        for populations in state.populations:
            directions = tuple(
                jnp.roll(
                    populations[..., direction],
                    shift=tuple(int(value) for value in velocity),
                    axis=tuple(range(self.rule.dimension)),
                )
                for direction, velocity in enumerate(velocities)
            )
            streamed_fields.append(jnp.stack(directions, axis=-1))
        candidate = CompressibleKineticPopulationState(
            tuple(streamed_fields),
            state.equilibrium_dual,
            state.stabilizer,
            state.frame_velocity,
            state.frame_temperature_scale,
            state.layout,
        )
        old_mass = jnp.sum(state.population("particle"))
        new_mass = jnp.sum(candidate.population("particle"))
        minimum = jnp.min(
            jnp.stack(tuple(jnp.min(value) for value in streamed_fields), axis=0)
        )
        finite = jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in streamed_fields))
        )
        defect = new_mass - old_mass
        tolerance = (
            256.0 * jnp.finfo(old_mass.dtype).eps * jnp.maximum(jnp.abs(old_mass), 1.0)
        )
        successful = finite & (minimum > 0.0) & (jnp.abs(defect) <= tolerance)
        accepted = CompressibleKineticPopulationState(
            tuple(
                jnp.where(successful, new_value, old_value)
                for new_value, old_value in zip(
                    streamed_fields, state.populations, strict=True
                )
            ),
            state.equilibrium_dual,
            state.stabilizer,
            state.frame_velocity,
            state.frame_temperature_scale,
            state.layout,
        )
        return accepted, IntegerLatticeTransportEvidence(
            mass_defect=defect,
            minimum_population=minimum,
            parity=(jnp.asarray(parity, dtype=jnp.int32) + 1) % 2,
            finite=finite,
            successful=successful,
        )


class KineticVelocityPartitionEvidence(StrictModule):
    covered_populations: Array
    duplicate_populations: Array
    reconstruction_defect: Array
    successful: Array


class KineticVelocityPartitionPlan(StrictModule, NonTrainableState):
    """Static velocity-axis partition for Q-sharded kinetic execution."""

    rule: CompressibleVelocityRule
    shard_slices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    shard_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, rule: CompressibleVelocityRule, shard_count: int, /):
        if not isinstance(rule, CompressibleVelocityRule):
            raise TypeError("rule must be a CompressibleVelocityRule.")
        count = int(shard_count)
        if count < 1 or count > rule.population_count:
            raise ValueError("shard_count must lie in [1, population_count].")
        base, remainder = divmod(rule.population_count, count)
        slices = []
        start = 0
        for shard in range(count):
            width = base + (1 if shard < remainder else 0)
            slices.append((start, start + width))
            start += width
        self.rule = rule
        self.shard_slices = tuple(slices)
        self.shard_count = count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "kinetic-velocity-partition",
                "rule": rule.rule_id,
                "shard_slices": [list(value) for value in slices],
            }
        )

    def partition(self, populations: ArrayLike, /) -> tuple[Array, ...]:
        values = jnp.asarray(populations)
        if values.shape[-1] != self.rule.population_count:
            raise ValueError("Population array does not match the partition rule.")
        return tuple(values[..., start:stop] for start, stop in self.shard_slices)

    def assemble(
        self, shards: tuple[ArrayLike, ...], /
    ) -> tuple[Array, KineticVelocityPartitionEvidence]:
        values = tuple(jnp.asarray(shard) for shard in shards)
        if len(values) != self.shard_count:
            raise ValueError("Shard tuple does not match shard_count.")
        prefix = values[0].shape[:-1]
        for value, (start, stop) in zip(values, self.shard_slices, strict=True):
            if value.shape != prefix + (stop - start,):
                raise ValueError("Velocity shard shape does not match its partition.")
        assembled = jnp.concatenate(values, axis=-1)
        repartitioned = self.partition(assembled)
        defect = jnp.max(
            jnp.stack(
                tuple(
                    jnp.max(jnp.abs(actual - expected))
                    for actual, expected in zip(repartitioned, values, strict=True)
                )
            )
        )
        covered = sum(stop - start for start, stop in self.shard_slices)
        successful = (
            jnp.all(jnp.isfinite(assembled))
            & (covered == self.rule.population_count)
            & (defect == 0.0)
        )
        return assembled, KineticVelocityPartitionEvidence(
            covered_populations=jnp.asarray(covered, dtype=jnp.int32),
            duplicate_populations=jnp.asarray(0, dtype=jnp.int32),
            reconstruction_defect=defect,
            successful=successful,
        )

    def reduced_moments(
        self,
        shards: tuple[ArrayLike, ...],
        features: ArrayLike,
        /,
    ) -> Array:
        feature_values = jnp.asarray(features)
        if feature_values.shape[0] != self.rule.population_count:
            raise ValueError("features must begin with population_count.")
        values = tuple(jnp.asarray(shard) for shard in shards)
        partials = tuple(
            value @ feature_values[start:stop]
            for value, (start, stop) in zip(values, self.shard_slices, strict=True)
        )
        return sum(partials)


__all__ = [
    "CompressibleKineticPrecisionPolicy",
    "IntegerLatticeTransportEvidence",
    "IntegerLatticeTransportPlan",
    "KineticVelocityPartitionEvidence",
    "KineticVelocityPartitionPlan",
    "KineticStorageLayout",
    "KineticStoragePlan",
    "KineticWorksetPlan",
    "ScaledPopulationField",
]
